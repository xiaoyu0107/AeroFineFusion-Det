import torch
import torch.nn as nn
import torch.nn.functional as F
# from .conv import *

# 先把 Conv 和 LSKA 引入
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def autopad(k, p=None):
    # kernel, padding
    if p is None:
        p = k // 2
    return p

class MSLSKABlock(nn.Module):
    """
    改进版 LSKA，适合小目标检测：
    - 并行多尺度 (3x3, 7x7)
    - Sigmoid 门控
    - 残差融合
    - Bottleneck 降维
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        # reduction降维比例系数
        hidden_dim = max(dim // reduction, 8)  # 防止过小

        # bottleneck 降维 -> 升维
        self.conv_reduce = nn.Conv2d(dim, hidden_dim, 1)
        
        # branch1: 小感受野 3x3
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1)  # 通道混合
        )
        
        # branch2: 大感受野 7x7 (由1x7+7x1分解得到)
        self.branch7x7 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1,7), padding=(0,3), groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(7,1), padding=(3,0), groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1)  # 通道混合
        )

        # 通道融合
        self.conv_fuse = nn.Conv2d(hidden_dim*2, dim, 1)
        
        # # 升维恢复
        # self.conv_expand = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x

        # bottleneck 降维
        x_reduced = self.conv_reduce(x)

        # 并行两个分支
        out3 = self.branch3x3(x_reduced)
        out7 = self.branch7x7(x_reduced)

        # concat + 融合
        attn = torch.cat([out3, out7], dim=1)
        attn = self.conv_fuse(attn)
        # # 升维
        # attn = self.conv_expand(attn)

        # sigmoid 门控
        attn = torch.sigmoid(attn)

        # 残差式输出
        return u + u * attn

class MSSPPFLKA(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher + LSKA SmallObj."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        self.lska = MSLSKABlock(c_ * 4)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        # 在拼接后的 4*c_ 通道上做 LSKA 注意力
        return self.cv2(self.lska(torch.cat((x, y1, y2, self.m(y2)), 1)))



def count_parameters(model):
    """
    统计模型参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    return total_params, trainable_params
# ========== 测试 ==========
if __name__ == "__main__":
    x = torch.randn(1, 256, 80, 80)  # 假设 FPN 的 P3 层
    net = MSSPPFLKA(256, 256)         # 输入 256，输出 256
    y = net(x)
    print(y.shape)  # (1, 256, 80, 80)
    count_parameters(net)
