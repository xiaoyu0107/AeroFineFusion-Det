import torch
import torch.nn as nn
import torch.nn.functional as F


# ======== 基础组件 ========

def autopad(k, p=None):  # 自动padding
    if p is None:
        p = k // 2
    return p


class Conv(nn.Module):
    """标准卷积块：Conv + BN + SiLU"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ======== 双尺度大核注意力模块 ========

class MSLABlock(nn.Module):
    """
    Multi-Scale Large Kernel Attention (3x3 + 7x7)
    特点：
    - 仅两个分支：3×3（局部特征）+ 7×7（大感受野）
    - 深度卷积 + 逐点卷积结构
    - 通道融合 + Sigmoid 注意力
    - 残差式输出
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden_dim = max(dim // reduction, 8)

        # Bottleneck 降维
        self.conv_reduce = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(hidden_dim)

        # 分支1：3×3 深度卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        # 分支2：7×7 深度卷积（大感受野）
        self.branch7 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 7, padding=3, groups=hidden_dim, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        # 融合与门控
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.act_gate = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = F.silu(self.bn_reduce(self.conv_reduce(x)))

        out3 = self.branch3(x)
        out7 = self.branch7(x)

        attn = torch.cat([out3, out7], dim=1)
        attn = self.fuse(attn)
        attn = self.act_gate(attn)

        # 残差式输出
        return identity + identity * attn


# ======== 改进版 SPPF + MSLA(3+7) ========

class MSSPPFLA(nn.Module):
    """SPPF + 双尺度(3x3+7x7) 大核注意力"""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.msla = MSLABlock(c_ * 4)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        x_cat = torch.cat((x, y1, y2, y3), 1)
        return self.cv2(self.msla(x_cat))

def count_parameters(model):
    """
    统计模型参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    return total_params, trainable_params
# ======== 测试 ========

if __name__ == "__main__":
    x = torch.randn(1, 256, 80, 80)
    net = MSSPPFLA(256, 256)
    y = net(x)
    print(y.shape)  # (1, 256, 80, 80)
    count_parameters(net)
