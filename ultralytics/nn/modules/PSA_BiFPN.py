import torch
import torch.nn as nn
import torch.nn.functional as F


class PSAFusion(nn.Module):
    """
    Pixel-wise Spatial Adaptive Fusion (PSAF)
    对每个像素位置计算来自不同层特征的权重并加权融合
    """
    def __init__(self, in_channels, num_levels=3):
        super(PSAFusion, self).__init__()
        self.num_levels = num_levels
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * num_levels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_levels, 1, bias=True)
        )

    def forward(self, features):
        # features: list of tensors [P3, P4, P5]
        size = features[0].shape[-2:]
        resized = [F.interpolate(f, size=size, mode='nearest') for f in features]
        concat = torch.cat(resized, dim=1)  # [B, C*num_levels, H, W]
        weight = self.attention(concat)     # [B, num_levels, H, W]
        weight = F.softmax(weight, dim=1)   # 对每个像素 softmax 权重

        fused = sum(weight[:, i:i+1, :, :] * resized[i] for i in range(self.num_levels))
        return fused


class ConvBNAct(nn.Module):
    """轻量卷积模块"""
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True):
        super(ConvBNAct, self).__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class PSA_BiFPN(nn.Module):
    """
    改进的 AWA-BiFPN：采用 PSAF（像素级空间自适应融合）代替 SE/Concat
    """
    def __init__(self, in_channels_list, out_channels):
        super(PSA_BiFPN, self).__init__()
        self.num_levels = len(in_channels_list)
        self.reduce_convs = nn.ModuleList([
            ConvBNAct(c, out_channels, 1, 1) for c in in_channels_list
        ])
        self.fusion = PSAFusion(out_channels, num_levels=self.num_levels)
        self.output_conv = ConvBNAct(out_channels, out_channels, 3, 1)

    def forward(self, features):
        # features: list of tensors [P3, P4, P5, ...]
        feats = [conv(f) for conv, f in zip(self.reduce_convs, features)]
        fused = self.fusion(feats)
        out = self.output_conv(fused)
        return out


# ✅ 测试模块功能
if __name__ == "__main__":
    # 模拟输入：三个尺度 P3, P4, P5
    P3 = torch.randn(1, 128, 80, 80)
    P4 = torch.randn(1, 256, 40, 40)
    P5 = torch.randn(1, 512, 20, 20)

    model = PSA_BiFPN(in_channels_list=[128, 256], out_channels=256)
    y = model([P3, P4, P5])
    inputs = ( [P3, P4, P5], )  # thop 只接受 tuple 输入
    import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format

# --- 你的 PSA_BiFPN 模块代码省略，可直接使用你提供的 --- #

if __name__ == "__main__":
    # 模拟输入：三个尺度 P3, P4, P5
    P3 = torch.randn(1, 128, 80, 80)
    P4 = torch.randn(1, 256, 40, 40)
    P5 = torch.randn(1, 512, 20, 20)

    model = PSA_BiFPN(in_channels_list=[128, 256, 512], out_channels=256)
    y = model([P3, P4, P5])
    print("Output shape:", y.shape)

    # --- 计算模型复杂度 ---
    inputs = ( [P3, P4, P5], )  # thop 只接受 tuple 输入
    flops, params = profile(model, inputs=inputs)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"Model parameters: {params}")
    print(f"Model FLOPs: {flops}")

    # y = model([P3, P4])
    print("Output shape:", y.shape)
