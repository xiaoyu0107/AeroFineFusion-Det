import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules import Conv

class RKA(nn.Module):
    """
    Rotation Kernel Attention (旋转感知卷积)
    利用 Depthwise + Group Conv 模拟卷积核旋转响应。
    """
    def __init__(self, c, kernel_size=3, groups=4):
        super().__init__()
        self.groups = groups

        # 多方向卷积：模拟旋转方向（0°, 45°, 90°, 135°）
        self.convs = nn.ModuleList([
            nn.Conv2d(c, c, kernel_size, stride=1, padding=1, groups=c, bias=False)
            for _ in range(groups)
        ])

        # 旋转方向的融合（group conv 作为方向融合器）
        self.fuse = nn.Conv2d(c * groups, c, 1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        # 模拟不同旋转卷积响应
        feats = [conv(x) for conv in self.convs]
        y = torch.cat(feats, dim=1)
        y = self.fuse(y)
        return self.act(self.bn(y))


class DGAConv_RKA(nn.Module):
    """
    Dynamic Gated Aggregation Convolution + Rotation Kernel Attention
    适合YOLOv8高分辨率层（P2-P3）的小目标检测。
    """
    def __init__(self, c1, c2, k=3, s=1, reduction=4, d_rate=2, rka_groups=4):
        super().__init__()
        hidden = c2 // 4

        # 四方向卷积
        self.conv_h = Conv(c1, hidden, (1, k), s=s, p=(0, k//2))
        self.conv_v = Conv(c1, hidden, (k, 1), s=s, p=(k//2, 0))
        self.conv_d1 = Conv(c1, hidden, (k, k), s=s, p=d_rate, d=d_rate)
        self.conv_d2 = Conv(c1, hidden, (3, 3), s=s, p=1)

        # 动态门控
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, 4, 1),
            nn.Sigmoid()
        )

        # 聚合与旋转感知增强
        self.fuse = Conv(hidden * 4, c2, 1)
        self.rka = RKA(c2, kernel_size=3, groups=rka_groups)

        # 通道注意力
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2 // reduction, 1),
            nn.SiLU(),
            nn.Conv2d(c2 // reduction, c2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        g = self.gate(x)
        g1, g2, g3, g4 = torch.chunk(g, 4, dim=1)

        y1 = self.conv_h(x) * g1
        y2 = self.conv_v(x) * g2
        y3 = self.conv_d1(x) * g3
        y4 = self.conv_d2(x) * g4

        y = torch.cat([y1, y2, y3, y4], dim=1)
        y = self.fuse(y)
        y = self.rka(y)     # 🌀 加入旋转感知增强
        y = y * self.se(y)
        return y
