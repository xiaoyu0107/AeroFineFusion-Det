import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


# ---------------------------------------------------------------------------
# 1️⃣ FSA-ODConv: 通道 + 空间注意力融合 + 核权重动态卷积
# ---------------------------------------------------------------------------
class FSAODConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 groups=1, kernel_num=4, reduction=16):
        super().__init__()
        self.kernel_num = kernel_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = kernel_size

        # 多核动态权重
        self.weight = nn.Parameter(
            torch.randn(kernel_num, out_channels, in_channels // groups, kernel_size, kernel_size)
        )

        # 通道注意力（SE）
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力（3×3 卷积）
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

        # 核注意力
        self.kernel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, kernel_num, 1),
            nn.Softmax(dim=1)
        )

        # 融合系数 λ
        self.lambda_fuse = nn.Parameter(torch.tensor(0.5))
        self._init_weights()

    def _init_weights(self):
        for k in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[k], mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        b, c, h, w = x.shape

        # 通道 + 空间融合注意力
        ca = self.channel_att(x)
        sa = self.spatial_att(x)
        attn = self.lambda_fuse * ca + (1 - self.lambda_fuse) * sa
        x = x * attn

        # 核权重融合
        kernel_att = self.kernel_att(x).view(b, self.kernel_num, 1, 1, 1, 1)
        fused_weight = torch.sum(kernel_att * self.weight.unsqueeze(0), dim=1)

        # 动态卷积
        x_r = x.view(1, -1, h, w)
        out = F.conv2d(
            x_r, weight=fused_weight.view(-1, self.in_channels // self.groups, self.kernel_size, self.kernel_size),
            stride=1, padding=self.kernel_size // 2, groups=self.groups * b
        )
        out = out.view(b, self.out_channels, h, w)
        return out


# ---------------------------------------------------------------------------
# 2️⃣ Bottleneck-FSA: 具有残差连接与FSA动态卷积的瓶颈结构
# ---------------------------------------------------------------------------
class Bottleneck_FSA(nn.Module):
    def __init__(self, c1, c2, shortcut=True, expansion=0.5, g=1, kernel_num=4):
        super().__init__()
        hidden = int(c2 * expansion)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.fsa_conv = FSAODConv2d(hidden, c2, kernel_size=3, kernel_num=kernel_num, groups=g)
        self.shortcut = shortcut and (c1 == c2)

    def forward(self, x):
        out = self.fsa_conv(self.cv1(x))
        return x + out if self.shortcut else out


# ---------------------------------------------------------------------------
# 3️⃣ C2f-FSA: 改进的 Cross-Stage Partial 结构 (含FSA动态瓶颈)
# ---------------------------------------------------------------------------
class C2f_FSA(nn.Module):
    """Enhanced C2f with FSA Bottlenecks and residual fusion."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, kernel_num=4):
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, 2 * hidden, 1, 1)
        self.cv2 = Conv((2 + n) * hidden, c2, 1, 1)
        self.blocks = nn.ModuleList(
            Bottleneck_FSA(hidden, hidden, shortcut, expansion=1.0, g=g, kernel_num=kernel_num) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for block in self.blocks:
            y.append(block(y[-1]))
        return self.cv2(torch.cat(y, 1))


