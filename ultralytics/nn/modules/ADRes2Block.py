import torch
import torch.nn as nn
import torch.nn.functional as F


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

# class ADRes2Block(nn.Module):
#     """稳定版 ADRes2Block，用于 backbone 替换 C2f"""
#     def __init__(self, in_channels, out_channels, scales=4, reduction=8, stride=1, kernel_num=2):
#         super().__init__()
#         assert out_channels % scales == 0
#         self.scales = scales
#         self.width = out_channels // scales

#         self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
#         self.norm1 = nn.GroupNorm(1, out_channels)

#         self.convs = nn.ModuleList([
#             FSAODConv2d(self.width, self.width, 3, stride, kernel_num=kernel_num, reduction=reduction)
#             for _ in range(scales - 1)
#         ])
#         self.norms = nn.ModuleList([nn.GroupNorm(1, self.width) for _ in range(scales - 1)])

#         self.conv3 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
#         self.norm3 = nn.GroupNorm(1, out_channels)

#         self.shortcut = (
#             nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
#                 nn.GroupNorm(1, out_channels)
#             ) if in_channels != out_channels else nn.Identity()
#         )

#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         out = self.relu(self.norm1(self.conv1(x)))
#         spx = torch.split(out, self.width, 1)
#         outputs = []
#         for i in range(self.scales):
#             if i == 0:
#                 outputs.append(spx[i])
#             else:
#                 y = spx[i] + outputs[i - 1]
#                 y = self.relu(self.norms[i - 1](self.convs[i - 1](y)))
#                 outputs.append(y)
#         out = torch.cat(outputs, 1)
#         out = self.norm3(self.conv3(out))
#         out = out + self.shortcut(x)
#         return self.relu(out)


class ADRes2Block(nn.Module):
    """Attention-Dynamic Res2Net Block (ADRes2Block)
    多尺度残差 + 动态注意力卷积块，用于小目标检测增强
    """
    def __init__(self, in_channels, out_channels, scales=4, reduction=16, stride=1, kernel_num=4):
        super().__init__()
        assert out_channels % scales == 0
        self.scales = scales
        self.width = out_channels // scales

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.convs = nn.ModuleList([
            FSAODConv2d(self.width, self.width, 3, stride,
                        kernel_num=kernel_num, reduction=reduction)
            for _ in range(scales - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(self.width) for _ in range(scales - 1)])

        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, 1)
        outputs = []
        for i in range(self.scales):
            if i == 0:
                outputs.append(spx[i])
            else:
                y = spx[i] + outputs[i - 1]
                y = self.relu(self.bns[i - 1](self.convs[i - 1](y)))
                outputs.append(y)
        out = torch.cat(outputs, 1)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)


# =============================================================
# ✅ 测试用例（可直接运行）
# =============================================================
if __name__ == "__main__":
    print("🔍 Testing ADRes2Block Module...")

    # 设置随机种子确保可复现
    torch.manual_seed(0)

    # 模拟输入: batch_size=1, channels=64, feature map = 64x64
    x = torch.randn(1, 64, 64, 64)

    # 实例化模块
    model = ADRes2Block(in_channels=64, out_channels=128, scales=4, reduction=16, stride=1, kernel_num=4)

    # 打印模型结构
    print(model)

    # 前向传播
    y = model(x)

    print(f"Input shape : {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")

    # 简单梯度测试
    y.mean().backward()
    print("✅ Forward & backward pass success!")
