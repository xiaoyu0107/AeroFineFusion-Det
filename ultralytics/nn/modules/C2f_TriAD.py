import torch
import torch.nn as nn
import torch.nn.functional as F
# from ultralytics.nn.modules.conv import Conv


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

# ---------------------------------------------------------------------------
# 1️⃣ TriADConv: Tri-Attention Dynamic Convolution
#    （通道 + 空间注意力融合 + 多核动态卷积）
# ---------------------------------------------------------------------------
class TriADConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 groups=1, kernel_num=4, reduction=16):
        super().__init__()
        self.kernel_num = kernel_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride

        # hidden 通道：加下限 16，防止 in_channels 很小时出问题
        hidden = max(in_channels // reduction, 16)
        self.hidden = hidden

        # 多核动态权重: (K, C_out, C_in/groups, k, k)
        self.weight = nn.Parameter(
            torch.randn(kernel_num, out_channels, in_channels // groups, kernel_size, kernel_size)
        )

        # 通道注意力（SE）
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1, bias=False),
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
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, kernel_num, 1),
            nn.Softmax(dim=1)
        )

        # 融合系数 λ，用 raw 参数 + sigmoid 约束到 (0,1)
        # sigmoid(0)=0.5，与原先手动设为 0.5 的行为一致
        self.lambda_raw = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for k in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[k], mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        b, c, h, w = x.shape

        # -------- 通道 + 空间 融合注意力 --------
        ca = self.channel_att(x)          # (B, C, 1, 1)
        sa = self.spatial_att(x)          # (B, 1, H, W)

        # λ = sigmoid(lambda_raw) ∈ (0,1)
        alpha = torch.sigmoid(self.lambda_raw)
        sa_b = sa.expand_as(x)            # (B, C, H, W)
        attn = alpha * ca + (1.0 - alpha) * sa_b
        x = x * attn

        # -------- 核权重融合（按样本自适应）--------
        # kernel_att: (B, K, 1, 1, 1, 1)
        kernel_att = self.kernel_att(x).view(b, self.kernel_num, 1, 1, 1, 1)
        # self.weight: (K, C_out, C_in/groups, k, k) → (1, K, C_out, C_in/groups, k, k)
        fused_weight = torch.sum(kernel_att * self.weight.unsqueeze(0), dim=1)  # (B, C_out, C_in/groups, k, k)

        # -------- 动态卷积：batch 折进 groups --------
        # x_r: (1, B*C_in, H, W)
        x_r = x.view(1, -1, h, w)
        # fused_weight: (B*C_out, C_in/groups, k, k)
        fused_weight = fused_weight.view(
            -1, self.in_channels // self.groups, self.kernel_size, self.kernel_size
        )

        out = F.conv2d(
            x_r,
            weight=fused_weight,
            stride=self.stride,
            padding=self.kernel_size // 2,
            groups=self.groups * b
        )
        out = out.view(b, self.out_channels, out.shape[-2], out.shape[-1])
        return out


# ---------------------------------------------------------------------------
# 2️⃣ TriADBlock: 使用 TriADConv 的 Bottleneck
# ---------------------------------------------------------------------------
class TriADBlock(nn.Module):
    def __init__(self, c1, c2, shortcut=True, expansion=0.5, g=1, kernel_num=4):
        """
        c1: 输入通道
        c2: 输出通道
        expansion: 中间隐层比例
        g: 分组（与 Conv 一致）
        """
        super().__init__()
        hidden = int(c2 * expansion)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.triad = TriADConv(hidden, c2, kernel_size=3, stride=1,
                               groups=g, kernel_num=kernel_num)
        self.shortcut = shortcut and (c1 == c2)

    def forward(self, x):
        out = self.triad(self.cv1(x))
        return x + out if self.shortcut else out


# ---------------------------------------------------------------------------
# 3️⃣ C2f_TriAD: C2f 结构 + TriADBlock
# ---------------------------------------------------------------------------
class C2f_TriAD(nn.Module):
    """
    C2f_TriAD: C2f with Tri-Attention Dynamic Convolution Blocks
    用法与原生 C2f 类似：C2f_TriAD(c1, c2, n, shortcut, g, e, kernel_num)
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, kernel_num=4):
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, 2 * hidden, 1, 1)
        self.cv2 = Conv((2 + n) * hidden, c2, 1, 1)
        self.blocks = nn.ModuleList(
            TriADBlock(hidden, hidden, shortcut, expansion=1.0, g=g, kernel_num=kernel_num)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))   # [y1, y2]
        for block in self.blocks:
            y.append(block(y[-1]))
        return self.cv2(torch.cat(y, 1))


# ---------------- 简单测试 ----------------
if __name__ == "__main__":
    x = torch.randn(4, 128, 64, 64)
    m = C2f_TriAD(128, 128, n=2, shortcut=True, g=1, e=0.5, kernel_num=4)
    y = m(x)
    print("Input :", x.shape)
    print("Output:", y.shape)