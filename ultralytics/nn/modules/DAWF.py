import torch
import torch.nn as nn
import torch.nn.functional as F


# class ELSA(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.q = nn.Conv2d(dim, dim, 1)
#         self.k = nn.Conv2d(dim, dim, 1)
#         self.v = nn.Conv2d(dim, dim, 1)
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depth-wise
#         self.out = nn.Conv2d(dim, dim, 1)

#     def forward(self, x):
#         q, k, v = self.q(x), self.k(x), self.v(x)
#         attn = (q * k).softmax(dim=-1)
#         x = v * attn
#         x = self.dwconv(x)
#         return self.out(x)

class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DAWF(nn.Module):
    def __init__(self, c1, c2=None, num_inputs=None, epsilon=1e-4, reduction=4, se_reduction=16):
        super().__init__()

        if isinstance(c1, (list, tuple)):
            self.in_channels = list(c1)
            self.num_inputs = len(self.in_channels)
        else:
            assert num_inputs is not None, "如果 c1 是单个整数，请指定 num_inputs"
            self.in_channels = [c1] * num_inputs
            self.num_inputs = num_inputs

        self.epsilon = epsilon
        total_c = sum(self.in_channels)
        self.out_channels = c2 or total_c
        self.mid_channels = max(self.out_channels // reduction, 16)

        #  添加 SE 注意力模块
        self.se_blocks = nn.ModuleList([
            SEBlock(c, reduction=se_reduction) for c in self.in_channels
        ])
        # self.elsa = ELSA(self.mid_channels)
        self.sa= SpatialAttention(3)
        # 通道对齐（Cin → mid_channels）
        self.align_convs = nn.ModuleList([
            nn.Conv2d(c, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
            for c in self.in_channels
        ])

        self.w = nn.Parameter(torch.ones(self.num_inputs, dtype=torch.float32), requires_grad=True)
        self.conv = nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.act = nn.SiLU()

    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple)), f"Expected list, got {type(inputs)}"
        num_inputs = len(inputs)
        assert num_inputs == self.num_inputs, f"Expected {self.num_inputs} inputs, but got {num_inputs}"

        target_size = inputs[0].shape[2:]

        resized = []
        for i, x in enumerate(inputs):
            x = self.se_blocks[i](x)                  # 🔹 加入 SE 注意力
            y = self.align_convs[i](x)                # 1x1 conv → mid_channels
            if x.shape[2:] != target_size:
                y = F.interpolate(y, size=target_size, mode='nearest')
            resized.append(y)

        w = F.relu(self.w)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)

        fused_mid = 0
        for i in range(num_inputs):
            fused_mid += weight[i] * resized[i]
        
        out = self.conv(self.act(self.sa(fused_mid)))
        return out
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# def autopad(k, p=None, d=1):  # kernel, padding, dilation
#     """Pad to 'same' shape outputs."""
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p

# class Conv(nn.Module):
#     """
#     Standard convolution module with batch normalization and activation.

#     Attributes:
#         conv (nn.Conv2d): Convolutional layer.
#         bn (nn.BatchNorm2d): Batch normalization layer.
#         act (nn.Module): Activation function layer.
#         default_act (nn.Module): Default activation function (SiLU).
#     """

#     default_act = nn.SiLU()  # default activation

#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         """
#         Initialize Conv layer with given parameters.

#         Args:
#             c1 (int): Number of input channels.
#             c2 (int): Number of output channels.
#             k (int): Kernel size.
#             s (int): Stride.
#             p (int, optional): Padding.
#             g (int): Groups.
#             d (int): Dilation.
#             act (bool | nn.Module): Activation function.
#         """
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         """
#         Apply convolution, batch normalization and activation to input tensor.

#         Args:
#             x (torch.Tensor): Input tensor.

#         Returns:
#             (torch.Tensor): Output tensor.
#         """
#         return self.act(self.bn(self.conv(x)))

#     def forward_fuse(self, x):
#         """
#         Apply convolution and activation without batch normalization.

#         Args:
#             x (torch.Tensor): Input tensor.

#         Returns:
#             (torch.Tensor): Output tensor.
#         """
#         return self.act(self.conv(x))


# class SpatialAttention(nn.Module):
#     """Spatial-attention module."""

#     def __init__(self, kernel_size=7):
#         """Initialize Spatial-attention module with kernel size argument."""
#         super().__init__()
#         assert kernel_size in (3, 7), "kernel size must be 3 or 7"
#         padding = 3 if kernel_size == 7 else 1
#         self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.act = nn.Sigmoid()

#     def forward(self, x):
#         """Apply spatial attention on input for feature recalibration."""
#         avg = torch.mean(x, 1, keepdim=True)
#         mx = torch.max(x, 1, keepdim=True)[0]
#         attn = self.act(self.cv1(torch.cat([avg, mx], 1)))
#         return x * attn


# class SEBlock(nn.Module):
#     """Squeeze-and-Excitation block"""
#     def __init__(self, channels, reduction=16):
#         super(SEBlock, self).__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction, bias=False),
#             nn.SiLU(),
#             nn.Linear(channels // reduction, channels, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y


# class LBAF(nn.Module):
#     """
#     Lightweight Bi-directional Attention Fusion (FPS-oriented, accuracy-speed balanced)

#     相比原版 LBAF：
#     - 去掉 SEBlock 和 SpatialAttention（减少注意力开销）
#     - 保留多输入加权融合（标量权重 + 归一化）
#     - 使用你自定义的 Conv（Conv+BN+SiLU）做通道对齐和输出
#     - 上采样用最近邻插值，下采样用 stride=2 的 Conv
#     """

#     def __init__(self, c1, c2=None, num_inputs=None,
#                  epsilon=1e-4, reduction=4):
#         """
#         c1: int or list[int]，各输入分支的通道数
#         c2: 输出通道数，不填则为所有输入通道数之和
#         num_inputs: 如果 c1 是单个 int，需要指定输入个数
#         reduction: 中间通道压缩比例，越大越省算力（8 比 4 更轻量）
#         """
#         super().__init__()

#         # 处理输入通道列表
#         if isinstance(c1, (list, tuple)):
#             self.in_channels = list(c1)
#             self.num_inputs = len(self.in_channels)
#         else:
#             assert num_inputs is not None, "如果 c1 是单个整数，请指定 num_inputs"
#             self.in_channels = [c1] * num_inputs
#             self.num_inputs = num_inputs

#         self.epsilon = epsilon
#         total_c = sum(self.in_channels)
#         self.out_channels = c2 or total_c

#         # 中间通道数：稍微压一压，比原来的 reduction=4 更省 FLOPs
#         self.mid_channels = max(self.out_channels // reduction, 16)

#         # 通道对齐（Cin → mid_channels），用你的 Conv（带 BN + SiLU）
#         self.align_convs = nn.ModuleList([
#             Conv(c, self.mid_channels, k=1, s=1, p=0, g=1, d=1, act=True)
#             for c in self.in_channels
#         ])

#         # 下采样 conv：用于把比 target 尺寸更大的 feature 往下采样
#         # 这里仍然用 depthwise conv（groups=mid_channels），act=False 减一点算力
#         self.downsample_convs = nn.ModuleList([
#             Conv(self.mid_channels, self.mid_channels,
#                  k=3, s=2, p=None, g=self.mid_channels, d=1, act=False)
#             for _ in self.in_channels
#         ])

#         # 多分支标量权重（可学习），长度 = num_inputs
#         self.w = nn.Parameter(torch.ones(self.num_inputs, dtype=torch.float32), requires_grad=True)

#         # 输出 1×1 conv：mid_channels → out_channels
#         self.out_conv = Conv(self.mid_channels, self.out_channels,
#                              k=1, s=1, p=0, g=1, d=1, act=True)

#     @staticmethod
#     def _resize_to_target(x, target_size, down_conv):
#         """
#         用 Upsample 和 Conv 实现的自动对齐：
#         - 如果 x 比 target 小：用最近邻上采样
#         - 如果 x 比 target 大：用 stride=2 的 Conv 下采样，直到不大于 target
#         （假设特征图之间大小多为 2 的倍数关系）
#         """
#         _, _, h, w = x.shape
#         th, tw = target_size

#         # 已经对齐
#         if (h, w) == (th, tw):
#             return x

#         # 上采样：直接插值到目标尺寸
#         if h <= th and w <= tw:
#             return F.interpolate(x, size=target_size, mode='nearest')

#         # 下采样：用 conv(stride=2) 逐步往下
#         while h > th or w > tw:
#             x = down_conv(x)
#             _, _, h, w = x.shape

#         # 如果不是刚好等于 target（比如尺度关系不是严格 2 的幂），
#         # 再做一次插值精调
#         if (h, w) != (th, tw):
#             x = F.interpolate(x, size=target_size, mode='nearest')
#         return x

#     def forward(self, inputs):
#         assert isinstance(inputs, (list, tuple)), f"Expected list/tuple, got {type(inputs)}"
#         num_inputs = len(inputs)
#         assert num_inputs == self.num_inputs, f"Expected {self.num_inputs} inputs, but got {num_inputs}"

#         # 以第一个输入分支的空间尺寸作为对齐目标
#         target_size = inputs[0].shape[2:]  # (H, W)

#         resized = []
#         for i, x in enumerate(inputs):
#             # 1) 通道对齐到 mid_channels（Conv 已含 BN + SiLU）
#             y = self.align_convs[i](x)

#             # 2) 用 upsample + conv 的策略对齐空间尺寸
#             y = self._resize_to_target(y, target_size, self.downsample_convs[i])

#             resized.append(y)

#         # 3) 标量权重归一化（类似 softmax，但更便宜）
#         w = F.relu(self.w)
#         weight = w / (torch.sum(w, dim=0) + self.epsilon)

#         # 4) 加权融合
#         fused_mid = 0
#         for i in range(num_inputs):
#             fused_mid = fused_mid + weight[i] * resized[i]

#         # 5) 输出 1×1 Conv（带 BN + SiLU）
#         out = self.out_conv(fused_mid)
#         return out
