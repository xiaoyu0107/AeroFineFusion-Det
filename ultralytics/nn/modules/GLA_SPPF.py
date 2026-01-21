import torch
from torch import nn
# from .conv import *


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

# class LSKA(nn.Module):
#     """
#     VisDrone 友好版 Large-Separable Kernel Attention:
#     - 保留大核 (水平+垂直+空洞) 结构
#     - 使用 soft residual attention: y = x * (1 + gamma * tanh(attn))
#       初始 gamma=0，等价恒等映射，避免一开始就“压死”小目标
#     """
#     def __init__(self, dim, k_size=23, reduction=4):
#         super().__init__()
#         self.k_size = k_size
#         mid = max(dim // reduction, 16)

#         # 1x1 降维
#         self.proj_in = nn.Conv2d(dim, mid, kernel_size=1, bias=False)
#         self.bn_in = nn.BatchNorm2d(mid)
#         self.act = nn.ReLU(inplace=True)

#         # 根据 k_size 选择基础核和空洞大核
#         if k_size in (7, 11):
#             base_ks = 3
#             spa_ks = 3 if k_size == 7 else 5
#             dil = 2
#         elif k_size in (23, 35, 41, 53):
#             base_ks = 5
#             spa_dict = {23: 7, 35: 11, 41: 13, 53: 17}
#             spa_ks = spa_dict[k_size]
#             dil = 3
#         else:
#             base_ks = 3
#             spa_ks = 3
#             dil = 1

#         pad_base = (base_ks - 1) // 2
#         pad_spa = dil * (spa_ks - 1) // 2

#         # depthwise 方向卷积
#         self.conv0h = nn.Conv2d(
#             mid, mid,
#             kernel_size=(1, base_ks),
#             stride=1,
#             padding=(0, pad_base),
#             groups=mid,
#             bias=False,
#         )
#         self.conv0v = nn.Conv2d(
#             mid, mid,
#             kernel_size=(base_ks, 1),
#             stride=1,
#             padding=(pad_base, 0),
#             groups=mid,
#             bias=False,
#         )

#         # 空洞大核（水平+垂直）
#         self.conv_spatial_h = nn.Conv2d(
#             mid, mid,
#             kernel_size=(1, spa_ks),
#             stride=1,
#             padding=(0, pad_spa),
#             dilation=(1, dil),
#             groups=mid,
#             bias=False,
#         )
#         self.conv_spatial_v = nn.Conv2d(
#             mid, mid,
#             kernel_size=(spa_ks, 1),
#             stride=1,
#             padding=(pad_spa, 0),
#             dilation=(dil, 1),
#             groups=mid,
#             bias=False,
#         )

#         self.bn_mid = nn.BatchNorm2d(mid)
#         self.proj_out = nn.Conv2d(mid, dim, kernel_size=1, bias=False)
#         self.bn_out = nn.BatchNorm2d(dim)

#         #  关键：soft residual attention 系数，初始为 0
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x):
#         identity = x

#         attn = self.act(self.bn_in(self.proj_in(x)))
#         attn = self.conv0h(attn)
#         attn = self.conv0v(attn)
#         attn = self.conv_spatial_h(attn)
#         attn = self.conv_spatial_v(attn)
#         attn = self.bn_mid(attn)
#         attn = self.bn_out(self.proj_out(self.act(attn)))

#         # [-1,1]，再用 1 + gamma * attn 做缩放
#         attn = torch.tanh(attn)
#         scale = 1 + self.gamma * attn

#         return identity * scale

class LSKA(nn.Module):
    def __init__(self, dim, k_size):
        super().__init__()

        self.k_size = k_size

        if k_size == 7:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, 2), groups=dim,
                                            dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0), groups=dim,
                                            dilation=2)
        elif k_size == 11:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, 4), groups=dim,
                                            dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), groups=dim,
                                            dilation=2)
        elif k_size == 23:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1, 1), padding=(0, 9), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1, 1), padding=(9, 0), groups=dim,
                                            dilation=3)
        elif k_size == 35:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1, 1), padding=(0, 15), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1, 1), padding=(15, 0), groups=dim,
                                            dilation=3)
        elif k_size == 41:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1, 1), padding=(0, 18), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1, 1), padding=(18, 0), groups=dim,
                                            dilation=3)
        elif k_size == 53:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1, 1), padding=(0, 24), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1, 1), padding=(24, 0), groups=dim,
                                            dilation=3)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn


class GLA_SPPF(nn.Module):
    """
    GLA-SPPF: Global–Local Aggregation Spatial Pyramid Pooling Fast

    - Branch 1: 标准 SPPF（局部 / 细节）
    - Branch 2: SPPF + LSKA（全局 / 大感受野）
    - 融合: concat 后 1x1 Conv
    - 可选残差: c1 == c2 时加回输入
    """
    def __init__(self, c1, c2, k=5,
                 k_lska=23,
                 use_res=True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.hidden = c1 // 2

        self.cv1 = Conv(c1, self.hidden, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        # Local branch
        self.cv_local = Conv(self.hidden * 4, c2, 1, 1)

        # Global branch (SPPF + LSKA)
        self.lska = LSKA(self.hidden * 4, k_size=k_lska)
        self.cv_global = Conv(self.hidden * 4, c2, 1, 1)

        # Fusion
        self.cv_fuse = Conv(c2 * 2, c2, 1, 1)

        self.use_res = use_res and (c1 == c2)

    def forward(self, x):
        identity = x

        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.m(y1)
        y3 = self.m(y2)
        feats = torch.cat((x1, y1, y2, y3), 1)

        local_out = self.cv_local(feats)

        global_feats = self.lska(feats)
        global_out = self.cv_global(global_feats)

        fused = torch.cat((local_out, global_out), 1)
        fused = self.cv_fuse(fused)

        if self.use_res:
            fused = fused + identity
        return fused

class DPSPPF(nn.Module):
    """
    DPSPPF: Dual-Path Spatial Pyramid Pooling Fast

    - Local branch: 标准 SPPF（金字塔池化，局部/多尺度）
    - Global branch: 直接对降维后的特征做 LSKA（大核/全局上下文）
    - 融合: concat 后 1x1 Conv
    - 可选残差: c1 == c2 时加回输入
    """
    def __init__(self, c1, c2, k=5,
                 k_lska=23,
                 use_res=True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.hidden = c1 // 2  # 中间通道 c_

        # 先做一次 1x1 降维，供两条分支共享
        self.cv1 = Conv(c1, self.hidden, 1, 1)

        # ---- Local（SPPF）分支 ----
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv_local = Conv(self.hidden * 4, c2, 1, 1)

        # ---- Global（LSKA）分支 ----
        self.lska = LSKA(self.hidden, k_size=k_lska)
        self.cv_global = Conv(self.hidden, c2, 1, 1)

        # ---- 融合 ----
        self.cv_fuse = Conv(c2 * 2, c2, 1, 1)

        self.use_res = use_res and (c1 == c2)

    def forward(self, x):
        identity = x
        x1 = self.cv1(x)                      # (B, hidden, H, W)

        # Local: SPPF
        y1 = self.m(x1)
        y2 = self.m(y1)
        y3 = self.m(y2)
        local_feats = torch.cat((x1, y1, y2, y3), dim=1)
        local_out = self.cv_local(local_feats)

        # Global: LSKA
        global_feats = self.lska(x1)
        global_out = self.cv_global(global_feats)

        # Fuse
        fused = torch.cat((local_out, global_out), dim=1)
        fused = self.cv_fuse(fused)

        if self.use_res:
            fused = fused + identity

        return fused




# ========== 简单自测 ==========
if __name__ == "__main__":
    x = torch.randn(2, 256, 40, 40)   # 假设 P5/32 这种尺度
    m = GLA_SPPF(256, 256, k=5, k_lska=23, use_res=True)
    y = m(x)
    print("Input :", x.shape)
    print("Output:", y.shape)

