# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from ultralytics.nn.modules.conv import Conv


# # ========== FSA-ODConv: 融合空间 + 通道 + 核注意力 ==========
# class FSAODConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, kernel_num=4, reduction=0.0625):
#         super().__init__()
#         hidden = max(int(in_channels * reduction), 16)
#         self.kernel_size = kernel_size
#         self.kernel_num = kernel_num

#         # 注意力生成分支
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(in_channels, hidden, 1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.channel_fc = nn.Conv2d(hidden, in_channels, 1)
#         self.spatial_fc = nn.Conv2d(hidden, kernel_size * kernel_size, 1)
#         self.kernel_fc = nn.Conv2d(hidden, kernel_num, 1)

#         # 多核可学习卷积
#         self.weight = nn.Parameter(torch.randn(kernel_num, out_channels, in_channels, kernel_size, kernel_size))

#     def forward(self, x):
#         B, C, H, W = x.shape
#         att = self.relu(self.fc1(self.avgpool(x)))
#         ca = torch.sigmoid(self.channel_fc(att)).view(B, C, 1, 1)
#         sa = torch.sigmoid(self.spatial_fc(att)).view(B, 1, 1, 1, self.kernel_size, self.kernel_size)
#         ka = F.softmax(self.kernel_fc(att), dim=1).view(B, self.kernel_num, 1, 1, 1, 1)

#         x = x * ca
#         agg_w = (sa * ka * self.weight.unsqueeze(0)).sum(1)
#         y = F.conv2d(x, agg_w.view(-1, C, self.kernel_size, self.kernel_size),
#                      padding=self.kernel_size // 2, groups=B)
#         y = y.view(B, -1, H, W)
#         return y


# # ========== FSA-RGBlock: 门控残差融合 ==========
# class FSARGBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_num=4, expansion=0.5):
#         super().__init__()
#         hidden_dim = int(out_channels * expansion)

#         # 特征主分支
#         self.conv_in = Conv(in_channels, hidden_dim, 1, 1)
#         self.fsa = FSAODConv2d(hidden_dim, hidden_dim, kernel_size=3, kernel_num=kernel_num)
#         self.conv_out = Conv(hidden_dim, out_channels, 1, 1)

#         # 门控机制
#         self.gate = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, 1, bias=False),
#             nn.Sigmoid()
#         )

#         self.add = (in_channels == out_channels)

#     def forward(self, x):
#         g = self.gate(x)
#         y = self.conv_in(x)
#         y = self.fsa(y)
#         y = self.conv_out(y)
#         if self.add:
#             return x + g * y
#         else:
#             return g * y





# # ========== 模块测试 ==========
# if __name__ == "__main__":
#     x = torch.randn(1, 128, 64, 64)
#     block = FSARGBlock(128, 128, kernel_num=4)
#     y = block(x)
#     print("Input:", x.shape)
#     print("Output:", y.shape)


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


# ========== TriAttn-DynConv: 通道 + 空间 + 核注意力的动态卷积 ==========
class TriAttnDynConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 kernel_num=4, reduction=0.0625):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num

        hidden = max(int(in_channels * reduction), 16)

        # 注意力生成分支：全局池化 + 轻量 MLP
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, hidden, 1, bias=False)
        self.act = nn.SiLU(inplace=True)

        # 通道注意力
        self.channel_fc = nn.Conv2d(hidden, in_channels, 1, bias=True)
        # 空间位置注意力（作用在 k×k 上）
        self.spatial_fc = nn.Conv2d(hidden, kernel_size * kernel_size, 1, bias=True)
        # 核选择注意力（K 个候选卷积核）
        self.kernel_fc = nn.Conv2d(hidden, kernel_num, 1, bias=True)

        # 多核可学习卷积权重：K 组基础卷积核
        self.weight = nn.Parameter(
            torch.randn(kernel_num, out_channels, in_channels, kernel_size, kernel_size)
        )

    def forward(self, x):
        """
        x: (B, C_in, H, W)
        return: (B, C_out, H, W)
        """
        B, C, H, W = x.shape
        assert C == self.in_channels, "Input channels mismatch."

        # -------- 1) 全局描述符 & 三种注意力 --------
        att = self.avgpool(x)              # (B, C, 1, 1)
        att = self.act(self.fc1(att))      # (B, hidden, 1, 1)

        # 通道注意力
        ca = torch.sigmoid(self.channel_fc(att))          # (B, C, 1, 1)

        # 空间注意力（k*k）
        k = self.kernel_size
        sa = torch.sigmoid(self.spatial_fc(att))          # (B, k*k, 1, 1)
        sa = sa.view(B, 1, 1, k, k)                       # (B, 1, 1, k, k)

        # 核注意力（对 K 个候选核 softmax）
        ka = self.kernel_fc(att)                          # (B, K, 1, 1)
        ka = F.softmax(ka, dim=1)
        ka = ka.view(B, self.kernel_num, 1, 1, 1, 1)      # (B, K, 1, 1, 1, 1)

        # 对输入进行通道重标定
        x = x * ca                                        # (B, C, H, W)

        # -------- 2) 聚合样本相关的动态卷积核 --------
        # base_weight: (1, K, C_out, C_in, k, k)
        base_weight = self.weight.unsqueeze(0)

        # sa: (B, 1, 1, 1, k, k)
        sa = sa.unsqueeze(1)                              # (B, 1, 1, 1, k, k)

        # 结合核注意力 + 空间注意力，对 K 个基础卷积核做加权求和
        # 结果：每个样本都有自己的一套卷积核
        # agg_w: (B, C_out, C_in, k, k)
        agg_w = (ka * sa * base_weight).sum(1)

        # -------- 3) 用 unfold + einsum 实现“样本自适应动态卷积” --------
        # x_unfold: (B, C_in * k * k, H * W)
        x_unfold = F.unfold(x, kernel_size=k, padding=k // 2)
        x_unfold = x_unfold.view(B, C * k * k, H * W)

        # 将 agg_w 展平到和 x_unfold 对齐
        # agg_w: (B, C_out, C_in * k * k)
        agg_w = agg_w.view(B, self.out_channels, C * k * k)

        # 对每个样本做： (C_out, C_in*k*k) @ (C_in*k*k, H*W) → (C_out, H*W)
        # y: (B, C_out, H*W)
        y = torch.einsum("boc, bcn -> bon", agg_w, x_unfold)

        # reshape 回 feature map
        y = y.view(B, self.out_channels, H, W)
        return y


# ========== TriAttn-GatedBlock: 门控残差融合 ==========
class TriAttnGatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_num=4, expansion=0.5):
        super().__init__()
        hidden_dim = int(out_channels * expansion)

        # 主分支：1x1 降/升维 → TriAttn 动态卷积 → 1x1 投影
        self.conv_in = Conv(in_channels, hidden_dim, k=1, s=1)
        self.tri_conv = TriAttnDynConv2d(
            hidden_dim, hidden_dim,
            kernel_size=3,
            kernel_num=kernel_num
        )
        self.conv_out = Conv(hidden_dim, out_channels, k=1, s=1)

        # 门控分支：用原始特征生成 [0,1] 门
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, bias=False),
            nn.Sigmoid()
        )

        # 是否可以加残差
        self.use_res = (in_channels == out_channels)

    def forward(self, x):
        # 门控系数
        g = self.gate(x)                       # (B, C_out, H, W)

        # 主分支
        y = self.conv_in(x)
        y = self.tri_conv(y)
        y = self.conv_out(y)

        if self.use_res:
            # 残差 + 门控调节
            return x + g * y
        else:
            # 仅门控输出
            return g * y

# class C2fTriAttn(nn.Module):
#     """
#     C2f 风格的模块，但内部的 Bottleneck 换成了 TriAttnGatedBlock。
#     用法与原生 C2f 基本兼容，可以在 YAML 里直接替换。
#     """
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5,
#                  kernel_num=4):
#         """
#         c1: 输入通道
#         c2: 输出通道
#         n:  内部 block 个数（对应 YAML 里的那个 n）
#         shortcut/g 参数为了保持接口兼容，这里可以不用
#         e:  通道缩放系数，和原 C2f 一样，决定中间隐层维度
#         kernel_num: TriAttn 动态卷积的候选核数量
#         expansion: TriAttnGatedBlock 内部的 expansion，通常设为 1.0
#         """
#         super().__init__()
#         self.c = int(c2 * e)

#         # 1x1 conv，将 c1 调整为 2 * c（和 C2f 一致）
#         self.cv1 = Conv(c1, 2 * self.c, k=1, s=1)

#         # 1x1 conv，将 concat 后的 ((2 + n) * c) 融合回 c2
#         self.cv2 = Conv((2 + n) * self.c, c2, k=1, s=1)

#         # 用 TriAttnGatedBlock 替换原来的 Bottleneck
#         self.m = nn.ModuleList(
#             TriAttnGatedBlock(
#                 in_channels=self.c,
#                 out_channels=self.c,
#                 kernel_num=kernel_num,
#             )
#             for _ in range(n)
#         )

#     def forward(self, x):
#         # 和原 C2f 一模一样的前向逻辑
#         y = list(self.cv1(x).split(self.c, 1))  # y = [y1, y2]
#         for block in self.m:
#             y.append(block(y[-1]))             # 逐层堆 TriAttn block
#         return self.cv2(torch.cat(y, 1))

class C2fTriAttn(nn.Module):
    """
    C2f 风格的模块，但内部的 Bottleneck 换成了 TriAttnGatedBlock。
    用法与原生 C2f 完全兼容，可以在 YAML 里直接用 C2fTriAttn 替换 C2f。
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5,
                 kernel_num=4, expansion=1.0):
        """
        c1: 输入通道
        c2: 输出通道
        n:  内部 block 个数（对应 YAML 里的 repeats）
        shortcut, g, e: 为了和原 C2f 接口兼容，不用也不要删
        kernel_num: TriAttn 动态卷积的候选核数量（你自定义）
        expansion: TriAttnGatedBlock 内部的 expansion
        """
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.n = n
        self.shortcut = shortcut
        self.g = g
        self.e = e

        # 中间隐藏通道数，与原 C2f 一致
        self.c = int(c2 * e)

        # 1x1 conv，将 c1 调整为 2 * c
        self.cv1 = Conv(c1, 2 * self.c, k=1, s=1)

        # 1x1 conv，将 concat 后的 ((2 + n) * c) 融合回 c2
        self.cv2 = Conv((2 + n) * self.c, c2, k=1, s=1)

        # 用 TriAttnGatedBlock 替换原来的 Bottleneck
        self.m = nn.ModuleList(
            TriAttnGatedBlock(
                in_channels=self.c,
                out_channels=self.c,
                kernel_num=kernel_num,
                expansion=expansion,
            )
            for _ in range(n)
        )

    def forward(self, x):
        # 和原 C2f 一模一样的前向逻辑
        # cv1 输出通道为 2*c，按通道维分成两个 c
        y = list(self.cv1(x).split(self.c, 1))  # y = [y1, y2]
        for block in self.m:
            y.append(block(y[-1]))              # 逐层堆 TriAttn block
        # 将 [y1, y2, y3, ..., y_(2+n)] 拼接，再 1x1 融合成 c2
        return self.cv2(torch.cat(y, 1))



# ========== 简单测试 ==========
if __name__ == "__main__":
    x = torch.randn(4, 128, 64, 64)
    block = TriAttnGatedBlock(128, 128, kernel_num=4)
    y = block(x)
    print("Input:", x.shape)
    print("Output:", y.shape)
