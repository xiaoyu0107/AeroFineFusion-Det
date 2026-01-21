import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPN_Add(nn.Module):
    """
    Lightweight BiFPN weighted feature fusion (支持 2 或 3 输入)
    自动根据输入特征数量确定权重和通道对齐层。

    改进点：
    - 引入中间通道 mid_channels（瓶颈），先对齐到 mid 再融合，最后再映射到 out_channels
    - 计算量大幅下降，功能保持：多尺度对齐 + 可学习权重融合 + 输出通道一致
    """

    def __init__(self, c1, c2=None, num_inputs=None, epsilon=1e-4, reduction=4):
        """
        Args:
            c1: 输入通道（可以是 int 或 list，例如 [512, 256] 或 [128, 256, 512]）
            c2: 输出通道（建议显式指定，例如 256）
            num_inputs: 输入数量（c1 是单个 int 时需要指定）
            epsilon: 防止除零
            reduction: 中间通道压缩比例，越大越省算力，默认 4 即 mid = out/4
        """
        super().__init__()

        # 自动识别输入结构
        if isinstance(c1, (list, tuple)):
            self.in_channels = list(c1)
            self.num_inputs = len(self.in_channels)
        else:
            assert num_inputs is not None, "如果 c1 是单个整数，请指定 num_inputs"
            self.in_channels = [c1] * num_inputs
            self.num_inputs = num_inputs

        self.epsilon = epsilon
        total_c = sum(self.in_channels)

        # 输出通道：跟原版保持一致逻辑（不传 c2 就用总通道数）
        self.out_channels = c2 or total_c

        # 🔹 新增中间通道（瓶颈），用来降计算量
        #     mid_channels 通常远小于 out_channels，比如 out/4
        self.mid_channels = max(self.out_channels // reduction, 16)

        # 每个输入都有独立的通道对齐 1×1 卷积：Cin -> mid_channels（比原来的 Cin -> out_channels 便宜多了）
        self.align_convs = nn.ModuleList([
            nn.Conv2d(c, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
            for c in self.in_channels
        ])

        # 可学习融合权重
        self.w = nn.Parameter(torch.ones(self.num_inputs, dtype=torch.float32), requires_grad=True)

        # 融合后的 1×1 卷积：mid_channels -> out_channels
        self.conv = nn.Conv2d(self.mid_channels, self.out_channels,
                              kernel_size=1, stride=1, padding=0, bias=False)
        self.act = nn.SiLU()

    def forward(self, inputs):
        """
        Args:
            inputs: list of feature maps [x1, x2, (x3)]
        """
        assert isinstance(inputs, (list, tuple)), f"Expected list, got {type(inputs)}"
        num_inputs = len(inputs)
        assert num_inputs == self.num_inputs, f"Expected {self.num_inputs} inputs, but got {num_inputs}"

        # 以第一个输入为参考尺度
        target_size = inputs[0].shape[2:]

        # 先用 1×1 映射到 mid_channels，再插值到同一空间尺寸
        resized = []
        for i, x in enumerate(inputs):
            y = self.align_convs[i](x)  # (B, mid, Hi, Wi)
            if x.shape[2:] != target_size:
                y = F.interpolate(y, size=target_size, mode='nearest')
            resized.append(y)           # (B, mid, Ht, Wt)

        # 权重归一化
        w = F.relu(self.w)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)

        # 在 mid_channels 空间里做加权融合
        fused_mid = 0
        for i in range(num_inputs):
            fused_mid = fused_mid + weight[i] * resized[i]

        # 再映射回 out_channels
        out = self.conv(self.act(fused_mid))
        return out


# ✅ 简单测试
if __name__ == "__main__":
    # 2 输入
    x1 = torch.randn(1, 256, 40, 40)
    x2 = torch.randn(1, 512, 20, 20)
    model2 = BiFPN_Add(c1=[256, 512], c2=256)
    y2 = model2([x1, x2])
    print("2-input output shape:", y2.shape)

    # 3 输入
    x3 = torch.randn(1, 128, 80, 80)
    model3 = BiFPN_Add(c1=[128, 256, 512], c2=256)
    y3 = model3([x3, x1, x2])
    print("3-input output shape:", y3.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保已经 from .bifpn import BiFPN_Add 或者提前定义好 BiFPN_Add
# 这里假设 BiFPN_Add(c1=[c_in1, c_in2, ...], c2=out_c)


class BiFPNBlockP2(nn.Module):
    """
    单层 P2-P5 BiFPN
    __init__ 形式专门为 Ultralytics parse_model 设计：

    Args:
        in_channels:  [c2_in, c3_in, c4_in, c5_in]  来自 backbone 对应层
        out_channels: [c2_out, c3_out, c4_out, c5_out]  输出给 Detect 的通道数

    forward:
        x: list/tuple [P2, P3, P4, P5]
        return:       [P2_out, P3_out, P4_out, P5_out]
    """

    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        assert isinstance(in_channels, (list, tuple)) and len(in_channels) == 4, \
            f"in_channels 必须是长度为 4 的 list，比如 [128,256,512,1024]，现在是 {in_channels}"

        if out_channels is None:
            out_channels = in_channels
        assert isinstance(out_channels, (list, tuple)) and len(out_channels) == 4, \
            f"out_channels 必须是长度为 4 的 list，比如 [128,256,512,1024]，现在是 {out_channels}"

        self.in_channels = in_channels
        self.out_channels = out_channels

        c2_in, c3_in, c4_in, c5_in = in_channels
        c2_out, c3_out, c4_out, c5_out = out_channels

        # -------- Top-Down 部分 --------
        # P5_td 先做一次 1×1
        self.p5_td_conv = nn.Conv2d(c5_in, c5_out, kernel_size=1, stride=1, padding=0, bias=False)

        # P4_td = f(P4, Up(P5_td))
        self.p4_td_fuse = BiFPN_Add(c1=[c4_in, c5_out], c2=c4_out)

        # P3_td = f(P3, Up(P4_td))
        self.p3_td_fuse = BiFPN_Add(c1=[c3_in, c4_out], c2=c3_out)

        # P2_td = f(P2, Up(P3_td))
        self.p2_td_fuse = BiFPN_Add(c1=[c2_in, c3_out], c2=c2_out)

        # -------- Bottom-Up 部分 --------
        # P2_out = f(P2, P2_td)
        self.p2_out_fuse = BiFPN_Add(c1=[c2_in, c2_out], c2=c2_out)

        # P3_out = f(P3, P3_td, Down(P2_out))
        self.p3_out_fuse = BiFPN_Add(c1=[c3_in, c3_out, c2_out], c2=c3_out)

        # P4_out = f(P4, P4_td, Down(P3_out))
        self.p4_out_fuse = BiFPN_Add(c1=[c4_in, c4_out, c3_out], c2=c4_out)

        # P5_out = f(P5, P5_td, Down(P4_out))
        self.p5_out_fuse = BiFPN_Add(c1=[c5_in, c5_out, c4_out], c2=c5_out)

    def forward(self, x):
        """
        x: [P2, P3, P4, P5]
        """
        assert isinstance(x, (list, tuple)) and len(x) == 4, \
            f"BiFPNBlockP2 输入必须是长度为 4 的 list，例如 [P2,P3,P4,P5]，现在是 {type(x)} 长度 {len(x)}"

        P2, P3, P4, P5 = x

        # -------- Top-Down --------
        P5_td = self.p5_td_conv(P5)  # (B, c5_out, H5, W5)

        P4_td = self.p4_td_fuse([
            P4,
            F.interpolate(P5_td, size=P4.shape[2:], mode="nearest")
        ])

        P3_td = self.p3_td_fuse([
            P3,
            F.interpolate(P4_td, size=P3.shape[2:], mode="nearest")
        ])

        P2_td = self.p2_td_fuse([
            P2,
            F.interpolate(P3_td, size=P2.shape[2:], mode="nearest")
        ])

        # -------- Bottom-Up --------
        P2_out = self.p2_out_fuse([P2, P2_td])

        P3_out = self.p3_out_fuse([
            P3,
            P3_td,
            F.max_pool2d(P2_out, kernel_size=2, stride=2)  # P2_out: stride 4 -> 8
        ])

        P4_out = self.p4_out_fuse([
            P4,
            P4_td,
            F.max_pool2d(P3_out, kernel_size=2, stride=2)  # P3_out: stride 8 -> 16
        ])

        P5_out = self.p5_out_fuse([
            P5,
            P5_td,
            F.max_pool2d(P4_out, kernel_size=2, stride=2)  # P4_out: stride 16 -> 32
        ])

        # 给 Detect 用，保持 list 形式
        return [P2_out, P3_out, P4_out, P5_out]


# 简单自测
if __name__ == "__main__":
    from bifpn import BiFPN_Add  # 如果你单独测试，注意正确导入

    P2 = torch.randn(1, 128, 80, 80)   # stride=4
    P3 = torch.randn(1, 256, 40, 40)   # stride=8
    P4 = torch.randn(1, 512, 20, 20)   # stride=16
    P5 = torch.randn(1, 1024, 10, 10)  # stride=32

    block = BiFPNBlockP2(in_channels=[128, 256, 512, 1024],
                         out_channels=[128, 256, 512, 1024])
    outs = block([P2, P3, P4, P5])
    for o in outs:
        print(o.shape)


# class BiFPN_Add(nn.Module):
#     """
#     Lightweight BiFPN weighted feature fusion (支持 2 或 3 输入)
#     自动根据输入特征数量确定权重和通道对齐层。
#     """

#     def __init__(self, c1, c2=None, num_inputs=None, epsilon=1e-4):
#         """
#         Args:
#             c1: 输入通道（可以是 int 或 list，例如 [512, 256] 或 [128, 256, 512]）
#             c2: 输出通道（默认为 c1 总和或第一个输入通道）
#             num_inputs: 输入数量（可自动推断）
#             epsilon: 防止除零
#         """
#         super(BiFPN_Add, self).__init__()

#         # ✅ 自动识别输入结构
#         if isinstance(c1, (list, tuple)):
#             self.in_channels = c1
#             self.num_inputs = len(c1)
#         else:
#             # 如果给的是单个 int，则需要指定输入数量
#             assert num_inputs is not None, "如果 c1 是单个整数，请指定 num_inputs"
#             self.in_channels = [c1] * num_inputs
#             self.num_inputs = num_inputs

#         self.epsilon = epsilon
#         total_c = sum(self.in_channels)
#         self.out_channels = c2 or total_c

#         # ✅ 每个输入都有独立的通道对齐 1×1 卷积
#         self.align_convs = nn.ModuleList([
#             nn.Conv2d(c, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
#             for c in self.in_channels
#         ])

#         # ✅ 可学习融合权重（根据输入数量动态创建）
#         self.w = nn.Parameter(torch.ones(self.num_inputs, dtype=torch.float32), requires_grad=True)

#         # ✅ 输出融合卷积
#         self.conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
#         self.act = nn.SiLU()

#     def forward(self, inputs):
#         """
#         Args:
#             inputs: list of feature maps [x1, x2, (x3)]
#         """
#         assert isinstance(inputs, (list, tuple)), f"Expected list, got {type(inputs)}"
#         num_inputs = len(inputs)
#         assert num_inputs == self.num_inputs, f"Expected {self.num_inputs} inputs, but got {num_inputs}"

#         # ✅ 将所有输入resize到相同空间大小（以第一个输入为参考）
#         target_size = inputs[0].shape[2:]
#         resized = [
#             F.interpolate(self.align_convs[i](x), size=target_size, mode='nearest')
#             if x.shape[2:] != target_size else self.align_convs[i](x)
#             for i, x in enumerate(inputs)
#         ]

#         # ✅ 权重归一化
#         w = F.relu(self.w)
#         weight = w / (torch.sum(w, dim=0) + self.epsilon)

#         # ✅ 加权融合
#         fused = sum(weight[i] * resized[i] for i in range(num_inputs))

#         # ✅ 输出
#         return self.conv(self.act(fused))


# # ✅ 测试用例
# if __name__ == "__main__":
#     # 测试 2 输入
#     x1 = torch.randn(1, 256, 40, 40)
#     x2 = torch.randn(1, 512, 20, 20)
#     model2 = BiFPN_Add(c1=[256, 512], c2=256)
#     y2 = model2([x1, x2])
#     print("✅ 2-input output shape:", y2.shape)

#     # 测试 3 输入
#     x3 = torch.randn(1, 128, 80, 80)
#     model3 = BiFPN_Add(c1=[128, 256, 512], c2=256)
#     y3 = model3([x3, x1, x2])
#     print("✅ 3-input output shape:", y3.shape)



# BiFPN
# # 两个特征图add操作
# import torch.nn as nn
# import torch
# # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
# 		# 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
# 		# 从而在参数优化的时候可以自动一起优化
# class BiFPN_Add2(nn.Module):
# 	def __init__(self, c1, c2):
# 		super(BiFPN_Add2, self).__init__()
# 		self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
# 		self.epsilon = 0.0001
# 		self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
# 		self.silu = nn.SiLU()
# 	def forward(self, x):
# 		x0, x1 = x
		
# 		w = self.w
# 		weight = w / (torch.sum(w, dim=0) + self.epsilon)
# 		return self.conv(self.silu(weight[0] * x0 + weight[1] * x1))

# import torch
# import torch.nn as nn

# class BiFPN_Add2(nn.Module):
#     def __init__(self, c1, c2):
#         super(BiFPN_Add2, self).__init__()
#         self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#         self.epsilon = 1e-4
#         self.silu = nn.SiLU()

#         # 为两个输入各自做1×1卷积，将通道统一到c1
#         self.align1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
#         self.align2 = nn.Conv2d(c1, c, kernel_size=1, stride=1, padding=0)

#         # 输出投影卷积
#         self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)

#     def forward(self, x):
#         x0, x1 = x
#         # 通过插值适配输入通道不一致的情况
        
#         if x0.shape[1] != x1.shape[1]:
#             # 自动扩展或压缩到相同通道数
#             if x0.shape[1] > x1.shape[1]:
#                 x1 = torch.nn.functional.interpolate(x1, size=x0.shape[2:], mode='nearest')
#                 x1 = torch.cat([x1, torch.zeros_like(x1[:, :x0.shape[1]-x1.shape[1], :, :])], dim=1)
#             elif x1.shape[1] > x0.shape[1]:
#                 x0 = torch.nn.functional.interpolate(x0, size=x1.shape[2:], mode='nearest')
#                 x0 = torch.cat([x0, torch.zeros_like(x0[:, :x1.shape[1]-x0.shape[1], :, :])], dim=1)

#         # 通道对齐卷积
#         x0 = self.align1(x0)
#         x1 = self.align2(x1)

#         # 权重归一化融合
#         w = torch.relu(self.w)
#         weight = w / (torch.sum(w, dim=0) + self.epsilon)

#         # 双特征融合 + 输出卷积
#         out = weight[0] * x0 + weight[1] * x1
#         return self.conv(self.silu(out))

# 三个特征图add操作
# class BiFPN_Add3(nn.Module):
# 	def __init__(self, c1, c2):
# 		super(BiFPN_Add3, self).__init__()
# 		self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
# 		self.epsilon = 0.0001
# 		self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
# 		self.silu = nn.SiLU()

# 	def forward(self, x):
# 		w = self.w
# 		weight = w / (torch.sum(w, dim=0) + self.epsilon)
# 		# Fast normalized fusion
# 		return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))