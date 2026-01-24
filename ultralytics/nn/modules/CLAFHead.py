
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from ultralytics.utils.tal import dist2bbox, make_anchors


# def autopad(k, p=None, d=1):
#     """Pad to 'same' shape outputs."""
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
#     return p


# class Conv(nn.Module):
#     """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
#     default_act = nn.SiLU()  # default activation

#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

#     def forward_fuse(self, x):
#         return self.act(self.conv(x))


# class DFL(nn.Module):
#     """
#     Integral module of Distribution Focal Loss (DFL).
#     Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
#     """

#     def __init__(self, c1=16):
#         super().__init__()
#         self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
#         x = torch.arange(c1, dtype=torch.float)
#         self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
#         self.c1 = c1

#     def forward(self, x):
#         b, c, a = x.shape  # batch, channels, anchors
#         return self.conv(
#             x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)
#         ).view(b, 4, a)


# class CLLABlock(nn.Module):
#     """
#     轻量版 Cross-Level Local-Aware Fusion Block
#     - 保留原有 __init__ 的参数形式 (range, ch, ch1, ch2, out)
#     - 内部用跨层 SE + DWConv 融合，替代原来的 q/k/v + Softmax 结构
#     """

#     def __init__(self, range=2, ch=256, ch1=128, ch2=256, out=0, reduction=4):
#         super().__init__()
#         self.c_mid = ch

#         # 对齐两个尺度的通道数
#         self.conv1 = Conv(ch1, self.c_mid, 1, 1)
#         self.conv2 = Conv(ch2, self.c_mid, 1, 1)

#         # 全局通道注意力（跨层 SE）
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         mid = max(self.c_mid // reduction, 8)
#         self.att_mlp = nn.Sequential(
#             nn.Conv2d(self.c_mid * 2, mid, 1),
#             nn.SiLU(),
#             nn.Conv2d(mid, self.c_mid, 1),
#             nn.Sigmoid(),
#         )

#         # 轻量卷积增强：DWConv + PWConv
#         self.dw = nn.Conv2d(self.c_mid, self.c_mid, 3, 1, 1, groups=self.c_mid, bias=False)
#         self.pw = nn.Conv2d(self.c_mid, self.c_mid, 1, 1, 0, bias=False)
#         self.bn = nn.BatchNorm2d(self.c_mid)
#         self.act = nn.SiLU()

#         # 最终输出到 no 通道（4*reg_max + nc）
#         self.det = nn.Conv2d(self.c_mid, out, 1)

#     def forward(self, x1, x2):
#         # 通道对齐
#         x1 = self.conv1(x1)
#         x2 = self.conv2(x2)

#         # 如果空间尺寸不一致，统一到低分辨率一侧（一般 x2 是更低分辨率）
#         if x1.shape[-2:] != x2.shape[-2:]:
#             x1 = F.adaptive_avg_pool2d(x1, x2.shape[-2:])

#         # 融合 + 通道注意力
#         fuse = x1 + x2
#         g = self.pool(torch.cat([x1, x2], dim=1))   # B, 2C, 1, 1
#         g = self.att_mlp(g)                         # B, C, 1, 1
#         fuse = fuse * g

#         # 轻量卷积增强
#         fuse = self.dw(fuse)
#         fuse = self.act(self.bn(self.pw(fuse)))

#         # 输出检测特征 (B, no, H, W)
#         return self.det(fuse)


# class Detect_CLAFHead(nn.Module):
#     """YOLOv8 Detect head with Lite Cross-Level Local-Aware Fusion for mid-scale feature."""
#     dynamic = False  # force grid reconstruction
#     export = False  # export mode
#     shape = None
#     anchors = torch.empty(0)  # init
#     strides = torch.empty(0)  # init

#     def __init__(self, nc=80, ch=()):
#         """
#         nc: number of classes
#         ch: tuple/list of input channels of each FPN level, e.g. (P2, P3, P4)
#         """
#         super().__init__()
#         self.nc = nc
#         self.nl = len(ch)  # number of detection layers
#         self.reg_max = 16
#         self.no = nc + self.reg_max * 4
#         self.stride = torch.zeros(self.nl)

#         c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))

#         # ====== 回归 & 分类塔 ======
#         self.cv2 = nn.ModuleList()
#         self.cv3 = nn.ModuleList()
#         for i, c in enumerate(ch):
#             if i == 1:
#                 # 中尺度由 CLLABlock 负责，不再单独建 head 塔，节省参数
#                 self.cv2.append(nn.Identity())
#                 self.cv3.append(nn.Identity())
#             else:
#                 self.cv2.append(
#                     nn.Sequential(
#                         Conv(c, c2, 3),
#                         Conv(c2, c2, 3),
#                         nn.Conv2d(c2, 4 * self.reg_max, 1),
#                     )
#                 )
#                 self.cv3.append(
#                     nn.Sequential(
#                         Conv(c, c3, 3),
#                         Conv(c3, c3, 3),
#                         nn.Conv2d(c3, self.nc, 1),
#                     )
#                 )

#         # ====== 中尺度跨层融合检测头 (例如 P2 + P3) ======
#         # 注意：保持初始化参数形式不变
#         self.det = CLLABlock(range=2, ch=ch[0], ch1=ch[0], ch2=ch[1], out=self.no)

#         self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

#     def forward(self, x):
#         """
#         x: list of feature maps [P2, P3, P4]，对应 ch 里的通道数
#         训练模式返回每层的 raw logits，推理模式返回 decode 之后的 boxes + cls
#         """
#         shape = x[0].shape  # BCHW
#         p = []
#         for i in range(self.nl):
#             if i == 1:
#                 # 中尺度用跨层融合的 CLAF head：比如用 P2 + P3
#                 p.append(self.det(x[0], x[1]))
#             else:
#                 # 其它尺度走标准 YOLOv8 head 塔
#                 p.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))

#         if self.training:
#             return p

#         # ====== 推理 decode 部分保持不变 ======
#         if self.dynamic or self.shape != shape:
#             self.anchors, self.strides = (
#                 y.transpose(0, 1) for y in make_anchors(p, self.stride, 0.5)
#             )
#             self.shape = shape

#         x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in p], 2)
#         box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
#         dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

#         if self.export and self.format in ("tflite", "edgetpu"):
#             img_h = shape[2] * self.stride[0]
#             img_w = shape[3] * self.stride[0]
#             img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
#             dbox /= img_size

#         y = torch.cat((dbox, cls.sigmoid()), 1)
#         return y if self.export else (y, p)

#     def bias_init(self):
#         """Initialize Detect() biases, WARNING: requires stride availability."""
#         m = self
#         for i, (a, b, s) in enumerate(zip(m.cv2, m.cv3, m.stride)):
#             if isinstance(a, nn.Sequential) and isinstance(b, nn.Sequential):
#                 a[-1].bias.data[:] = 1.0  # box
#                 b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)


# if __name__ == "__main__":
#     # Example with P2-P4 三尺度
#     image1 = torch.rand(1, 64, 160, 160)   # P2
#     image2 = torch.rand(1, 128, 80, 80)    # P3
#     image3 = torch.rand(1, 256, 40, 40)    # P4
#     feats = [image1, image2, image3]
#     ch = (64, 128, 256)

#     head = Detect_CLAFHead(nc=80, ch=ch)
#     out = head(feats)
#     print(type(out), len(out) if not isinstance(out, tuple) else len(out[1]))



######################################################
# Detect_CLAFDecoupledHead###############################
######################################################
import math
import torch
import torch.nn as nn
from ultralytics.utils.tal import dist2bbox, make_anchors


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d),
                              groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL)."""

    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(
            x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)
        ).view(b, 4, a)




class CLLAFuse(nn.Module):
    """
    轻量版 Cross-Level Local-Aware Fusion Block（只做特征融合，不做预测）
    - 输入：x1, x2（比如 P2, P3）
    - 输出：fused feature，通道为 c_mid，用于后续 Decoupled Head 的 reg/cls 塔
    """

    def __init__(self, range=2, ch=256, ch1=128, ch2=256, reduction=4):
        super().__init__()
        self.c_mid = ch  # 融合后通道数，默认用低层通道

        # 对齐两个尺度的通道数
        self.conv1 = Conv(ch1, self.c_mid, 1, 1)
        self.conv2 = Conv(ch2, self.c_mid, 1, 1)

        # 全局通道注意力（跨层 SE）
        self.pool = nn.AdaptiveAvgPool2d(1)
        mid = max(self.c_mid // reduction, 8)
        self.att_mlp = nn.Sequential(
            nn.Conv2d(self.c_mid * 2, mid, 1),
            nn.SiLU(),
            nn.Conv2d(mid, self.c_mid, 1),
            nn.Sigmoid(),
        )

        # 轻量卷积增强：DWConv + PWConv
        self.dw = nn.Conv2d(self.c_mid, self.c_mid, 3, 1, 1, groups=self.c_mid, bias=False)
        self.pw = nn.Conv2d(self.c_mid, self.c_mid, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(self.c_mid)
        self.act = nn.SiLU()

    def forward(self, x1, x2):
        # 通道对齐
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        # 若尺寸不一致，统一到低分辨率一侧（一般 x2 是更低分辨率）
        if x1.shape[-2:] != x2.shape[-2:]:
            x1 = F.adaptive_avg_pool2d(x1, x2.shape[-2:])

        # 融合 + 通道注意力
        fuse = x1 + x2
        g = self.pool(torch.cat([x1, x2], dim=1))   # B, 2C, 1, 1
        g = self.att_mlp(g)                         # B, C, 1, 1
        fuse = fuse * g

        # 轻量卷积增强
        fuse = self.dw(fuse)
        fuse = self.act(self.bn(self.pw(fuse)))

        # 返回融合后的 feature，不做 det conv
        return fuse
    


# Detect_CLAFDecoupledHead###############################


class Detect_CLAFHead(nn.Module):
    """YOLOv8 Detect head with Lite Cross-Level Local-Aware Fusion + Decoupled Head."""
    dynamic = False  # force grid reconstruction
    export = False   # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """
        nc: number of classes
        ch: tuple/list of input channels of each FPN level, e.g. (P2, P3, P4)
        """
        super().__init__()
        self.nc = nc
        self.nl = len(ch)       # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        # 这里保持你原来的 c2, c3 设计，便于做 ablation
        c2 = max((16, ch[0] // 4, self.reg_max * 4))     # 中间回归通道
        c3 = max(ch[0], min(self.nc, 100))               # 中间分类通道

        # ====== 中尺度跨层融合模块：只输出 feature ======
        # 注意：参数形式保持与原 CLLABlock 一致，方便替换
        self.fuse_mid = CLLAFuse(range=2, ch=ch[0], ch1=ch[0], ch2=ch[1])

        # ====== Decoupled Head：每个尺度都有一套 reg / cls 塔 ======
        self.reg_convs = nn.ModuleList()  # 回归塔
        self.cls_convs = nn.ModuleList()  # 分类塔

        for i in range(self.nl):
            # 对于中尺度，输入通道是融合后的 c_mid；其它尺度用原 ch[i]
            if i == 1:
                in_c = self.fuse_mid.c_mid
            else:
                in_c = ch[i]

            # 回归塔：两层 3x3 Conv + 预测层
            self.reg_convs.append(
                nn.Sequential(
                    Conv(in_c, c2, 3),
                    Conv(c2, c2, 3),
                    nn.Conv2d(c2, 4 * self.reg_max, 1),
                )
            )
            # 分类塔：两层 3x3 Conv + 预测层
            self.cls_convs.append(
                nn.Sequential(
                    Conv(in_c, c3, 3),
                    Conv(c3, c3, 3),
                    nn.Conv2d(c3, self.nc, 1),
                )
            )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """
        x: list of feature maps [P2, P3, P4]，对应 ch 里的通道数
        训练模式返回每层的 raw logits，推理模式返回 decode 之后的 boxes + cls
        """
        shape = x[0].shape  # BCHW
        p = []

        for i in range(self.nl):
            # ====== 1) 先得到每个尺度的输入特征 ======
            if i == 1:
                # 中尺度用 P2 + P3 做 CLAF 融合
                feat = self.fuse_mid(x[0], x[1])
            else:
                feat = x[i]

            # ====== 2) Decoupled Head：分别走 reg / cls 塔 ======
            reg_out = self.reg_convs[i](feat)   # [B, 4*reg_max, H, W]
            cls_out = self.cls_convs[i](feat)   # [B, nc, H, W]

            # 拼成 [B, no, H, W]
            p.append(torch.cat((reg_out, cls_out), 1))

        # ====== 训练：直接返回各尺度 logits ======
        if self.training:
            return p

        # ====== 推理：decode 部分保持与你原来的一致 ======
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                y.transpose(0, 1) for y in make_anchors(p, self.stride, 0.5)
            )
            self.shape = shape

        # [B, no, HW] 拼接所有尺度
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in p], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        # DFL + dist2bbox 解码为 xywh
        dbox = dist2bbox(self.dfl(box),
                         self.anchors.unsqueeze(0),
                         xywh=True,
                         dim=1) * self.strides  # [B, 4, HW_all]

        # tflite / edgetpu 导出时做归一化（和你原来的逻辑保持一致）
        if self.export and self.format in ("tflite", "edgetpu"):
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h],
                                    device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        # 最终输出 [B, 4 + nc, HW_all]
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, p)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self
        for i, (a, b, s) in enumerate(zip(m.reg_convs, m.cls_convs, m.stride)):
            # 回归分支 bias
            if isinstance(a, nn.Sequential):
                a[-1].bias.data[:] = 1.0  # box

            # 分类分支 bias
            if isinstance(b, nn.Sequential):
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)


# #################################################
# # Detect_CLAFHeadHybrid
# #################################################

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from ultralytics.utils.tal import dist2bbox, make_anchors


# def autopad(k, p=None, d=1):
#     """Pad to 'same' shape outputs."""
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
#     return p


# class Conv(nn.Module):
#     """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
#     default_act = nn.SiLU()  # default activation

#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))

#     def forward_fuse(self, x):
#         return self.act(self.conv(x))


# class DFL(nn.Module):
#     """
#     Integral module of Distribution Focal Loss (DFL).
#     Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
#     """

#     def __init__(self, c1=16):
#         super().__init__()
#         self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
#         x = torch.arange(c1, dtype=torch.float)
#         self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
#         self.c1 = c1

#     def forward(self, x):
#         b, c, a = x.shape  # batch, channels, anchors
#         return self.conv(
#             x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)
#         ).view(b, 4, a)


# class CLLAFuseLite(nn.Module):
#     """
#     轻量版 Cross-Level Local-Aware Fusion Block（只做特征融合，不做预测）
#     - 输入：x1, x2（比如 P2, P3）
#     - 输出：fused feature，通道为 c_mid，用于后续 Decoupled Head 的 reg/cls 塔
#     """

#     def __init__(self, range=2, ch=256, ch1=128, ch2=256, reduction=4):
#         """
#         range: 兼容原 CLLABlock 接口，占位，不实际使用
#         ch:    融合后的中间通道数 c_mid（建议不大于 min(ch1, ch2)）
#         ch1:   上层/低层通道（如 P2）
#         ch2:   下层/高层通道（如 P3）
#         """
#         super().__init__()
#         self.c_mid = ch  # 融合后通道数，默认用较小一侧

#         # 对齐两个尺度的通道数
#         self.conv1 = Conv(ch1, self.c_mid, 1, 1)
#         self.conv2 = Conv(ch2, self.c_mid, 1, 1)

#         # 全局通道注意力（跨层 SE）
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         mid = max(self.c_mid // reduction, 8)
#         self.att_mlp = nn.Sequential(
#             nn.Conv2d(self.c_mid * 2, mid, 1),
#             nn.SiLU(),
#             nn.Conv2d(mid, self.c_mid, 1),
#             nn.Sigmoid(),
#         )

#         # 轻量卷积增强：DWConv + PWConv
#         self.dw = nn.Conv2d(self.c_mid, self.c_mid, 3, 1, 1, groups=self.c_mid, bias=False)
#         self.pw = nn.Conv2d(self.c_mid, self.c_mid, 1, 1, 0, bias=False)
#         self.bn = nn.BatchNorm2d(self.c_mid)
#         self.act = nn.SiLU()

#     def forward(self, x1, x2):
#         # 通道对齐
#         x1 = self.conv1(x1)
#         x2 = self.conv2(x2)

#         # 若尺寸不一致，统一到低分辨率一侧（一般 x2 是更低分辨率）
#         if x1.shape[-2:] != x2.shape[-2:]:
#             x1 = F.adaptive_avg_pool2d(x1, x2.shape[-2:])

#         # 融合 + 通道注意力
#         fuse = x1 + x2
#         g = self.pool(torch.cat([x1, x2], dim=1))   # B, 2C, 1, 1
#         g = self.att_mlp(g)                         # B, C, 1, 1
#         fuse = fuse * g

#         # 轻量卷积增强
#         fuse = self.dw(fuse)
#         fuse = self.act(self.bn(self.pw(fuse)))

#         # 返回融合后的 feature，不做 det conv
#         return fuse


# class Detect_CLAFHead(nn.Module):
#     """
#     折中版 YOLOv8 Detect Head：
#     - 保留 Decoupled Head（reg / cls 分支分开）
#     - 中尺度引入 CLLAFuseLite 做 P2+P3 融合
#     - 同时对 head 通道数做上限截断，并减浅中尺度的 head 深度
#     """
#     dynamic = False  # force grid reconstruction
#     export = False   # export mode
#     shape = None
#     anchors = torch.empty(0)  # init
#     strides = torch.empty(0)  # init

#     def __init__(self, nc=80, ch=()):
#         """
#         nc: number of classes
#         ch: tuple/list of input channels of each FPN level, e.g. (P2, P3, P4)
#         """
#         super().__init__()
#         self.nc = nc
#         self.nl = len(ch)  # number of detection layers
#         self.reg_max = 16
#         self.no = nc + self.reg_max * 4
#         self.stride = torch.zeros(self.nl)

#         # ====== 中尺度跨层融合模块：只输出融合特征 ======
#         # 这里用较小一侧通道作为 c_mid，进一步控制计算量
#         c_mid = min(ch[0], ch[1])
#         self.fuse_mid = CLLAFuseLite(range=2, ch=c_mid, ch1=ch[0], ch2=ch[1])

#         # ====== Decoupled Head：每个尺度 reg / cls 各一套塔 ======
#         self.reg_convs = nn.ModuleList()
#         self.cls_convs = nn.ModuleList()

#         for i, c in enumerate(ch):
#             # 1) 确定该尺度的输入通道 & depth
#             if i == 1:
#                 # 中尺度用融合后的 c_mid，head 深度减为 1
#                 in_c = self.fuse_mid.c_mid
#                 depth = 1
#             else:
#                 in_c = c
#                 depth = 2  # 其它尺度保持两层 3x3

#             # 2) 轻量化的中间通道宽度（带上限截断）
#             # 回归塔：不超过 64，且至少 >= 4*reg_max
#             c2 = max(16, min(self.reg_max * 4, in_c // 2))  # in_c//2 再和 4*reg_max 取小，典型 64
#             # 分类塔：不超过 128，且至少 >= nc
#             c3 = max(self.nc, min(128, in_c))

#             # 3) 构造回归塔
#             reg_layers = [Conv(in_c, c2, 3)]
#             for _ in range(depth - 1):
#                 reg_layers.append(Conv(c2, c2, 3))
#             reg_layers.append(nn.Conv2d(c2, 4 * self.reg_max, 1))
#             self.reg_convs.append(nn.Sequential(*reg_layers))

#             # 4) 构造分类塔
#             cls_layers = [Conv(in_c, c3, 3)]
#             for _ in range(depth - 1):
#                 cls_layers.append(Conv(c3, c3, 3))
#             cls_layers.append(nn.Conv2d(c3, self.nc, 1))
#             self.cls_convs.append(nn.Sequential(*cls_layers))

#         self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

#     def forward(self, x):
#         """
#         x: list of feature maps [P2, P3, P4]，对应 ch 里的通道数
#         训练模式返回每层的 raw logits，推理模式返回 decode 之后的 boxes + cls
#         """
#         shape = x[0].shape  # BCHW
#         p = []

#         for i in range(self.nl):
#             # ====== 1) 先得到每个尺度的输入特征 ======
#             if i == 1:
#                 # 中尺度用 P2 + P3 做 CLAF 融合
#                 feat = self.fuse_mid(x[0], x[1])
#             else:
#                 feat = x[i]

#             # ====== 2) Decoupled Head：分别走 reg / cls 塔 ======
#             reg_out = self.reg_convs[i](feat)   # [B, 4*reg_max, H, W]
#             cls_out = self.cls_convs[i](feat)   # [B, nc, H, W]

#             # 拼成 [B, no, H, W]
#             p.append(torch.cat((reg_out, cls_out), 1))

#         # ====== 训练：直接返回各尺度 logits ======
#         if self.training:
#             return p

#         # ====== 推理 decode 部分保持与原 Detect 一致 ======
#         if self.dynamic or self.shape != shape:
#             self.anchors, self.strides = (
#                 y.transpose(0, 1) for y in make_anchors(p, self.stride, 0.5)
#             )
#             self.shape = shape

#         # [B, no, HW] 拼接所有尺度
#         x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in p], 2)
#         box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

#         # DFL + dist2bbox 解码为 xywh
#         dbox = dist2bbox(self.dfl(box),
#                          self.anchors.unsqueeze(0),
#                          xywh=True,
#                          dim=1) * self.strides  # [B, 4, HW_all]

#         # tflite / edgetpu 导出时做归一化
#         if self.export and self.format in ("tflite", "edgetpu"):
#             img_h = shape[2] * self.stride[0]
#             img_w = shape[3] * self.stride[0]
#             img_size = torch.tensor([img_w, img_h, img_w, img_h],
#                                     device=dbox.device).reshape(1, 4, 1)
#             dbox /= img_size

#         # 最终输出 [B, 4 + nc, HW_all]
#         y = torch.cat((dbox, cls.sigmoid()), 1)
#         return y if self.export else (y, p)

#     def bias_init(self):
#         """Initialize Detect() biases, WARNING: requires stride availability."""
#         m = self
#         for i, (a, b, s) in enumerate(zip(m.reg_convs, m.cls_convs, m.stride)):
#             if isinstance(a, nn.Sequential):
#                 a[-1].bias.data[:] = 1.0  # box
#             if isinstance(b, nn.Sequential):
#                 b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)


# if __name__ == "__main__":
#     # Example with P2-P4 三尺度
#     image1 = torch.rand(1, 64, 160, 160)   # P2
#     image2 = torch.rand(1, 128, 80, 80)    # P3
#     image3 = torch.rand(1, 256, 40, 40)    # P4
#     feats = [image1, image2, image3]
#     ch = (64, 128, 256)

#     head = Detect_CLAFHead(nc=80, ch=ch)
#     head.train()
#     out_train = head(feats)
#     print("Train mode output lens:", len(out_train), [o.shape for o in out_train])

#     head.eval()
#     out_eval = head(feats)
#     y, p = out_eval
#     print("Eval mode y shape:", y.shape, "p lens:", len(p))
