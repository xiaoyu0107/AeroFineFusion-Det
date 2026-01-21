import torch
import torch.nn as nn

# autopad & Conv 与你现有一致（YOLO 风格）
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k,p,d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# ========== Lite CGB ==========
class CGB(nn.Module):
    """
    轻量化 CGB：
      - bottleneck: 在投影后把 c2 -> c_mid，降低后续计算
      - 局部/周围采用 depthwise conv（DW）+ pointwise（PW）
      - 全局注意力使用 GAP->FC but with larger reduction
      - 融合用 add（避免 concat 后通道翻倍），如需 concat 可切换
    """
    def __init__(self, c1, c2, k=3, s=2, reduction=32, use_concat=False):
        super().__init__()
        c_mid = max(8, c2 // 2)              # bottleneck 中间通道
        self.use_concat = use_concat

        # 1x1 投影 + 下采样到 c_mid
        self.proj = Conv(c1, c_mid, k=1, s=s)

        # 局部分支：DW(3x3) + PW(1x1)
        self.loc_dw = Conv(c_mid, c_mid, k=3, s=1, g=c_mid)    # 深度卷积
        self.loc_pw = Conv(c_mid, c_mid, k=1, s=1)

        # 周围分支：dilated DW + PW
        self.sur_dw = Conv(c_mid, c_mid, k=3, s=1, d=2, g=c_mid)
        self.sur_pw = Conv(c_mid, c_mid, k=1, s=1)

        # 融合：若 use_concat，则 concat -> BN+PReLU -> GAP->FC；否则用 add（通道不变）
        if use_concat:
            self.post_bn_prelu = nn.Sequential(nn.BatchNorm2d(c_mid * 2), nn.PReLU())
            att_channels = c_mid * 2
        else:
            self.post_bn_prelu = nn.Sequential(nn.BatchNorm2d(c_mid), nn.PReLU())
            att_channels = c_mid

        # 全局上下文（轻量化 FC）
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(att_channels, max(1, att_channels // reduction), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(max(1, att_channels // reduction), att_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 如果 use_concat：最后降回 c2 用 1x1 conv，否则直接用 1x1 conv 恢复到 c2
        if use_concat:
            self.final_conv = Conv(att_channels, c2, k=1, s=1)
        else:
            self.final_conv = Conv(c_mid, c2, k=1, s=1)

    def forward(self, x):
        x_p = self.proj(x)        # [B, c_mid, H, W]

        # local
        loc = self.loc_dw(x_p)
        loc = self.loc_pw(loc)

        # surround
        sur = self.sur_dw(x_p)
        sur = self.sur_pw(sur)

        if self.use_concat:
            f = torch.cat([loc, sur], dim=1)    # [B, 2*c_mid, H, W]
        else:
            f = loc + sur                        # [B, c_mid, H, W]  (element-wise add)

        f = self.post_bn_prelu(f)

        # channel-wise attention
        b, c, h, w = f.shape
        g = self.global_pool(f).view(b, c)
        g = self.fc1(g)
        g = self.relu(g)
        g = self.fc2(g)
        g = self.sigmoid(g).view(b, c, 1, 1)

        f = f * g

        out = self.final_conv(f)   # 恢复到 c2 通道
        return out


# import torch
# import torch.nn as nn


# # ===== YOLO 风格 Conv 封装 =====
# def autopad(k, p=None, d=1):  # kernel, padding, dilation
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
#     return p


# class Conv(nn.Module):
#     default_act = nn.SiLU()

#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))


# # ===== 上下文引导下采样模块 =====
# class ContextGuidedDownsample(nn.Module):
#     def __init__(self, c1, c2, k=3, s=2, reduction=16):
#         super().__init__()

#         # 下采样 + 通道对齐
#         self.conv1x1 = Conv(c1, c2, k=1, s=s)

#         # 局部特征
#         self.local_conv = Conv(c2, c2, k=k, s=1)

#         # 周围特征 (DWConv + PWConv)
#         self.surround_dwconv = Conv(c2, c2, k=3, s=1, d=2, g=c2)  # 深度卷积
#         self.surround_pwconv = Conv(c2, c2, k=1, s=1)

#         # 拼接后的 BN + PReLU
#         self.bn_prelu = nn.Sequential(
#             nn.BatchNorm2d(c2 * 2),
#             nn.PReLU()
#         )

#         # 全局上下文 (类似 SE 注意力)
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Linear(c2 * 2, c2 * 2 // reduction, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(c2 * 2 // reduction, c2 * 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#         # 融合
#         self.fusion_conv = Conv(c2 * 2, c2, k=1, s=1)

#     def forward(self, x):
#         x_proj = self.conv1x1(x)

#         f_loc = self.local_conv(x_proj)

#         f_sur = self.surround_dwconv(x_proj)
#         f_sur = self.surround_pwconv(f_sur)

#         f_cat = torch.cat([f_loc, f_sur], dim=1)
#         f_cat = self.bn_prelu(f_cat)

#         b, c, _, _ = f_cat.size()
#         g = self.global_pool(f_cat).view(b, c)
#         g = self.fc1(g)
#         g = self.relu(g)
#         g = self.fc2(g)
#         g = self.sigmoid(g).view(b, c, 1, 1)

#         f_out = f_cat * g
#         f_out = self.fusion_conv(f_out)
#         return f_out


# # ===== CGB: 封装成 YOLO cfg 可调用的 Block =====
# class CGB(nn.Module):
#     def __init__(self, c1, c2, k=3, s=2):
#         """
#         Context-Guided Block (CGB), 替换 YOLO backbone 里的 Conv 下采样模块
#         Args:
#             c1: 输入通道数
#             c2: 输出通道数
#             k: 卷积核大小（默认 3）
#             s: 步长（默认 2，用于下采样）
#         """
#         super().__init__()
#         self.cgd = ContextGuidedDownsample(c1, c2, k=k, s=s)

#     def forward(self, x):
#         return self.cgd(x)
