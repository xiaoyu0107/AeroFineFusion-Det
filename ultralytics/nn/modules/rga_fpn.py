# file: ultralytics/nn/modules/rga_fpn.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class RGABlock(nn.Module):
    """
    Region-Guided Aggregation Block
    - feat_low:  低层特征, [B, C_l, H_l, W_l]  (高分辨率、细节好)
    - feat_high: 高层特征, [B, C_h, H_h, W_h]  (语义强，分辨率低)
    - 内部自动对齐分辨率到低层，生成区域引导图，再做加权融合。
    输出:
      fused: [B, C_out, H_l, W_l]
    """

    def __init__(self, c_low, c_high, c_out=None):
        super().__init__()
        if c_out is None:
            c_out = c_low

        self.proj_low = nn.Conv2d(c_low, c_out, 1, bias=False)
        self.proj_high = nn.Conv2d(c_high, c_out, 1, bias=False)

        # 区域引导：根据低层特征生成显著图
        self.region_conv = nn.Conv2d(1, 1, 3, padding=1, bias=True)
        self.fuse_conv = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)

        self.act = nn.SiLU(inplace=True)

    def forward(self, feat_low, feat_high):
        """
        feat_low: [B,C_l,H_l,W_l]
        feat_high: [B,C_h,H_h,W_h]
        """
        B, Cl, Hl, Wl = feat_low.shape
        _, Ch, Hh, Wh = feat_high.shape

        # 高层特征下采样/上采样到低层分辨率
        feat_high_rs = F.interpolate(
            feat_high, size=(Hl, Wl), mode='bilinear', align_corners=False
        )

        # 区域引导图：低层特征按通道平均 -> 1 通道显著图
        region = feat_low.mean(dim=1, keepdim=True)  # [B,1,Hl,Wl]
        region_score = torch.sigmoid(self.region_conv(region))  # [B,1,Hl,Wl]

        low_proj = self.proj_low(feat_low)
        high_proj = self.proj_high(feat_high_rs)

        # 区域引导加权融合
        fused = region_score * high_proj + (1.0 - region_score) * low_proj
        fused = self.fuse_conv(fused)
        fused = self.act(fused + low_proj)  # 残差增强稳定性

        return fused
class RGAFPNNeck(nn.Module):
    """
    RGA-FPN Neck
    适合替换 YOLOv8 的 PAN/FPN 结构，用于无人机小目标检测。
    输入:
      feats: list[Tensor], len=3
        [x_small, x_mid, x_large]，例如来自 backbone 的 [P3,P4,P5]
        每个 shape: [B, C_i, H_i, W_i]
    输出:
      list[Tensor], len=3
        [y_small, y_mid, y_large]，通道统一为 out_channels
    """

    def __init__(self, in_channels, out_channels=256):
        """
        in_channels: list[int]，长度=3，比如 [64, 128, 256]
        out_channels: 融合后的统一通道数，比如 256
        """
        super().__init__()
        assert len(in_channels) == 3, "RGAFPNNeck 目前实现的是 3 层输入版本"

        c_small, c_mid, c_large = in_channels

        # 统一通道数
        self.proj_small = nn.Conv2d(c_small, out_channels, 1, bias=False)
        self.proj_mid = nn.Conv2d(c_mid, out_channels, 1, bias=False)
        self.proj_large = nn.Conv2d(c_large, out_channels, 1, bias=False)

        # Top-Down: 大→中→小
        self.rga_mid_td = RGABlock(c_low=out_channels, c_high=out_channels, c_out=out_channels)
        self.rga_small_td = RGABlock(c_low=out_channels, c_high=out_channels, c_out=out_channels)

        # Bottom-Up: 小→中→大
        self.down_small = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.down_mid = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False)

        self.rga_mid_bu = RGABlock(c_low=out_channels, c_high=out_channels, c_out=out_channels)
        self.rga_large_bu = RGABlock(c_low=out_channels, c_high=out_channels, c_out=out_channels)

    def forward(self, feats):
        """
        feats: list[Tensor] = [x_small, x_mid, x_large]
        """
        x_small, x_mid, x_large = feats  # 例如 [P3, P4, P5]

        # 统一通道
        s = self.proj_small(x_small)
        m = self.proj_mid(x_mid)
        l = self.proj_large(x_large)

        # ---------- Top-Down 路径 ----------
        # 大尺度 → 中尺度
        l_up = F.interpolate(l, size=m.shape[2:], mode='bilinear', align_corners=False)
        m_td = self.rga_mid_td(feat_low=m, feat_high=l_up)  # [B,out,Hm,Wm]

        # 中尺度 → 小尺度
        m_up = F.interpolate(m_td, size=s.shape[2:], mode='bilinear', align_corners=False)
        s_td = self.rga_small_td(feat_low=s, feat_high=m_up)  # [B,out,Hs,Ws]

        # ---------- Bottom-Up 路径 ----------
        # 小尺度 → 中尺度
        s_down = self.down_small(s_td)  # [B,out,Hm,Wm]
        m_bu = self.rga_mid_bu(feat_low=m_td, feat_high=s_down)

        # 中尺度 → 大尺度
        m_down = self.down_mid(m_bu)    # [B,out,Hl,Wl]
        l_bu = self.rga_large_bu(feat_low=l, feat_high=m_down)

        # 输出 3 个尺度
        y_small = s_td
        y_mid = m_bu
        y_large = l_bu

        return [y_small, y_mid, y_large]
