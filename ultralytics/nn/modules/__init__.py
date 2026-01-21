# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics modules.

This module provides access to various neural network components used in Ultralytics models, including convolution blocks,
attention mechanisms, transformer components, and detection/segmentation heads.

Examples:
    Visualize a module with Netron.
    >>> from ultralytics.nn.modules import *
    >>> import torch
    >>> import os
    >>> x = torch.ones(1, 128, 40, 40)
    >>> m = Conv(128, 128)
    >>> f = f"{m._get_name()}.onnx"
    >>> torch.onnx.export(m, x, f)
    >>> os.system(f"onnxslim {f} {f} && open {f}")  # pip install onnxslim
"""

from .block import (
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    MaxSigmoidAttnBlock,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    TorchVision,
)
from .conv import (
    VoVGSCSP,
    GSConv,
    VoVGSCSPC,
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    Index,
    LightConv,
    RepConv,
    SpatialAttention,
)
from .head import (
    OBB,
    Classify,
    Detect,
    LRPCHead,
    Pose,
    RTDETRDecoder,
    Segment,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    v10Detect,
)
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)
from .AFPN import Detect_AFPN4
from .CAFM import C2f_AT
from .QA import QuadrangleAttention
from .bifpn import BiFPN_Add,BiFPNBlockP2
from .DWR import C2f_DWR
from .FreqFusion import FreqFusion
from .bifpn_cat import BiFPN_Concat,BiFPN_Concatv2
from .odconv import C2f_OD
from .LSKA import SPPFLK
from .CGB import CGB
from .MSCAM import EMCAD_block
from .MSSPPFLA import  MSSPPFLA
from .MSSPPFLKA import MSSPPFLKA
from .Hyper import HyperComputeModule
from .EfficientViM import C2f_EfficientViM
from .PConv import PConv
from .DGAConv import DGAConv_RKA
from .odconv2 import C2f_ODv2
from .PSA_BiFPN import PSA_BiFPN
from .FSA_RGBlock import C2fTriAttn
from .mdfm import MDFM
from .Moganet import C2f_MultiOGA
from .ADRes2Block import ADRes2Block
from .Down_WT import WADown
from .GraphFPN import GraphFPNNeck
from .rga_fpn import RGAFPNNeck
from .GLA_SPPF import GLA_SPPF,DPSPPF
from .GCSPPF import GCSPPF
from .C2f_TriAD import C2f_TriAD
from .FSA_ODConv import C2f_FSA
from .Detect_SA import Detect_SA
__all__ = (
    "Detect_SA",
    "BiFPNBlockP2",
    "C2f_FSA",
    "C2f_TriAD",
    "DPSPPF",
    "GCSPPF",
    "GLA_SPPF",
    "C2fTriAttn",
    "RGAFPNNeck",
    "GraphFPNNeck",
    "WADown",
    "ADRes2Block",
    "C2f_MultiOGA",
    "MDFM",
    "FSARGBlock",
    "BiFPN_Add",
    "PSA_BiFPN",
    "C2f_FSA",
    "C2f_ODv2",
    "DGAConv_RKA",
    "PConv",
    "C2f_EfficientViM",
    "HyperComputeModule",
    "GSConv",
    "VoVGSCSP",
    "VoVGSCSPC",
    "MSSPPFLKA",
    "MSSPPFLA",
    "EMCAD_block",
    "CGB",
    "SPPFLK",
    "C2f_OD",
    "BiFPN_Concat",
    "BiFPN_Concatv2",
    "FreqFusion",
    "C2f_DWR",
    "QuadrangleAttention",
    "C2f_AT",
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C3k2",
    "SCDown",
    "C2fPSA",
    "C2PSA",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "WorldDetect",
    "YOLOEDetect",
    "YOLOESegment",
    "v10Detect",
    "LRPCHead",
    "ImagePoolingAttn",
    "MaxSigmoidAttnBlock",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "AConv",
    "ELAN1",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "TorchVision",
    "Index",
    "A2C2f",
    "Detect_AFPN4"
)
