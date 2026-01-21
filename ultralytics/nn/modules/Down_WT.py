import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
from .conv import Conv,autopad
from torch.nn import functional as F

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





class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.SiLU(inplace=True),
                                    )
    def forward(self, x):
        # print(f"input: {x.shape}")
        # residual = F.avg_pool2d(x, kernel_size=2)

        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        
        x = self.conv_bn_relu(x)
        # x = x + residual  # 残差连接
        # print(f"output: {x.shape}")
        return x



class Down_wtLHF(nn.Module):
    def __init__(self, in_ch, out_ch, mode='channel'):  # mode: 'scalar' or 'channel'
        super(Down_wtLHF, self).__init__()
        self.mode = mode
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        # 低频分支
        self.low_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

        # 高频分支（拼接 HL, LH, HH）
        self.high_conv = nn.Sequential(
            nn.Conv2d(in_ch * 3, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

        # 可学习融合权重
        if self.mode == 'scalar':
            self.alpha = nn.Parameter(torch.tensor(0.5))  # 单个权重
        elif self.mode == 'channel':
            self.alpha = nn.Parameter(torch.ones(out_ch))  # 每通道一个

        # 输出归一化
        self.out_bn_relu = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )

    def forward(self, x_in):
        # print(x_in.shape)
        yL, yH = self.wt(x_in)
        y_HL = yH[0][:, :, 0, :, :]
        y_LH = yH[0][:, :, 1, :, :]
        y_HH = yH[0][:, :, 2, :, :]

        low = self.low_conv(yL)
        high = self.high_conv(torch.cat([y_HL, y_LH, y_HH], dim=1))
        # print(low.shape)
        # print(high.shape)
        # print(f"Input: {x_in.shape}")
        # print(f"yL (low freq): {yL.shape}")
        # print(f"y_HL: {y_HL.shape}, y_LH: {y_LH.shape}, y_HH: {y_HH.shape}")
        # print(f"low_conv: {low.shape}, high_conv: {high.shape}")


        # 加权融合
        if self.mode == 'scalar':
            x = self.alpha * low + (1 - self.alpha) * high
        elif self.mode == 'channel':
            alpha = self.alpha.view(1, -1, 1, 1)
            x = alpha * low + (1 - alpha) * high

        return self.out_bn_relu(x)

import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward


class WADown(nn.Module):
    """Wavelet Adaptive Downsample Block (WADown)
    自适应频率融合下采样，适配 ADRes2Block
    """
    def __init__(self, in_ch, out_ch, reduction=16, mode='channel'):
        super().__init__()
        self.mode = mode
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        # 低频卷积
        self.low_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )

        # 高频卷积
        self.high_conv = nn.Sequential(
            nn.Conv2d(in_ch * 3, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )

        # 注意力融合
        if mode == 'scalar':
            self.alpha = nn.Parameter(torch.tensor(0.5))
        elif mode == 'channel':
            self.alpha = nn.Parameter(torch.ones(out_ch))

        # 通道注意力增强（提升高频响应）
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // reduction, out_ch, 1, bias=False),
            nn.Sigmoid()
        )

        # 输出归一化
        self.out_norm = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, :, :]
        y_LH = yH[0][:, :, 1, :, :]
        y_HH = yH[0][:, :, 2, :, :]

        low = self.low_conv(yL)
        high = self.high_conv(torch.cat([y_HL, y_LH, y_HH], dim=1))

        # 可学习融合
        if self.mode == 'scalar':
            out = self.alpha * low + (1 - self.alpha) * high
        else:
            alpha = self.alpha.view(1, -1, 1, 1)
            out = alpha * low + (1 - alpha) * high

        # 通道注意力增强
        out = out * self.ca(out)
        out = self.out_norm(out)
        return out




if __name__ =='__main__':

    Down_WT = Down_wt(256,128)
    #创建一个输入张量
    batch_size = 8
    input_tensor=torch.randn(batch_size, 256, 64, 64 )
    #运行模型并打印输入和输出的形状
    output_tensor =Down_WT(input_tensor)
    print("Input shape:",input_tensor.shape)
    print("0utput shape:",output_tensor.shape)



# # Ultralytics YOLO 🚀, AGPL-3.0 license
# # YOLOv8 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect
#
# # Parameters
# nc: 80 # number of classes
# scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
#   # [depth, width, max_channels]
#   n: [0.33, 0.25, 1024] # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
#   s: [0.33, 0.50, 1024] # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
#   m: [0.67, 0.75, 768] # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
#   l: [1.00, 1.00, 512] # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
#   x: [1.00, 1.25, 512] # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
#
# # YOLOv8.0n backbone
# backbone:
#   # [from, repeats, module, args]
#   - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2     1x3x640x640 -> 1x16x320x320
#   - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4   1x16x320x320 -> 1x32x160x160
#   - [-1, 3, C2f, [128, True]]      #1x32x160x160 -> 1x32x160x160
#   - [-1, 1, Down_wt, [256]] # 3-P3/8    1x32x160x160 -> 1x64x80x80
#   - [-1, 6, C2f, [256, True]]              # 1x64x80x80 -> 1x64x80x80
#   - [-1, 1, Down_wt, [512]] # 5-P4/16    1x64x80x80 -> 1x128x40x40
#   - [-1, 6, C2f, [512, True]]             #1x128x40x40-> 1x128x40x40
#   - [-1, 1, Down_wt, [1024]] # 7-P5/32  1x128x40x40-> 1x256x20x20
#   - [-1, 3, C2f, [1024, True]]            # 1x256x20x20-> 1x256x20x20
#   - [-1, 1, SPPF, [1024, 5]] # 9             1x256x20x20-> 1x256x20x20
#
# # YOLOv8.0n head
# head:
#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 1x256x40x40
#   - [[-1, 6], 1, Concat, [1]] # cat backbone P4  # # 1x384x40x40
#   - [-1, 3, C2f, [512]] # 12                       1x128x40x40
#
#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]] #   1x128x80x80
#   - [[-1, 4], 1, Concat, [1]] # cat backbone P3    1x192x80x80
#   - [-1, 3, C2f, [256]] # 15 (P3/8-small)          1x64x80x80
#
#   - [-1, 1, Conv, [256, 3, 2]]                     #1x64x40x40
#   - [[-1, 12], 1, Concat, [1]] # cat head P4        #1x192x40x40
#   - [-1, 3, C2f, [512]] # 18 (P4/16-medium)       #1x128x40x40
#
#   - [-1, 1, Conv, [512, 3, 2]]                     #1x128x20x20
#   - [[-1, 9], 1, Concat, [1]] # cat head P5        #1x384x20x20
#   - [-1, 3, C2f, [1024]] # 21 (P5/32-large)       #1x256x20x20
#
#   - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5)
