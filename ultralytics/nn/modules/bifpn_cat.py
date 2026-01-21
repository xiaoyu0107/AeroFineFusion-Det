import torch
import torch.nn as nn




class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


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


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


import torch
import torch.nn as nn
import torch.nn.functional as F

# 可选的轻量注意力模块：SE
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w.expand_as(x)#expanda_as把w广播成x的形状
    
class EMA(nn.Module):
    def __init__(self, channels, factor=16):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)



class BiFPN_Concatv2(nn.Module):
    def __init__(self, dimension=1, weight_count=2):
        super(BiFPN_Concatv2, self).__init__()
        self.d = dimension
        self.w_count = weight_count
        self.w = nn.Parameter(torch.ones(self.w_count, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4

        # 初始化 SE 注意力模块，通道数动态设置（forward 内做 lazy init）
        self.attention = nn.ModuleList([None] * self.w_count)

    def forward(self, x):
        # print("SEBlock")
        # 初始化 attention 模块（lazy init）
        for i in range(self.w_count):
            if self.attention[i] is None:
                self.attention[i] = SEBlock(x[i].shape[1]).to(x[i].device)

        # Softmax 归一化权重，更平滑地引导信息融合
        weight = F.softmax(self.w, dim=0)

        result = []
        for i in range(self.w_count):
            xi = self.attention[i](x[i])        # 加入注意力增强分支特征
            result.append(weight[i] * xi)       # 加权分支
        return torch.cat(result, dim=self.d)    # 沿指定维度 concat


# class BiFPN_Concatv2(nn.Module):
#     def __init__(self, dimension=1, weight_count=2):
#         super(BiFPN_Concatv2, self).__init__()
#         self.d = dimension
#         self.w_count = weight_count
#         self.w = nn.Parameter(torch.ones(self.w_count, dtype=torch.float32), requires_grad=True)
#         self.epsilon = 1e-4

#         # 动态初始化 CBAM 模块（lazy init）
#         self.attention = nn.ModuleList([None] * self.w_count)

#     def forward(self, x):
#         # print("CBAM")
#         for i in range(self.w_count):
#             if self.attention[i] is None:
#                 self.attention[i] = CBAM(x[i].shape[1]).to(x[i].device)

#         weight = F.softmax(self.w, dim=0)

#         result = []
#         for i in range(self.w_count):
#             xi = self.attention[i](x[i])
#             result.append(weight[i] * xi)
#         return torch.cat(result, dim=self.d)





class BiFPN_Concat(nn.Module):
    def __init__(self, dimension=1, weight_count=2):
        super(BiFPN_Concat, self).__init__()
        # 初始化维度，默认为1，表示将在哪个维度上进行拼接操作  
        self.d = dimension
        # 初始化权重个数，默认为2，表示要处理的分支数量  
        self.w_count = weight_count
        # 初始化可学习的权重参数，初始值均为1，并设置requires_grad=True以便在训练过程中更新  
        self.w = nn.Parameter(torch.ones(self.w_count, dtype=torch.float32), requires_grad=True)
        # 定义一个小的正数epsilon，用于防止分母为0的情况，保证数值稳定性  
        self.epsilon = 0.0001

    def forward(self, x):
        # 获取可学习的权重  
        w = self.w
        # 对权重进行归一化，确保它们的和为1，避免梯度消失或爆炸  
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        # 初始化一个空列表，用于存储加权后的分支  
        result = []
        # 遍历每个分支，应用归一化后的权重，并将结果添加到result列表中  
        for i in range(self.w_count):
            result.append(weight[i] * x[i])
            # 沿着指定的维度（self.d）将加权后的分支连接（concat）起来  
        # 返回连接后的tensor  
        return torch.cat(result, dim=self.d)
    
