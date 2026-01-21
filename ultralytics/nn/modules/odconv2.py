# odconv_improved.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv  # 保持与 Ultralytics 的 Conv 对接（如需移植，可替换）

class Attention(nn.Module):
    """
    Improved attention used by ODConv. Allows enabling/disabling specific attention branches
    for easy ablation experiments.
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, groups=1,
                 reduction=0.0625, kernel_num=4, min_channel=16,
                 enable_channel=True, enable_filter=False, enable_spatial=False, enable_kernel=True):
        super().__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        # flags to enable/disable branches (for ablation)
        self.enable_channel = enable_channel
        self.enable_filter = enable_filter
        self.enable_spatial = enable_spatial
        self.enable_kernel = enable_kernel

        # shared trunk: global pooling -> bottleneck fc
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        # channel attention
        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        # filter attention (per-out-channel)
        if out_planes is None:
            out_planes = in_planes
        if not (in_planes == groups and in_planes == out_planes):
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
        # spatial attention (per-kernel spatial map)
        if kernel_size != 1:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
        # kernel attention (weight across kernel_num)
        if kernel_num != 1:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip_out(shape_like):
        # helper to return ones (broadcastable) for skip branches
        return torch.ones_like(shape_like)

    def get_channel_attention(self, x):
        # returns shape (B, C, 1, 1)
        out = self.channel_fc(x)
        out = torch.sigmoid(out.view(x.size(0), -1, 1, 1) / self.temperature)
        return out

    def get_filter_attention(self, x):
        # returns shape (B, outC, 1, 1)
        out = self.filter_fc(x)
        out = torch.sigmoid(out.view(x.size(0), -1, 1, 1) / self.temperature)
        return out

    def get_spatial_attention(self, x):
        # returns shape (B, 1, 1, 1, k, k) to multiply with weight tensor later
        out = self.spatial_fc(x)
        out = out.view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        out = torch.sigmoid(out / self.temperature)
        return out

    def get_kernel_attention(self, x):
        # returns shape (B, kernel_num, 1, 1, 1, 1)
        out = self.kernel_fc(x)
        out = out.view(x.size(0), -1, 1, 1, 1, 1)
        out = F.softmax(out / self.temperature, dim=1)
        return out

    def forward(self, x):
        """
        x: feature map (B, C_in, H, W)
        returns: a tuple of four attention tensors or 1.0 (if disabled)
        """
        b = x.size(0)
        z = self.avgpool(x)       # (B, C_in, 1, 1)
        z = self.fc(z)
        z = self.relu(z)

        # channel attention
        if self.enable_channel:
            channel_att = self.get_channel_attention(z)  # (B, C, 1, 1)
        else:
            channel_att = torch.ones((b, self.channel_fc.out_channels if hasattr(self, 'channel_fc') else x.size(1), 1, 1), device=x.device)

        # filter attention
        if self.enable_filter and hasattr(self, 'filter_fc'):
            filter_att = self.get_filter_attention(z)  # (B, outC, 1, 1)
        else:
            # default to ones with outC = (if filter_fc exists else in_planes)
            outc = self.filter_fc.out_channels if hasattr(self, 'filter_fc') else x.size(1)
            filter_att = torch.ones((b, outc, 1, 1), device=x.device)

        # spatial attention
        if self.enable_spatial and hasattr(self, 'spatial_fc'):
            spatial_att = self.get_spatial_attention(z)  # (B,1,1,1,k,k)
        else:
            # ones corresponding to kernel spatial shape
            spatial_att = torch.ones((b, 1, 1, 1, self.kernel_size, self.kernel_size), device=x.device)

        # kernel attention
        if self.enable_kernel and hasattr(self, 'kernel_fc'):
            kernel_att = self.get_kernel_attention(z)  # (B, K, 1,1,1,1)
        else:
            kernel_att = torch.ones((b, self.kernel_num, 1, 1, 1, 1), device=x.device)
            # if kernel attention isn't used, normalize to uniform distribution
            kernel_att = kernel_att / float(self.kernel_num)

        return channel_att, filter_att, spatial_att, kernel_att


class ODConv2d(nn.Module):
    """
    Improved ODConv2d with:
      - flags to enable/disable attention branches (via Attention enable_* args)
      - residual blend option to mix static & adaptive outputs
      - method to extract attentions for visualization
      A1=enable_channel=True, A2=enable_filter=False, A3=enable_spatial=False, A4=enable_kernel=False
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4, blend=0.0,
                 enable_channel=False, enable_filter=False, enable_spatial=True, enable_kernel=False):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.blend = float(blend)  # how much to blend static conv: output = blend*static + (1-blend)*dynamic

        # attention module with enable flags
        self.attention = Attention(in_planes, out_planes, kernel_size=kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num, min_channel=16,
                                   enable_channel=enable_channel, enable_filter=enable_filter,
                                   enable_spatial=enable_spatial, enable_kernel=enable_kernel)

        # K candidate weights: shape (K, outC, inC/groups, k, k)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size), requires_grad=True)
        # optional bias (not used here)
        self.bias = None

        # static kernel used for residual blend (first kernel)
        # we keep self.weight[0] as representative static kernel
        self._initialize_weights()

        # choose forward implementation
        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def get_attentions(self, x):
        """
        Convenience method to get attentions for visualization on input x.
        Returns the tuple (channel_att, filter_att, spatial_att, kernel_att)
        """
        return self.attention(x)

    def _forward_impl_common(self, x):
        # forward with dynamic aggregation
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, H, W = x.size()

        # apply channel attention to input feature maps
        x_att = x * channel_attention  # broadcast (B, C, 1,1)

        # reshape for grouped convolution trick: merge batch and channel groups
        x_reshaped = x_att.reshape(1, -1, H, W)  # (1, B*C_in, H, W)

        # aggregate weights across kernel branches:
        # spatial_attention: (B,1,1,1,k,k)
        # kernel_attention: (B,K,1,1,1,1)
        # weight: (K, outC, inC/groups, k, k) -> unsqueeze(0) -> (1, K, outC, inC/groups, k, k)
        # agg = spatial_attention * kernel_attention.unsqueeze(2) * self.weight.unsqueeze(0)
        agg = spatial_attention * kernel_attention * self.weight.unsqueeze(0)

        # sum over kernel dimension -> shape (B, outC, inC/groups, k, k)
        agg = torch.sum(agg, dim=1)
        # reshape to match grouped conv needed weights dimension:
        # current agg: (B, outC, inC/groups, k, k)
        # we want a single conv with groups = groups * batch_size
        agg = agg.view(-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size)

        # perform grouped conv: groups = self.groups * batch_size
        output = F.conv2d(x_reshaped, weight=agg, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        # reshape output back to (B, outC, H_out, W_out)
        out_h = output.size(-2)
        out_w = output.size(-1)
        output = output.view(batch_size, self.out_planes, out_h, out_w)

        # apply filter attention if present
        output = output * filter_attention

        # residual blend with static kernel if requested
        if self.blend > 0.0:
            # static conv using first kernel self.weight[0]
            static_out = F.conv2d(x, weight=self.weight[0], bias=None, stride=self.stride,
                                   padding=self.padding, dilation=self.dilation, groups=self.groups)
            # static_out shape: (B, outC, H_out, W_out)
            output = self.blend * static_out + (1.0 - self.blend) * output

        return output

    def _forward_impl_pw1x(self, x):
        # simplified path for 1x1 kernels
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x_att = x * channel_attention
        # single static weight (weights squeezed)
        output = F.conv2d(x_att, weight=self.weight.squeeze(0), bias=None, stride=self.stride,
                          padding=self.padding, dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        if self.blend > 0.0:
            static_out = F.conv2d(x, weight=self.weight[0].squeeze(0), bias=None, stride=self.stride,
                                  padding=self.padding, dilation=self.dilation, groups=self.groups)
            output = self.blend * static_out + (1.0 - self.blend) * output
        return output

    def forward(self, x):
        return self._forward_impl(x)


# Example Bottleneck and C2f modules (compatible with your code)
class Bottleneck_OD(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, blend=0.0, **att_flags):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = ODConv2d(c_, c2, kernel_size=k[1][0], stride=1, groups=g, blend=blend, **att_flags)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f_ODv2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, blend=0.0, **att_flags):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck_OD(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0, blend=blend, **att_flags) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


if __name__ == '__main__':
    # quick sanity test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    odm = ODConv2d(128, 256, kernel_size=3, stride=1, padding=1, groups=1, kernel_num=4, blend=0.2,
                   enable_channel=True, enable_filter=True, enable_spatial=True, enable_kernel=True).to(device)
    x = torch.randn(8, 128, 64, 64).to(device)
    out = odm(x)
    print('Input shape:', x.shape)
    print('Output shape:', out.shape)

    # get attentions for visualization
    ch_att, f_att, s_att, k_att = odm.get_attentions(x)
    print('channel_att shape:', ch_att.shape)
    print('filter_att shape:', f_att.shape)
    print('spatial_att shape:', s_att.shape)
    print('kernel_att shape:', k_att.shape)
