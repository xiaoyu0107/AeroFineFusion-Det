import torch.nn as nn


class PPA(nn.Module):
 def __init__(self, in_features, filters) -> None:
     super().__init__()

     # 定义跳跃连接卷积块
     self.skip = conv_block(in_features=in_features,
                            out_features=filters,
                            kernel_size=(1, 1),
                            padding=(0, 0),
                            norm_type='bn',
                            activation=False)

     # 定义连续卷积块
     self.c1 = conv_block(in_features=in_features,
                          out_features=filters,
                          kernel_size=(3, 3),
                          padding=(1, 1),
                          norm_type='bn',
                          activation=True)
     self.c2 = conv_block(in_features=filters,
                          out_features=filters,
                          kernel_size=(3, 3),
                          padding=(1, 1),
                          norm_type='bn',
                          activation=True)
     self.c3 = conv_block(in_features=filters,
                          out_features=filters,
                          kernel_size=(3, 3),
                          padding=(1, 1),
                          norm_type='bn',
                          activation=True)

     # 定义空间注意力模块
     self.sa = SpatialAttentionModule()
     # 定义ECA模块
     self.cn = ECA(filters)
     # 定义局部和全局注意力模块
     self.lga2 = LocalGlobalAttention(filters, 2)
     self.lga4 = LocalGlobalAttention(filters, 4)

     # 定义批归一化层、dropout层和激活函数
     self.bn1 = nn.BatchNorm2d(filters)
     self.drop = nn.Dropout2d(0.1)
     self.relu = nn.ReLU()
     self.gelu = nn.GELU()

 def forward(self, x):
     x_skip = self.skip(x)  # 跳跃连接输出
     x_lga2 = self.lga2(x_skip)  # 局部和全局注意力输出（大小为2的patch）
     x_lga4 = self.lga4(x_skip)  # 局部和全局注意力输出（大小为4的patch）
     x1 = self.c1(x)  # 第一个卷积块输出
     x2 = self.c2(x1)  # 第二个卷积块输出
     x3 = self.c3(x2)  # 第三个卷积块输出
     x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4  # 合并所有输出
     x = self.cn(x)  # ECA模块
     x = self.sa(x)  # 空间注意力模块
     x = self.drop(x)  # Dropout层
     x = self.bn1(x)  # 批归一化层
     x = self.relu(x)  # 激活函数
     return x
