import torch.nn as nn
from detectron2.layers import Conv2d

from detectron2.layers import  get_norm

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_rate=16, conv_norm=""):
        super(ChannelAttention, self).__init__()
        self.squeeze = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1)
        ])
        self.excitation = nn.Sequential(
            Conv2d(in_channels=channels,
                   out_channels=channels // reduction_rate,
                   kernel_size=1,
                   bias=False,
                   norm=get_norm(conv_norm, channels // reduction_rate),
                   activation=nn.ReLU()
            ),
            Conv2d(in_channels=channels // reduction_rate,
                   out_channels=channels,
                   kernel_size=1,
                   bias=False,
                   norm=get_norm(conv_norm, channels))
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # perform squeeze with independent Pooling
        avg_feat = self.squeeze[0](x)
        max_feat = self.squeeze[1](x)
        # perform excitation with the same excitation sub-net
        avg_out = self.excitation(avg_feat)
        max_out = self.excitation(max_feat)
        # attention
        attention = self.sigmoid(avg_out + max_out)
        return attention * x
