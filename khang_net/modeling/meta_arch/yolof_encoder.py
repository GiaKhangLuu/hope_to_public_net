from typing import List

import torch
import torch.nn as nn

from detectron2.layers import ShapeSpec, get_norm

#from ..layers.batch_norm import get_norm
from ..nn_utils.weight_init import c2_xavier_fill 

class DilatedEncoder(nn.Module):
    """
    Dilated Encoder for YOLOF.

    This module contains two types of components:
        - the original FPN lateral convolution layer and fpn convolution layer,
          which are 1x1 conv + 3x3 conv
        - the dilated residual block
    """

    def __init__(
        self, 
        *,
        input_shape: ShapeSpec,
        out_channels: int,
        block_mid_channels: int,
        num_residual_blocks: int,
        block_dilations: List[int],
        norm=""
    ): 
        """
        Args:
            TODO: Write document 
        """        

        super().__init__()

        assert len(block_dilations) == num_residual_blocks

        # Init layer
        self.lateral_conv = nn.Conv2d(input_shape.channels,
                                      out_channels,
                                      kernel_size=1)
        self.lateral_norm = get_norm(norm, out_channels)
        self.fpn_conv = nn.Conv2d(out_channels,
                                  out_channels,
                                  kernel_size=3,
                                  padding=1)
        self.fpn_norm = get_norm(norm, out_channels)
        encoder_blocks = []
        for i in range(num_residual_blocks):
            dilation = block_dilations[i]
            encoder_blocks.append(
                Bottleneck(
                    out_channels,
                    block_mid_channels,
                    dilation=dilation,
                    norm_type=norm
                )
            )
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

        self._init_weight()

    def _init_weight(self):
        c2_xavier_fill(self.lateral_conv)
        c2_xavier_fill(self.fpn_conv)
        for m in [self.lateral_norm, self.fpn_norm]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        for m in self.dilated_encoder_blocks.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        out = self.lateral_norm(self.lateral_conv(feature))
        out = self.fpn_norm(self.fpn_conv(out))
        return self.dilated_encoder_blocks(out)

class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int = 512,
        mid_channels: int = 128,
        dilation: int = 1,
        norm_type: str = 'BN',
    ):

        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            get_norm(norm_type, mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=3, padding=dilation, dilation=dilation),
            get_norm(norm_type, mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            get_norm(norm_type, in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out