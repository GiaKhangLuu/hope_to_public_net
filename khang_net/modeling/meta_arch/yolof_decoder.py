import math
from typing import List, Tuple

import torch
import torch.nn as nn

from detectron2.layers import ShapeSpec, get_norm

#from ..layers.batch_norm import get_norm

class YOLOFDecoder(nn.Module):
    """
    Head Decoder for YOLOF.

    This module contains two types of components:
        - A classification head with two 3x3 convolutions and one
            classification 3x3 convolution
        - A regression head with four 3x3 convolutions, one regression 3x3
          convolution, and one implicit objectness 3x3 convolution
    """

    def __init__(
        self,
        *,
        input_shape: ShapeSpec,
        num_classes: int,
        num_anchors: int,
        cls_num_convs: int,
        reg_num_convs: int,
        norm="",
        prior_prob=0.01
    ):
        """
        Args:
            TODO: Write document 
        """        

        super().__init__()

        self.INF = 1e8
        self.num_classes = num_classes
        self.prior_prob = prior_prob

        cls_subnet, bbox_subnet = [], []
        for i in range(cls_num_convs):
            cls_subnet.append(
                nn.Conv2d(input_shape.channels, input_shape.channels, 
                          kernel_size=3, stride=1, padding=1))
            norm_layer = get_norm(norm, input_shape.channels)
            if norm_layer:
                cls_subnet.append(norm_layer)
            cls_subnet.append(nn.ReLU(inplace=True))

        for i in range(reg_num_convs):
            bbox_subnet.append(
                nn.Conv2d(input_shape.channels, input_shape.channels, 
                          kernel_size=3, stride=1, padding=1))
            norm_layer = get_norm(norm, input_shape.channels)
            if norm_layer:
                bbox_subnet.append(norm_layer)
            bbox_subnet.append(nn.ReLU(inplace=True))

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)

        self.cls_score = nn.Conv2d(input_shape.channels, 
                                   num_anchors * num_classes, 
                                   kernel_size=3, stride=1, padding=1)

        self.bbox_pred = nn.Conv2d(input_shape.channels, 
                                   num_anchors * 4, 
                                   kernel_size=3, stride=1, padding=1)

        self.object_pred = nn.Conv2d(input_shape.channels, 
                                     num_anchors, 
                                     kernel_size=3, stride=1, padding=1)
        
        self._init_weight()
        
    def _init_weight(self):
        for m in self.modules():
            if m:
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_score = self.cls_score(self.cls_subnet(feature))
        N, _, H, W = cls_score.shape
        cls_score = cls_score.view(N, -1, self.num_classes, H, W)

        reg_feat = self.bbox_subnet(feature)
        bbox_reg = self.bbox_pred(reg_feat)
        objectness = self.object_pred(reg_feat)

        # implicit objectness
        objectness = objectness.view(N, -1, 1, H, W)
        normalized_cls_score = cls_score + objectness - torch.log(
            1. + torch.clamp(cls_score.exp(), max=self.INF) + torch.clamp(
                objectness.exp(), max=self.INF))
        normalized_cls_score = normalized_cls_score.view(N, -1, H, W)
        return normalized_cls_score, bbox_reg
