from detectron2.config import LazyCall as L
from ...modeling.meta_arch.yolof import YOLOF
from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.meta_arch.retinanet import RetinaNetHead

from detectron2.model_zoo.configs.common.data.constants import constants

model=L(YOLOF)(
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=50,
            stride_in_1x1=True,
            norm="FrozenBN"
        ),
        out_features=['res5']
    ),
    #encoder=L(...),
    #decoder=L(...),
    head=L(RetinaNetHead)(
        # Shape for each input feature map
        input_shape=[ShapeSpec(channels=2048)],
        num_classes="${..num_classes}",
        conv_dims=[256],
        prior_prob=0.01,
        num_anchors=9
    ),
    head_in_features=['res5'],
    anchor_generator=L(DefaultAnchorGenerator)(
        sizes=[[x, x * 2 ** (1.0 / 3), x * 2 ** (2.0 / 3)] for x in [128]],
        aspect_ratios=[0.5, 1.0, 2.0],
        strides=[32],
        offset=0.0,
    ),
    box2box_transform=L(Box2BoxTransform)(weights=[1.0, 1.0, 1.0, 1.0]),
    anchor_matcher=L(Matcher)(
        thresholds=[0.4, 0.5], labels=[0, -1, 1], allow_low_quality_matches=True
    ),
    num_classes=80,
    focal_loss_alpha=0.25,
    focal_loss_gamma=2.0,
    pixel_mean=constants['imagenet_bgr256_mean'],
    pixel_std=constants['imagenet_bgr256_std'],
    input_format="BGR"
)

