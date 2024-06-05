from detectron2.config import LazyCall as L
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.model_zoo.configs.common.data.constants import constants

from ...modeling.meta_arch.yolof import YOLOF
from ...modeling.meta_arch.yolof_encoder import DilatedEncoder
from ...modeling.meta_arch.yolof_decoder import YOLOFDecoder
from ...modeling.anchor_generator import YOLOFAnchorGenerator
from ...modeling.box2box_transform import Box2BoxTransform
from ...modeling.uniform_matcher import UniformMatcher
from ...modeling.backbone.resnet import ResNet, BasicStem 

model=L(YOLOF)(
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=101,
            stride_in_1x1=True,
            norm="FrozenBN"
        ),
    ),
    encoder=L(DilatedEncoder)(
        input_shape=ShapeSpec(channels=2048),
        out_channels=512,
        block_mid_channels=128, 
        num_residual_blocks=4,
        block_dilations=[2, 4, 6, 8],
        norm='FrozenBN'
    ),
    decoder=L(YOLOFDecoder)(
        input_shape=ShapeSpec(channels=512),
        num_classes="${..num_classes}",
        num_anchors=5,
        cls_num_convs=2,
        reg_num_convs=4,
        norm='FrozenBN',
        prior_prob=0.01
    ),
    anchor_generator=L(YOLOFAnchorGenerator)(
        sizes=[[32, 64, 128, 256, 512]],
        aspect_ratios=[[1.0]],
        strides=[32],
        offset=0.0,
    ),
    box2box_transform=L(Box2BoxTransform)(
        weights=[1.0, 1.0, 1.0, 1.0],
        add_ctr_clamp=True
    ),
    anchor_matcher=L(UniformMatcher)(
        match_topk=4
    ),
    num_classes=80,
    focal_loss_alpha=0.25,
    focal_loss_gamma=2.0,
    pixel_mean=constants['imagenet_bgr256_mean'],
    pixel_std=constants['imagenet_bgr256_std'],
    input_format="BGR",
    neg_ignore_thresh=0.7,
    pos_ignore_thresh=0.15,
    score_thresh_test=0.05,
    topk_candidates_test=1000,
    nms_thresh_test=0.6,
    max_detections_per_image=100
)

