from detectron2.config import LazyCall as L
from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.model_zoo.configs.common.data.constants import constants
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.matcher import Matcher

from khang_net.modeling.meta_arch.huflit_net import HUFLIT_Net
from khang_net.modeling.mask_head import MaskRCNNConvUpsampleHead
from khang_net.modeling.attention.spatial_attention import SpatialAttention
#from khang_net.configs.yolof.yolof_se import model as yolof
from khang_net.configs.yolof.yolof_v_39_ese import model as yolof

yolof.num_classes = 9

model=L(HUFLIT_Net)(
    yolof=yolof,
    pooler=L(ROIPooler)(
        output_size=14,
        scales=(1.0 / 32,),
        sampling_ratio=0,
        pooler_type="ROIAlignV2",
    ),
    mask_head=L(MaskRCNNConvUpsampleHead)(
        input_shape=L(ShapeSpec)(
            # Set channels to 512 in order for the compatibility with 'p5' 
            # (output from encoder of yolof)
            channels=512,
            width=14,
            height=14
        ),
        num_classes=yolof.num_classes,
        conv_dims=[256, 256, 256, 256, 256],
        spatial_attention=L(SpatialAttention)(
            kernel_size=7
        )
    ),
    proposal_matcher=L(Matcher)(
        thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
    ),
    num_classes=yolof.num_classes,
    batch_size_per_image=512,
    positive_fraction=0.25,
    pixel_mean=constants['imagenet_bgr256_mean'],
    pixel_std=constants['imagenet_bgr256_std'],
    input_format="BGR",
    train_yolof=False
)