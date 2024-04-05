#from detectron2.model_zoo.configs.common.train import train

from khang_net.configs.huflit_net.train import train
from khang_net.configs.yolof.optim import YOLOF_SGD as optimizer
from khang_net.configs.yolof.coco_schedule import lr_multiplier_1x as lr_multiplier
from khang_net.configs.yolof.coco_dataloader import dataloader
from khang_net.configs.huflit_net.base_huflitnet import model


dataloader.train.mapper.use_instance_mask = True

model.yolof.backbone.freeze_at = 2

train['init_checkpoint'] = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"