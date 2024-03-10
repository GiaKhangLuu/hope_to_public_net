from detectron2.model_zoo.configs.common.optim import SGD as optimizer
from detectron2.model_zoo.configs.common.coco_schedule import lr_multiplier_1x as lr_multiplier
from detectron2.model_zoo.configs.common.data.coco import dataloader
from detectron2.model_zoo.configs.common.train import train

from khang_net.configs.yolof.base_yolof import model

dataloader.train.mapper.use_instance_mask = False
#model.backbone.bottom_up.freeze_at = 2
optimizer.lr = 0.01

train['init_checkpoint'] = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"