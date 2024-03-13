from detectron2.model_zoo.configs.common.train import train

from khang_net.configs.yolof.base_yolof import model
from khang_net.configs.yolof.optim import YOLOF_SGD as optimizer
from khang_net.configs.yolof.coco_schedule import lr_multiplier_1x as lr_multiplier
from khang_net.configs.yolof.coco_dataloader import dataloader

dataloader.train.mapper.use_instance_mask = False

model.backbone.freeze_at = 2

train['init_checkpoint'] = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
train['max_iter'] = 22500
