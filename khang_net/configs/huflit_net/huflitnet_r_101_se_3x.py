from khang_net.configs.huflit_net.train import train
from khang_net.configs.yolof.optim import YOLOF_SGD as optimizer
from khang_net.configs.yolof.coco_schedule import lr_multiplier_3x as lr_multiplier
from khang_net.configs.yolof.coco_dataloader import dataloader
from khang_net.configs.huflit_net.huflitnet_r_101_se import model

dataloader.train.mapper.use_instance_mask = True
dataloader.train.total_batch_size = 16

train['output_dir'] = "./output_huflitnet_r_101_se_3x"
train['max_iter'] = 90000 * 3
train['eval_period'] = 5000 * 3
train['best_checkpointer']['val_metric'] = "segm/AP50"
train['init_checkpoint'] = "detectron2://ImageNetPretrained/MSRA/R-101.pkl"

model.num_classes = 80
model.yolof.num_classes = 80
model.mask_head.num_classes = 80
model.yolof.backbone.freeze_at = 2

optimizer.params.base_lr = 0.01
optimizer.lr = 0.01