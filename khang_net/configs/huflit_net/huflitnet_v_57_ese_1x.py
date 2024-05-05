from khang_net.configs.huflit_net.train import train
from khang_net.configs.yolof.optim import YOLOF_SGD as optimizer
from khang_net.configs.yolof.coco_schedule import lr_multiplier_1x as lr_multiplier
from khang_net.configs.yolof.coco_dataloader import dataloader
#from khang_net.configs.huflit_net.huflitnet_se import model
#from khang_net.configs.huflit_net.huflitnet_se import model
from khang_net.configs.huflit_net.huflitnet_v_57_ese import model

dataloader.train.mapper.use_instance_mask = True
dataloader.train.total_batch_size = 16

train['output_dir'] = "./output_huflitnet_v_57_ese_1x"
train['max_iter'] = 90000
train['eval_period'] = 5000
train['best_checkpointer']['val_metric'] = "segm/AP50"

model.num_classes = 80
model.yolof.num_classes = 80
model.mask_head.num_classes = 80

optimizer.params.base_lr = 0.01
optimizer.lr = 0.01
