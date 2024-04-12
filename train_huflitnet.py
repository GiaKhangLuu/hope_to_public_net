import sys
sys.path.insert(0, './detectron2')

import argparse
import sys
import os
import torch, detectron2

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.model_zoo import get_config
from detectron2.config import LazyConfig
from detectron2.config.instantiate import instantiate
from detectron2.engine import default_setup

from tools.lazyconfig_train_net import do_train

dataset = 'cityscapes'
annot_dir = './cityscapes/annotations'
imgs_dir = './cityscapes'

for split in ['train', 'val']:
    annot_path = os.path.join(annot_dir, f'instancesonly_filtered_gtFine_{split}.json')
    d_name = dataset + f'_{split}'
    register_coco_instances(d_name, {}, annot_path, imgs_dir)


class Args(argparse.Namespace):
    config_file='khang_net/configs/huflit_net/huflit_net_1x.py'
    eval_only=False
    num_gpus=1
    num_machines=1
    resume=True

args = Args()


cfg = LazyConfig.load("khang_net/configs/huflit_net/huflit_net_1x.py")
cfg.train.device = 'cuda'
cfg.dataloader.evaluator.dataset_name = 'cityscapes_val'
cfg.dataloader.train.dataset.names = 'cityscapes_train'
cfg.dataloader.test.dataset.names = 'cityscapes_val'
cfg.dataloader.train.total_batch_size = 16

batch_on_paper = 16
actual_batch = cfg.dataloader.train.total_batch_size
lr_scale = actual_batch / batch_on_paper
cfg.optimizer.lr = cfg.optimizer.lr * lr_scale
cfg.optimizer.params.base_lr = cfg.optimizer.params.base_lr * lr_scale
cfg.optimizer.params.bias_lr_factor = cfg.optimizer.params.bias_lr_factor * lr_scale
cfg.optimizer.params.backbone_lr_factor = cfg.optimizer.params.backbone_lr_factor * lr_scale

cfg.model.num_classes = 8
cfg.model.yolof.num_classes = 8
cfg.model.mask_head.num_classes = 8

#cfg.train.eval_period = 100000
#cfg.train.checkpointer.period = 1000
cfg.model.yolof_weight = None
cfg.model.train_yolof = True

cfg.optimizer.params.base_lr = 0.01
cfg.optimizer.lr = 0.01

cfg.train.max_iter = 30000
#cfg.train.init_checkpoint = './huflitnet_10k_iters/model_0009999.pth'

default_setup(cfg, args)

do_train(args, cfg)