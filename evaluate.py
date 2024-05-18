import sys
#sys.path.insert(0, './detectron2')

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances

dataset = 'coco2017'
annot_dir = './coco2017/annotations'
imgs_dir = './coco2017/{}2017'

#annot_dir = './coco_test_annotations'
#imgs_dir = './coco_test2017'


for split in ['train', 'val']:
    annot_path = os.path.join(annot_dir, f'instances_{split}2017.json')
    d_name = dataset + f'_{split}'
    register_coco_instances(d_name, {}, annot_path, imgs_dir.format(split))


#annot_path = os.path.join(annot_dir, 'image_info_test-dev2017.json')
#register_coco_instances('coco2017_test-dev', {}, annot_path, imgs_dir)

# Load dataset
dataset_dicts = DatasetCatalog.get('coco2017_val')
metadata = MetadataCatalog.get('coco2017_val')

from detectron2.model_zoo import get_config
from detectron2.config import LazyConfig
from detectron2.config.instantiate import instantiate
from detectron2.engine import DefaultPredictor

cfg = LazyConfig.load("khang_net/configs/huflit_net/huflitnet_r_50_se_3x.py")

cfg.train.device = 'cuda:1'
cfg.dataloader.evaluator.dataset_name = 'coco2017_val'
cfg.dataloader.train.dataset.names = 'coco2017_train'
cfg.dataloader.test.dataset.names = 'coco2017_val'
cfg.dataloader.train.total_batch_size = 16

cfg.model.num_classes = 80
cfg.model.yolof.num_classes = 80
cfg.model.mask_head.num_classes = 80
#cfg.model.yolof.score_thresh_test = 0.7
#cfg.model.yolof.max_detections_per_image = 50

cfg.train.init_checkpoint = "./output_huflitnet_r_50_se_3x/model_0259999.pth"

model = instantiate(cfg.model)
model.to(cfg.train.device)

from detectron2.checkpoint import DetectionCheckpointer
DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

from detectron2.evaluation import inference_on_dataset

inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test),
            instantiate(cfg.dataloader.evaluator),
)