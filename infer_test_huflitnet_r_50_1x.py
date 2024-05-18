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
import json
import cv2
import os
from tqdm import tqdm

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.model_zoo import get_config
from detectron2.config import LazyConfig
from detectron2.config.instantiate import instantiate
from khang_net.engine.default_predictor import DefaultPredictor

from pycocotools.mask import encode as cvt_mask_to_rle
from pycocotools.mask import decode as cvt_rle_to_mask

dataset = 'coco2017'
annot_dir = './coco2017/annotations'
imgs_dir = './coco2017/{}2017'

for split in ['train', 'val']:
    annot_path = os.path.join(annot_dir, f'instances_{split}2017.json')
    d_name = dataset + f'_{split}'
    register_coco_instances(d_name, {}, annot_path, imgs_dir.format(split))

# Load dataset
dataset_dicts = DatasetCatalog.get('coco2017_val')
metadata = MetadataCatalog.get('coco2017_val')

config_file = "khang_net/configs/huflit_net/huflitnet_r_50_1x.py"
alg = config_file.split('/')[-1][:-3]
checkpoint_file = "./output_{}/model_final.pth".format(alg)
annot_info_file = './coco2017/coco_test_annotations/image_info_test-dev2017.json'
img_dir = './coco2017/coco_test2017'
results_file = './detections_test-dev2017_r501x_results.json'

cfg = LazyConfig.load(config_file)
cfg.train.device = 'cuda:3'
cfg.dataloader.evaluator.dataset_name = 'coco2017_val'
cfg.dataloader.train.dataset.names = 'coco2017_train'
cfg.dataloader.test.dataset.names = 'coco2017_val'
cfg.dataloader.train.total_batch_size = 16

cfg.model.num_classes = 80
cfg.model.yolof.num_classes = 80
cfg.model.mask_head.num_classes = 80

cfg.train.init_checkpoint = checkpoint_file

predictor = DefaultPredictor(cfg)
predictor.model.to(cfg.train.device)

coco_ids = list(MetadataCatalog.get("coco2017_val").thing_dataset_id_to_contiguous_id.keys())
model_ids = list(MetadataCatalog.get("coco2017_val").thing_dataset_id_to_contiguous_id.values())

if os.path.exists(results_file):
    os.remove(results_file)
else:
    fp = open(results_file, 'w')
    fp.close()

with open(annot_info_file, 'r') as file:
    annots = json.load(file)

num_imgs = len(annots['images'])

annType = ['segm','bbox','keypoints']
annType = annType[0]

def convert_bbox_xyxy_to_xywh(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]

with open(results_file, 'a') as file:
    for i, image in enumerate(tqdm(annots['images'])):

        if i == 0:
            file.write('[')
        else:
            file.write(',')

        file_name = image['file_name']
        image_id = image['id']
        results_each_img = []

        if (file_name is None) or (image_id is None):
            print('file_name or image_id is null')

        image_path = os.path.join(img_dir, file_name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image: {image_path}")

        predictions = predictor(img)
        output = predictions['instances']
        pred_boxes = output.pred_boxes.tensor.to('cpu')
        pred_masks = output.pred_masks.to('cpu')
        pred_cls_ids = output.pred_classes.to('cpu')
        pred_confs = output.scores.to('cpu')
        assert pred_boxes.size()[0] == pred_masks.size()[0] == pred_cls_ids.size()[0] == pred_confs.size()[0]

        for j in range(pred_boxes.size()[0]):
            box, mask, model_cls_id, conf = pred_boxes[j], pred_masks[j], pred_cls_ids[j], pred_confs[j]
            pred_coco_id = coco_ids[model_cls_id.item()]
            result = {"image_id": image_id,
                      "category_id": pred_coco_id,
                      "score": round(conf.item(), 3)}

            if annType == "bbox":
                result["bbox"] = convert_bbox_xyxy_to_xywh(*box.tolist())
            elif annType == 'segm':
                mask = mask.numpy().astype(np.uint8)
                mask = np.asfortranarray(mask)
                rle = cvt_mask_to_rle(mask)
                rle["counts"] = rle["counts"].decode()
                result["segmentation"] = rle
            else:
                print("Wronge annType")

            results_each_img.append(result)

        # Write result to file
        file.write(json.dumps(results_each_img)[1:-1])

        if i == num_imgs - 1:
            file.write(']')

