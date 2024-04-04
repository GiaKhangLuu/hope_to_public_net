import logging
import math
from typing import List, Tuple, Optional, Dict, Union
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import Tensor, nn
from torch.nn import functional as F

from detectron2.layers import CycleBatchNormList, ShapeSpec, batched_nms, cat, get_norm, move_device_like, diou_loss
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.box_regression import Box2BoxTransform 
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.meta_arch.dense_detector import permute_to_N_HWA_K  # noqa

from ..box_ops import box_iou, generalized_box_iou
from khang_net.modeling.nms import batched_nms
from khang_net.modeling.postprocessing import detector_postprocess

class YOLOF(nn.Module):
    """
    Implement RetinaNet in :paper:`YOLOF`.
    """

    def __init__(
        self,
        *,
        backbone: Backbone,
        encoder: nn.Module,
        decoder: nn.Module,
        anchor_generator,
        box2box_transform,
        anchor_matcher,
        num_classes,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        max_detections_per_image=100,
        pixel_mean,
        pixel_std,
        vis_period=0,
        input_format="BGR",
        neg_ignore_thresh=0.7,
        pos_ignore_thresh=0.15,
        score_thresh_test=0.05,
        topk_candidates_test=1000,
        nms_thresh_test=0.6
    ):
        
        super().__init__()

        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

        self.num_classes = num_classes

        # Anchors
        self.anchor_generator = anchor_generator
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher
        self.neg_ignore_thresh = neg_ignore_thresh
        self.pos_ignore_thresh = pos_ignore_thresh

        # Loss parameters:
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type
        # Inference parameters:
        self.score_threshold = score_thresh_test
        self.topk_candidates_test = topk_candidates_test
        self.nms_threshold = nms_thresh_test
        self.max_detections_per_image = max_detections_per_image
        # Vis parameters
        self.vis_period = vis_period
        self.input_format = input_format

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)
    
    def preprocess_image(self, batched_inputs: List[Dict[str, Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images
    
    def forward(self, batched_inputs: List[Dict[str, Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            In training, dict[str, Tensor]: mapping from a named loss to a tensor storing the
            loss. Used during training only. In inference, the standard output format, described
            in :doc:`/tutorials/models`.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        # Temporarily hard code this feature out
        features = features['res5']
        features = [features]
        box_cls, box_delta = self.decoder(self.encoder(features[0]))
        anchors = self.anchor_generator(features)
        pred_logits = [permute_to_N_HWA_K(box_cls, self.num_classes)]
        pred_anchor_deltas = [permute_to_N_HWA_K(box_delta, 4)]

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        
            indices = self.get_ground_truth(
                anchors, pred_anchor_deltas, gt_instances)

            losses = self.losses(
                indices, gt_instances, anchors,
                pred_logits, pred_anchor_deltas)
            return losses
        else:
            results = self.inference(
                [box_cls], [box_delta], anchors, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
    
    def losses(self,
               indices,
               gt_instances,
               anchors,
               pred_class_logits,
               pred_anchor_deltas):
        # TODO: redocument shape
        pred_class_logits = cat(
            pred_class_logits, dim=1).view(-1, self.num_classes)
        pred_anchor_deltas = cat(pred_anchor_deltas, dim=1).view(-1, 4)

        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
        N = len(anchors)
        # list[Tensor(R, 4)], one for each image
        all_anchors = Boxes.cat(anchors).tensor
        # Boxes(Tensor(N*R, 4))
        predicted_boxes = self.box2box_transform.apply_deltas(
            pred_anchor_deltas, all_anchors)
        predicted_boxes = predicted_boxes.reshape(N, -1, 4)

        ious = []
        pos_ious = []
        for i in range(N):
            src_idx, tgt_idx = indices[i]
            iou, _ = box_iou(predicted_boxes[i, ...],
                          gt_instances[i].gt_boxes.tensor)
            if iou.numel() == 0:
                max_iou = iou.new_full((iou.size(0),), 0)
            else:
                max_iou = iou.max(dim=1)[0]
            a_iou, _ = box_iou(anchors[i].tensor,
                            gt_instances[i].gt_boxes.tensor)
            if a_iou.numel() == 0:
                pos_iou = a_iou.new_full((0,), 0)
            else:
                pos_iou = a_iou[src_idx, tgt_idx]
            ious.append(max_iou)
            pos_ious.append(pos_iou)
        ious = torch.cat(ious)
        ignore_idx = ious > self.neg_ignore_thresh
        pos_ious = torch.cat(pos_ious)
        pos_ignore_idx = pos_ious < self.pos_ignore_thresh

        src_idx = torch.cat(
            [src + idx * anchors[0].tensor.shape[0] for idx, (src, _) in
             enumerate(indices)])
        gt_classes = torch.full(pred_class_logits.shape[:1],
                                self.num_classes,
                                dtype=torch.int64,
                                device=pred_class_logits.device)
        gt_classes[ignore_idx] = -1
        target_classes_o = torch.cat(
            [t.gt_classes[J] for t, (_, J) in zip(gt_instances, indices)])
        target_classes_o[pos_ignore_idx] = -1
        gt_classes[src_idx] = target_classes_o

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        #dist.all_reduce(num_foreground)
        num_foreground = num_foreground * 1.0 # / dist.get_world_size()

        # cls loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )
        # reg loss
        target_boxes = torch.cat(
            [t.gt_boxes.tensor[i] for t, (_, i) in zip(gt_instances, indices)],
            dim=0)
        target_boxes = target_boxes[~pos_ignore_idx]
        matched_predicted_boxes = predicted_boxes.reshape(-1, 4)[
            src_idx[~pos_ignore_idx]]

        # ----------------- GIOU -----------------
        #loss_box_reg = (1 - torch.diag(generalized_box_iou(
            #matched_predicted_boxes, target_boxes))).sum()
        
        # ----------------- L1 ------------------
        # Temporarily change to smooth_l1_loss
        #loss_box_reg = smooth_l1_loss(matched_predicted_boxes,
                                      #target_boxes,
                                      #beta=0.0,
                                      #reduction='mean')
        
        # ----------------- DIOU ------------------
        loss_box_reg = diou_loss(matched_predicted_boxes,
                                 target_boxes,
                                 reduction='mean')
        

        return {
            "loss_cls": loss_cls / max(1, num_foreground),
            "loss_box_reg": loss_box_reg / max(1, num_foreground)
        }
    
    @torch.no_grad()
    def get_ground_truth(self, anchors, bbox_preds, targets):
        # TODO: redocument shape
        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
        N = len(anchors)
        # list[Tensor(R, 4)], one for each image
        all_anchors = Boxes.cat(anchors).tensor.reshape(N, -1, 4)
        # Boxes(Tensor(N*R, 4))
        box_delta = cat(bbox_preds, dim=1)
        # box_pred: xyxy; targets: xyxy
        box_pred = self.box2box_transform.apply_deltas(box_delta, all_anchors)
        indices = self.anchor_matcher(box_pred, all_anchors, targets)

        return indices
    
    def inference(self, box_cls, box_delta, anchors, image_sizes):
        """
        Arguments:
            box_cls, box_delta: Same as the output of :meth:`YOLOFHead.forward`
            anchors (list[list[Boxes]]): a list of #images elements. Each is a
                list of #feature level Boxes. The Boxes contain anchors of this
                image on the specific feature level.
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(anchors) == len(image_sizes)
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)

        for img_idx, anchors_per_image in enumerate(anchors):
            image_size = image_sizes[img_idx]
            box_cls_per_image = [
                box_cls_per_level[img_idx] for box_cls_per_level in box_cls
            ]
            box_reg_per_image = [
                box_reg_per_level[img_idx] for box_reg_per_level in box_delta
            ]
            results_per_image = self.inference_single_image(
                box_cls_per_image, box_reg_per_image, anchors_per_image,
                tuple(image_size))
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, anchors, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta,
                                                   anchors):
            # (HxWxAxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates_test, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(
                box_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]

        keep = batched_nms(boxes_all, scores_all, class_idxs_all,
                            self.nms_threshold)
        keep = keep[:self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result