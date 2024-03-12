import logging
import math
from typing import List, Tuple, Optional, Dict, Union
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import Tensor, nn
from torch.nn import functional as F

from detectron2.layers import CycleBatchNormList, ShapeSpec, batched_nms, cat, get_norm, move_device_like
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.box_regression import Box2BoxTransform 
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.meta_arch.dense_detector import permute_to_N_HWA_K  # noqa

#from .yolof_encoder import DilatedEncoder

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
        test_score_thresh=0.05,
        test_topk_candidates=1000,
        test_nms_thresh=0.5,
        max_detections_per_image=100,
        pixel_mean,
        pixel_std,
        vis_period=0,
        input_format="BGR",
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

        # Loss parameters:
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type
        # Inference parameters:
        self.test_score_thresh = test_score_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_thresh = test_nms_thresh
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
        print(features)
        print(features.shape)
        predictions = self.decoder(self.encoder(features))

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            return self.forward_training(images, features, predictions, gt_instances)
        else:
            # TODO: From now, haven't modified training_inference yet
            return
            #results = self.forward_inference(images, features, predictions)
            #if torch.jit.is_scripting():
                #return results

            #processed_results = []
            #for results_per_image, input_per_image, image_size in zip(
                #results, batched_inputs, images.image_sizes
            #):
                #height = input_per_image.get("height", image_size[0])
                #width = input_per_image.get("width", image_size[1])
                #r = detector_postprocess(results_per_image, height, width)
                #processed_results.append({"instances": r})
            #return processed_results
        
    def forward_training(self, images, features, predictions, gt_instances):
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(
            predictions, [self.num_classes, 4]
        )

        # anchors.shape = (H x W x num_cell_anchors, 4)
        # 4 is (XYXY)
        anchors = self.anchor_generator(features)
        gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
        return self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes)
    
    def _transpose_dense_predictions(
        self, predictions: List[Tensor], dims_per_anchor: List[int]
    ) -> List[Tensor]:
        """
        Transpose the dense per-level predictions.

        Args:
            predictions: a list of outputs, each is predictions with 
                shape (N, A x K, H, W), where N is the batch size,
                A is the number of anchors per location on
                feature map (grid), K is the dimension of predictions per anchor.
            dims_per_anchor: the value of K for each predictions. e.g. 4 for
                box prediction, #classes for classification prediction.

        Returns:
            List[Tensor]: each prediction is transposed to (N, H x W x A, K).
        """
        assert len(predictions) == len(dims_per_anchor)
        res: List[Tensor] = []
        for pred, dim_per_anchor in zip(predictions, dims_per_anchor):
            res.append(permute_to_N_HWA_K(pred, dim_per_anchor))
        return res
    
    @torch.no_grad()
    def label_anchors(self, anchors, gt_instances):
        """
        Args:
            anchors (Boxes): The Boxes contains all anchors of this image 
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.

        Returns:
            list[Tensor]: List of #img tensors. i-th element is a vector of labels whose length is
            the total number of anchors across all feature maps (sum(Hi * Wi * A)).
            Label values are in {-1, 0, ..., K}, with -1 means ignore, and K means background.

            list[Tensor]: i-th element is a Rx4 tensor, where R is the total number of anchors
            across feature maps. The values are the matched gt boxes for each anchor.
            Values are undefined for those anchors not labeled as foreground.
        """

        gt_labels = []
        matched_gt_boxes = []
        for gt_per_image in gt_instances:
            # match_quality_matrix.shape = (num_gt_instances, R), R is the total #anchors
            match_quality_matrix = pairwise_iou(gt_per_image.gt_boxes, anchors)  
            matched_idxs, anchor_labels = self.anchor_matcher(match_quality_matrix)
            del match_quality_matrix

            if len(gt_per_image) > 0:
                matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]

                gt_labels_i = gt_per_image.gt_classes[matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_labels_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_labels_i[anchor_labels == -1] = -1
            else:
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes
    
    def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes):
        """
        Args:
            anchors (Boxes): The Boxes contains all anchors of this image 
            gt_labels, gt_boxes: see output of :meth:`RetinaNet.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors, R = H x W x A
            pred_logits, pred_anchor_deltas: both are Tensor. Predictions from
                the model has shape (N, Hi * Wi * Ai, K or 4).
                Where K is the number of classes used in `pred_logits`.

        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor storing the loss.
                Used during training only. The dict keys are: "loss_cls" and "loss_box_reg"
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, R)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        normalizer = self._ema_update("loss_normalizer", max(num_pos_anchors, 1), 100)

        # classification and regression loss
        gt_labels_target = F.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[
            :, :-1
        ]  # no loss for the last (background) class
        loss_cls = sigmoid_focal_loss_jit(
            pred_logits[valid_mask],
            gt_labels_target.to(pred_logits.dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )

        loss_box_reg = self._dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        return {
            "loss_cls": loss_cls / normalizer,
            "loss_box_reg": loss_box_reg / normalizer,
        }
    
    def _ema_update(self, name: str, value: float, initial_value: float, momentum: float = 0.9):
        """
        Apply EMA update to `self.name` using `value`.

        This is mainly used for loss normalizer. In Detectron1, loss is normalized by number
        of foreground samples in the batch. When batch size is 1 per GPU, #foreground has a
        large variance and using it lead to lower performance. Therefore we maintain an EMA of
        #foreground to stabilize the normalizer.

        Args:
            name: name of the normalizer
            value: the new value to update
            initial_value: the initial value to start with
            momentum: momentum of EMA

        Returns:
            float: the updated EMA value
        """
        if hasattr(self, name):
            old = getattr(self, name)
        else:
            old = initial_value
        new = old * momentum + value * (1 - momentum)
        setattr(self, name, new)
        return new
    
    def _dense_box_regression_loss(
        anchors:Union[Boxes, torch.Tensor],
        box2box_transform: Box2BoxTransform,
        pred_anchor_deltas: torch.Tensor,
        gt_boxes: List[torch.Tensor],
        fg_mask: torch.Tensor,
        box_reg_loss_type="smooth_l1",
        smooth_l1_beta=0.0,
    ):
        """
        Compute loss for dense multi-level box regression.
        Loss is accumulated over ``fg_mask``.

        Args:
            anchors: anchor boxes, shape = (HxWxA, 4)
            pred_anchor_deltas: box regression predictions, shape = (N, HxWxA, 4)
            gt_boxes: N ground truth boxes, each has shape (R, 4) (R = H * W * A))
            fg_mask: the foreground boolean mask of shape (N, R) to compute loss on
            box_reg_loss_type (str): Loss type to use. Supported losses: "smooth_l1", "giou",
                "diou", "ciou".
            smooth_l1_beta (float): beta parameter for the smooth L1 regression loss. Default to
                use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"
        """
        if box_reg_loss_type == "smooth_l1":
            gt_anchor_deltas = [box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
            gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)
            loss_box_reg = smooth_l1_loss(
                cat(pred_anchor_deltas, dim=1)[fg_mask],
                gt_anchor_deltas[fg_mask],
                beta=smooth_l1_beta,
                reduction="sum",
            )
        else:
            raise ValueError(f"Invalid dense box regression loss type '{box_reg_loss_type}'")
        return loss_box_reg
