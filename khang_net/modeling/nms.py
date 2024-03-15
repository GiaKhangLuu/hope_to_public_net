# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms  # BC-compat

import functools

try:
    from apex import amp
except ImportError:
    pass

#from cvpods.utils.apex_wrapper import float_function

def is_amp_training():
    """
    check weather amp training is enabled.

    Returns:
        bool: True if amp training is enabled
    """
    try:
        return hasattr(amp._amp_state, "loss_scalers")
    except Exception:
        return False

def float_function(func):

    @functools.wraps(func)
    def float_wraps(*args, **kwargs):
        if is_amp_training():
            return amp.float_function(func)(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return float_wraps


@float_function
def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    assert boxes.shape[-1] == 4
    # TODO may need better strategy.
    # Investigate after having a fully-cuda NMS op.
    if len(boxes) < 40000:
        return box_ops.batched_nms(boxes, scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero(as_tuple=False).view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero(as_tuple=False).view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep