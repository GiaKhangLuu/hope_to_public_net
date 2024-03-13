from typing import Any, Dict, List, Set

import torch
from torch import optim

from detectron2.config import LazyCall as L

NORM_MODULE_TYPES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    # NaiveSyncBatchNorm inherits from BatchNorm2d
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LayerNorm,
    torch.nn.LocalResponseNorm,
)

def get_yolof_default_optimizer_params(
    model: torch.nn.Module, 
    base_lr=0.12, 
    bias_lr_factor=1.0, 
    weight_decay=0.0001, 
    weight_decay_norm=0.0, 
    momentum=0.9, 
    backbone_lr_factor=0.334
) -> List[Dict[str, Any]]:
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for name, module in model.named_modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = base_lr
            weight_decay = weight_decay
            if name.startswith("backbone"):
                lr = lr * backbone_lr_factor
            if isinstance(module, NORM_MODULE_TYPES):
                weight_decay = weight_decay_norm
            elif key == "bias":
                lr = base_lr * bias_lr_factor 
                weight_decay = weight_decay
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    return params

YOLOF_SGD = L(torch.optim.SGD)(
    params=L(get_yolof_default_optimizer_params)(
        base_lr=0.12,
        bias_lr_factor=1.0,
        weight_decay=0.0001,
        weight_decay_norm=0.0,
        momentum=0.9,
        backbone_lr_factor=0.334
    ),
    lr=0.12,
    momentum=0.9,
    weight_decay=1e-4
)