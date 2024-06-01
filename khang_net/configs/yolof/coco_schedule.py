from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

def default_X_scheduler(num_X):
    """
    Returns the config for a default multi-step LR scheduler such as "1x", "3x",
    commonly referred to in papers, where every 1x has the total length of 1440k
    training images (~12 COCO epochs). LR is decayed twice at the end of training
    following the strategy defined in "Rethinking ImageNet Pretraining", Sec 4.

    Args:
        num_X: a positive real number

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # total number of iterations assuming 16 batch size, using 1440000/16=90000
    #total_steps_16bs = num_X * 90000
    total_steps_10bs = num_X * 90000 * 16 / 10

    warmup_iters = 1500

    if num_X <= 2:
        scheduler = L(MultiStepParamScheduler)(
            values=[1.0, 0.1, 0.01],
            milestones=[60000, 80000, total_steps_10bs],
        )
    else:
        scheduler = L(MultiStepParamScheduler)(
            values=[1.0, 0.1, 0.01],
            milestones=[total_steps_10bs - 60000, total_steps_10bs - 20000, total_steps_10bs],
        )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_iters / total_steps_10bs,
        warmup_method="linear",
        warmup_factor=0.00066667
    )

lr_multiplier_1x = default_X_scheduler(1)
lr_multiplier_2x = default_X_scheduler(2)
lr_multiplier_3x = default_X_scheduler(3)
lr_multiplier_6x = default_X_scheduler(6)
lr_multiplier_9x = default_X_scheduler(9)