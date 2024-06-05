from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

def default_X_scheduler(num_X, batch_size_16=True):
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
    total_steps_16bs = num_X * 90000
    total_steps_10bs = num_X * 90000 * 16 / 10

    if batch_size_16:
        total_steps_based_on_batch_size = total_steps_16bs
    else:
        total_steps_based_on_batch_size = total_steps_10bs

    warmup_iters = 1500 if batch_size_16 else (1500 * 16 / 10)

    if num_X <= 2:

        if batch_size_16:
            milestones=[60000, 80000, total_steps_based_on_batch_size]
        else:
            milestones=[60000 * 16 / 10, 80000 * 16 / 10, total_steps_based_on_batch_size]

        scheduler = L(MultiStepParamScheduler)(
            values=[1.0, 0.1, 0.01],
            milestones=milestones
        )
    else:
        
        if batch_size_16:
            milestones=[total_steps_based_on_batch_size - 60000, 
                        total_steps_based_on_batch_size - 20000, 
                        total_steps_based_on_batch_size]
        else:
            milestones=[total_steps_based_on_batch_size - (60000 * 16 / 10),
                        total_steps_based_on_batch_size - (20000 * 16 / 10),
                        total_steps_based_on_batch_size
            ]

        scheduler = L(MultiStepParamScheduler)(
            values=[1.0, 0.1, 0.01],
            milestones=milestones
        )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_iters / total_steps_based_on_batch_size,
        warmup_method="linear",
        warmup_factor=0.00066667
    )

lr_multiplier_1x_b16 = default_X_scheduler(1)
lr_multiplier_2x_b16 = default_X_scheduler(2)
lr_multiplier_3x_b16 = default_X_scheduler(3)
lr_multiplier_6x_b16 = default_X_scheduler(6)
lr_multiplier_9x_b16 = default_X_scheduler(9)

lr_multiplier_1x_b10 = default_X_scheduler(1, batch_size_16=False)
lr_multiplier_2x_b10 = default_X_scheduler(2, batch_size_16=False)
lr_multiplier_3x_b10 = default_X_scheduler(3, batch_size_16=False)
lr_multiplier_6x_b10 = default_X_scheduler(6, batch_size_16=False)
lr_multiplier_9x_b10 = default_X_scheduler(9, batch_size_16=False)