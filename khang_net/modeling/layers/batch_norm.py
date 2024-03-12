from functools import partial

import torch
import torch.nn as nn

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None
    
if TORCH_VERSION > (1, 4):
    BatchNorm2d = torch.nn.BatchNorm2d
else:
    class BatchNorm2d(torch.nn.BatchNorm2d):
        """
        A wrapper around :class:`torch.nn.BatchNorm2d` to support zero-size tensor.
        """
        def forward(self, x):
            if x.numel() > 0:
                return super(BatchNorm2d, self).forward(x)
            # get output shape
            output_shape = x.shape
            return _NewEmptyTensorOp.apply(x, output_shape)

def get_norm(norm, out_channels, **kwargs):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
        kwargs: Additional parameters in normalization layers,
            such as, eps, momentum

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm2d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            #"SyncBN": NaiveSyncBatchNorm if env.TORCH_VERSION <= (
                #1, 5) else nn.SyncBatchNorm,
            #"FrozenBN": FrozenBatchNorm2d,
            #"GN": lambda channels: nn.GroupNorm(32, channels),
            ## for debugging:
            #"nnSyncBN": nn.SyncBatchNorm,
            #"naiveSyncBN": NaiveSyncBatchNorm,
        }[norm]
    return norm(out_channels, **kwargs)
