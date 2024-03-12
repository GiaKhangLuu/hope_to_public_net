# Copyright (c) Facebook, Inc. and its affiliates.
import collections
import math
from typing import List
import torch
from torch import nn

from detectron2.layers import ShapeSpec, move_device_like
from detectron2.structures import Boxes

class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers):
        super().__init__()
        for i, buffer in enumerate(buffers):
            # Use non-persistent buffer so the values are not saved in checkpoint
            self.register_buffer(str(i), buffer, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


def _create_grid_offsets(
    size: List[int], stride: int, offset: float, target_device_tensor: torch.Tensor
):
    """
    Args:
        sizes (List[int]): H x W of feature map (grid) 
        stride (int): total stride 
            grid_height, grid_width = size
            stride * grid_height = img_height
            stride * grid_width = img_width
        offset (float): position of center point of anchor
    """
    grid_height, grid_width = size
    shifts_x = move_device_like(
        torch.arange(offset * stride, grid_width * stride, step=stride, dtype=torch.float32),
        target_device_tensor,
    )  # len(shifts_x) = grid_width
    shifts_y = move_device_like(
        torch.arange(offset * stride, grid_height * stride, step=stride, dtype=torch.float32),
        target_device_tensor,
    )  # len(shifts_y) = grid_height

    assert len(shifts_x) == grid_width
    assert len(shifts_y) == grid_height

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)  # shift_x.shape = shift_y.shape = (grid_height, grid_width)
    shift_x = shift_x.reshape(-1)  # shift_x.shape = (grid_height * grid_width)
    shift_y = shift_y.reshape(-1)  # shift_y.shape = (grid_height * grid_width)

    return shift_x, shift_y

class YOLOFAnchorGenerator(nn.Module):
    """
    Compute anchors described in YOLOF
    In this style, each location has total 5 anchor boxes with 
    5 different sizes and only 1 scale on only one feature map
    """

    box_dim: torch.jit.Final[int] = 4
    """
    the dimension of each anchor box.
    """

    def __init__(self, *, sizes, aspect_ratio, stride, offset=0.5):
        """
        This interface is experimental.

        Args:
            TODO: Write document
        """
        super().__init__()

        self.stride = stride
        self.base_anchors = self._calculate_anchors(sizes, aspect_ratio)  

        self.offset = offset
        assert 0.0 <= self.offset < 1.0, self.offset

    def _calculate_anchors(self, sizes, aspect_ratio):
        return self.generate_cell_anchors(sizes, aspect_ratio)
    
    def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratio=1.0):
        """
        Generate a tensor storing canonical anchor boxes, which are all anchor
        boxes of different sizes and aspect_ratios centered at (0, 0).
        We can later build the set of anchors for a full feature map by
        shifting and tiling these tensors (see `meth:_grid_anchors`).

        Args:
            sizes (tuple[float]):
            aspect_ratios int:

        Returns:
            Tensor of shape (len(sizes), 4) storing anchor boxes
                in XYXY format.
        """

        anchors = []
        for size in sizes:
            area = size**2.0
            w = math.sqrt(area / aspect_ratio)
            h = aspect_ratio * w
            x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
            anchors.append([x0, y0, x1, y1])
        return torch.tensor(anchors)

    def _grid_anchors(self, grid_size: List[int]):
        """
        Returns:
            Tensor: shape (#locations x #cell_anchors) x 4
        """
        anchors = []
        shift_x, shift_y = _create_grid_offsets(grid_size, self.stride, self.offset, self.base_anchors)  # len(shift_x) = len(shift_y) = H x W
        assert len(shift_x) == len(shift_y) == grid_size[0] * grid_size[1]

        # shifts.shape = (Hi x Wi, 4), 4 is shifted value for x
        # 4 is shifted value for (XYXY)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)  
        assert tuple(shifts.shape) ==  (grid_size[0] * grid_size[1], 4)

        anchors = (shifts.view(-1, 1, 4) + self.base_anchors.view(1, -1, 4)).reshape(-1, 4)  # (R, 4), R is total #anchors

        return anchors


    def forward(self, features: torch.Tensor):
        """
        Args:
            features (Tensor): backbone feature map on which to generate anchors.

        Returns:
            Boxes: a Boxes containing all the anchors for a feature map
                (i.e. the cell anchors repeated over all locations in the feature map).
                The number of anchors of feature map is H x W x num_cell_anchors,
                where H, W are resolution of the feature map divided by anchor stride.
        """
        grid_size = features.shape[-2:]

        # anchors_over_grid.shape = (H x W x #cell_anchors, 4) 
        anchors_over_grid = self._grid_anchors(grid_size)  

        return Boxes(anchors_over_grid)
