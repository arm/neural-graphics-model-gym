# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
from functools import partial
from typing import List

import torch
from torch import nn

from ng_model_gym.nss.model.dense_warp_utils import (
    backward_warp_nearest,
    bilinear_oob_zero,
    catmull_rom_warp,
    dense_image_warp,
)


class DenseWarp(nn.Module):
    """Image warping with dense optical flow in PyTorch."""

    def __init__(self, interpolation="bilinear"):
        """
        Args:
            interpolation (str): The interpolation method to use. Supported methods include:
            - 'bilinear': Bilinear interpolation
            - 'bilinear_oob_zero': Bilinear interpolation with out-of-bounds set to zero
            - 'catmull': Catmull-Rom interpolation
            - 'nearest': Nearest neighbour interpolation
            - 'nearest_oob_zero': Nearest neighbour  interpolation with out-of-bounds set to zero
        """
        super().__init__()

        _supported_methods = {
            "bilinear": dense_image_warp,
            "bilinear_oob_zero": bilinear_oob_zero,
            "catmull": catmull_rom_warp,
            "nearest": backward_warp_nearest,
            "nearest_oob_zero": partial(backward_warp_nearest, oob_zero=True),
        }

        if interpolation not in _supported_methods:
            raise ValueError(f"{interpolation} method not in {_supported_methods}")

        self.interpolation = interpolation
        self.interp_func = _supported_methods[interpolation]

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for DenseWarp layer in PyTorch.

        Args:
            inputs (List[torch.Tensor]): A list containing two tensors:
            - frame: The frame to be warped (tensor of shape [batch_size, channels, height, width])
            - flow_vectors: Flow vectors (tensor of shape [batch_size, 2, height, width])

        Returns:
            torch.Tensor: The warped frame
        """
        frame_dt = inputs[0]
        flow_vec = inputs[1]

        return self.interp_func(frame_dt, flow_vec)
