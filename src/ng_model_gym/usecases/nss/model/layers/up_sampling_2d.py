# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import math

import torch
import torch.nn.functional as F
from torch import nn


class UpSampling2D(nn.Module):
    """
    A PyTorch layer wrapper around
    torch.nn.functional.interpolate for 2D upsampling.
    """

    def __init__(self, size=(2, 2), interpolation="nearest"):
        """
        Args:
            size: Int, or tuple of 2 integers.
                  The upsampling factors for rows and columns.
            interpolation: A string, one of `"bicubic"`, `"bilinear"`,
                    `"nearest"`.
        """
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

        _upsampling_supported_methods = {
            "nearest",
            "bilinear",
            "bicubic",
        }

        if interpolation not in _upsampling_supported_methods:
            raise ValueError(
                f"{interpolation} method not in {_upsampling_supported_methods}"
            )
        self.interpolation = interpolation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        UpSampling2D

        Args:
            inputs (torch.Tensor): Tensor to be upsampled

        Returns:
            torch.Tensor: Upsampled tensor
        """
        _, _, h, w = inputs.shape

        # We need to wrap into torch.tensor for ceil function.
        hr_height = math.ceil(h * self.size[0])
        hr_width = math.ceil(w * self.size[1])

        hr_shape = (hr_height, hr_width)

        # There is an option align_corners for bilinear/bicubic
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        # It is False by default.
        upsampled = F.interpolate(inputs, size=hr_shape, mode=self.interpolation)

        return upsampled
