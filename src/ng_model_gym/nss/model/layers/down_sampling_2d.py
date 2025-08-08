# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F
from torch import nn


class DownSampling2D(nn.Module):
    """
    A PyTorch layer wrapper around torch.nn.functional.interpolate for 2D downsampling.
    """

    def __init__(self, size=(2, 2), interpolation="nearest", antialias=False):
        """
        Args:
            size: Int, or tuple of 2 integers.
                  The downsampling factors for rows and columns.
            interpolation: A string, one of `"bicubic"`, `"bilinear"`,
                           `"nearest"`.
            antialias (bool, optional): Whether to use an anti-aliasing filter when
                                        downsampling an image. Defaults to False.
        """
        super().__init__()

        self.size = (size, size) if not isinstance(size, (list, tuple)) else size

        _supported_methods = {
            "nearest",
            "bilinear",
            "bicubic",
        }

        if interpolation not in _supported_methods:
            raise ValueError(f"{interpolation} method not in {_supported_methods}")

        self.interpolation = interpolation

        # When downsampling an image with anti-aliasing the sampling filter kernel is
        # scaled in order to properly anti-alias the input image signal.
        self.antialias = antialias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        DownSampling2D

        Args:
            inputs (torch.Tensor): Tensor to be downsampled

        Returns:
            torch.Tensor: Downsampled tensor
        """
        _, _, h, w = inputs.shape

        # We need torch.tensor for torch.ceil, so we wrap our int
        lr_height = torch.ceil(
            torch.tensor(h / self.size[0], dtype=torch.float32)
        ).int()

        lr_width = torch.ceil(torch.tensor(w / self.size[1], dtype=torch.float32)).int()

        lr_shape = (lr_height, lr_width)

        # Perform the downsampling
        # See doc: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        # The option antialias is valid only for "bilinear" or "bicubic" interpolation.
        if self.interpolation != "nearest":
            return F.interpolate(
                inputs, size=lr_shape, mode=self.interpolation, antialias=self.antialias
            )
        return F.interpolate(inputs, size=lr_shape, mode=self.interpolation)
