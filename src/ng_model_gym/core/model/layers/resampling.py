# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import math

import torch
from torch import nn
from torch.nn import functional as F


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


class ZeroUpsample(nn.Module):
    """Custom zero/jitter encoded upsample in PyTorch for non-integer scale factors"""

    def __init__(self, scale=(2.0, 2.0)):
        """
        Args:
            scale (Tuple[float]): (H, W) scale values. Default is (2.0, 2.0).
        """
        super().__init__()
        self.scale = scale

    def forward(self, inputs):
        """
        Args:
            inputs (Tuple[torch.Tensor, torch.Tensor]): (tensor to zero upsample, jitter)

        Returns:
            torch.Tensor: zero upsampled input tensor.
        """
        # Unpack expected input data
        ten_in, jitter = inputs

        # Zero Upsample
        zero_upsampled = zero_upsample(ten_in, jitter, self.scale)

        return zero_upsampled


def zero_upsample(
    ten_in: torch.Tensor, jitter: torch.Tensor, scale=(2.0, 2.0)
) -> torch.Tensor:
    """Implementation of Zero upsampling that works for non-integer scale factors.

    Args:
        ten_in: Tensor to zero upsample
        jitter: Jitter offset, in texels @ input resolution, (y, x), display-space (i.e., +y = down)
        scale: Upsample scale factor in: (height, width)

    Returns:
        torch.Tensor: Zero upsampled input tensor.
    """

    def rank4(_ten_in: torch.Tensor) -> torch.Tensor:
        return _ten_in.unsqueeze(0).unsqueeze(-1)

    # Calculate Shapes
    batch_size, channels, in_height, in_width = (
        ten_in.size(0),
        ten_in.size(1),
        ten_in.size(2),
        ten_in.size(3),
    )
    scale_x = torch.tensor(scale[1], device=ten_in.device)
    scale_y = torch.tensor(scale[0], device=ten_in.device)
    out_height, out_width = int(float(in_height) * scale_y), int(
        float(in_width) * scale_x
    )

    # Convert Jitter to Grid
    jit_y, jit_x = torch.split(jitter, split_size_or_sections=1, dim=1)

    # Create a sampling grid in output shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, in_height, device=ten_in.device),
        torch.arange(0, in_width, device=ten_in.device),
        indexing="ij",
    )

    # Indices for each dimension
    batch_idx = torch.tile(
        torch.reshape(
            torch.arange(0, batch_size, device=ten_in.device), (batch_size, 1, 1, 1)
        ),
        (1, in_height, in_width, channels),
    )
    height_idx = torch.clamp(
        torch.floor(
            (torch.tile(rank4(grid_y), (batch_size, 1, 1, channels)) + 0.5 + jit_y)
            * scale_y
        ).int(),
        min=0,
        max=out_height - 1,
    )
    width_idx = torch.clamp(
        torch.floor(
            (torch.tile(rank4(grid_x), (batch_size, 1, 1, channels)) + 0.5 + jit_x)
            * scale_x
        ).int(),
        min=0,
        max=out_width - 1,
    )

    # Combined Indices
    indices = torch.reshape(
        batch_idx * out_height * out_width + height_idx * out_width + width_idx,
        (-1, channels),
    )

    # Scatter Samples to correct locations, permute to NHWC then permute back
    src = torch.reshape(ten_in.permute(0, 2, 3, 1), (-1, channels))
    dst = torch.zeros(
        (batch_size * out_height * out_width, channels), device=src.device
    )
    scattered = dst.scatter(dim=0, index=indices, src=src)
    zero_upsampled = torch.reshape(
        scattered, (batch_size, out_height, out_width, channels)
    ).permute(0, 3, 1, 2)

    return zero_upsampled
