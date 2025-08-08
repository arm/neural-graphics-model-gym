# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn


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
