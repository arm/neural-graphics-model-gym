# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import Optional, Tuple, Union

import torch
from torch import nn

# NOTE: This intentionally diverges from core resampling:
# - Uses `scale_factor` instead of explicit output-size rounding.
# - Normalizes `nearest` to `nearest-exact` to avoid legacy nearest artifacts.


class UpSampling2D(nn.Module):
    """
    Module wrapper around [`F.interpolate`]
    (https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html)
    """

    def __init__(
        self,
        *args,
        size: Optional[Union[Tuple[float], float]] = 2.0,
        interpolation: Optional[str] = "nearest",
        **kwargs,
    ):
        """

        Args:
            size: Int, or tuple of 2 integers.
                  The upsampling factors for rows and columns.
            interpolation: A string, one of:
                - 'nearest'
                - 'linear'
                - 'bilinear'
                - 'bicubic'
                - 'trilinear'
                - 'area'
                - 'nearest-exact'
        """
        super().__init__(*args, **kwargs)
        self.size = (size, size) if not isinstance(size, (list, tuple)) else size
        self.interpolation = str(interpolation).lower()
        # Ensure we never accidentally use the "buggy" `nearest` implementation
        self.interpolation = (
            "nearest-exact" if self.interpolation == "nearest" else self.interpolation
        )
        self.method = partial(
            torch.nn.functional.interpolate,
            scale_factor=self.size,
            mode=self.interpolation,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """UpSampling2D

        Args:
            inputs (torch.Tensor): Tensor to be upsampled

        Returns:
            torch.Tensor: Upsampled tensor
        """
        return self.method(inputs)
