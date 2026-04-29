# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
from functools import partial
from typing import Optional, Tuple, Union

import torch
from torch import nn

# NOTE: This intentionally diverges from core resampling:
# - Downsampling is expressed as inverse `scale_factor`.
# - Normalizes `nearest` to `nearest-exact` to avoid legacy nearest artifacts.


class DownSampling2D(nn.Module):
    """A module wrapper around [`F.interpolate`]
    (https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html)"""

    def __init__(
        self,
        *args,
        size: Optional[Union[Tuple[float], float]] = 2.0,
        interpolation: Optional[str] = "nearest",
        antialias: Optional[bool] = False,
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
            antialias: Use an anti-aliasing filter when downsampling an image. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        self.size = [size, size] if not isinstance(size, (list, tuple)) else size
        self.size = [1.0 / s for s in self.size]
        self.interpolation = str(interpolation).lower()
        # Ensure we never accidentally use the "buggy" `nearest` implementation
        self.interpolation = (
            "nearest-exact" if self.interpolation == "nearest" else self.interpolation
        )
        # When downsampling an image with anti-aliasing the sampling filter kernel is
        # scaled in order to properly anti-alias the input image signal.
        self.antialias = antialias
        self.method = partial(
            torch.nn.functional.interpolate,
            scale_factor=self.size,
            mode=self.interpolation,
            antialias=self.antialias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """DownSampling2D

        Args:
            inputs (torch.Tensor): Tensor to be downsampled

        Returns:
            torch.Tensor: Downsampled tensor
        """
        return self.method(inputs)
