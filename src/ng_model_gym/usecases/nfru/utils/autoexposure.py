# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import torch

from ng_model_gym.usecases.nfru.utils.constants import _REC709_LUMA_WEIGHTS

_AUTO_EXPOSURE_MIN_LUMINANCE = 1e-6


def rgb_2_luminance(x: torch.Tensor) -> torch.Tensor:
    """Convert NCHW RGB tensors to single-channel luminance."""
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("Expected NCHW RGB tensor with 3 channels for auto exposure.")

    coefficients = torch.tensor(
        _REC709_LUMA_WEIGHTS, device=x.device, dtype=x.dtype
    ).view(1, 3, 1, 1)
    return torch.sum(x * coefficients, dim=1, keepdim=True)


def KeyValueAE(
    colour_frame_linear: torch.Tensor, key_value: float = 1.0
) -> torch.Tensor:
    """Compute key-value auto exposure for NCHW torch tensors."""
    luminance = rgb_2_luminance(colour_frame_linear)
    mean_luminance = torch.mean(luminance, dim=(1, 2, 3), keepdim=True)
    mean_luminance = torch.clamp(mean_luminance, min=_AUTO_EXPOSURE_MIN_LUMINANCE)
    exposure = key_value / mean_luminance
    return torch.log(exposure)
