# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

"""Colour grading operators used by the NFRU v1 training recipe."""

from __future__ import annotations

from typing import Any, Callable, Dict

import torch

from ng_model_gym.usecases.nfru.utils.constants import (
    _REC709_LUMA_WEIGHTS,
    _TEMPERATURE_TINT_SCALE,
)


def grade_none(tonemapped: torch.Tensor, **_: Any) -> torch.Tensor:
    """Return the image unchanged."""
    return tonemapped


def grade_contrast(
    tonemapped: torch.Tensor, strength: float = 1.2, **_: Any
) -> torch.Tensor:
    """Adjust contrast around the display-space midpoint."""
    adjusted = (tonemapped - 0.5) * strength + 0.5
    return torch.clamp(adjusted, 0.0, 1.0)


def grade_saturation(
    tonemapped: torch.Tensor, strength: float = 1.2, **_: Any
) -> torch.Tensor:
    """Scale distance from luminance to adjust saturation."""
    weights = torch.tensor(
        _REC709_LUMA_WEIGHTS, device=tonemapped.device, dtype=tonemapped.dtype
    ).view(1, 3, 1, 1)
    luminance = torch.sum(tonemapped * weights, dim=1, keepdim=True)
    adjusted = luminance + (tonemapped - luminance) * strength
    return torch.clamp(adjusted, 0.0, 1.0)


def grade_temperature_tint(
    tonemapped: torch.Tensor,
    temperature: float = 0.0,
    tint: float = 0.0,
    **_: Any,
) -> torch.Tensor:
    """Apply a simple white-balance style temperature/tint shift."""
    multiplier = torch.tensor(
        [
            1.0 + temperature * _TEMPERATURE_TINT_SCALE,
            1.0 - tint * _TEMPERATURE_TINT_SCALE,
            1.0 - temperature * _TEMPERATURE_TINT_SCALE,
        ],
        device=tonemapped.device,
        dtype=tonemapped.dtype,
    ).view(1, 3, 1, 1)
    adjusted = tonemapped * multiplier
    return torch.clamp(adjusted, 0.0, 1.0)


_COLOUR_GRADING_REGISTRY: Dict[str, Callable] = {
    "none": grade_none,
    "contrast": grade_contrast,
    "saturation": grade_saturation,
    "temperature_tint": grade_temperature_tint,
}


def get_colour_grading_op(name: str) -> Callable:
    """Retrieve a registered colour grading operator."""
    key = name.lower()
    if key not in _COLOUR_GRADING_REGISTRY:
        raise KeyError(
            f"Unknown colour grading operator '{name}'. "
            f"Available: {sorted(_COLOUR_GRADING_REGISTRY)}"
        )
    return _COLOUR_GRADING_REGISTRY[key]
