# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

"""Tonemapping operators registry and implementations."""

from __future__ import annotations

from typing import Any, Callable, Dict, Union

import numpy as np
import torch

from ng_model_gym.core.data import tonemap_forward as torch_tonemap_forward
from ng_model_gym.core.utils.enum_definitions import ToneMapperMode


def tonemap_forward(
    x: Union[torch.Tensor, np.ndarray], max_val: float = 1.0, mode: str = "reinhard"
) -> Union[torch.Tensor, np.ndarray]:
    """
    Reinhard:
    http://behindthepixels.io/assets/files/TemporalAA.pdf#page5
    """
    if isinstance(x, torch.Tensor):
        if isinstance(mode, ToneMapperMode):
            tone_mode = mode
        else:
            tone_mode = ToneMapperMode(str(mode).lower())
        return torch_tonemap_forward(x, max_val=max_val, mode=tone_mode)

    x = np.nan_to_num(x)

    if mode == "reinhard":
        x = x * (max_val / (1.0 + x))
    else:
        raise ValueError(f"Tonemap: {mode} unsupported")

    return np.clip(x, 0.0, max_val)


def tonemap_none(linear_rgb: torch.Tensor, **_: Any) -> torch.Tensor:
    """Placeholder tonemapping algorithm which does nothing"""
    return linear_rgb


def tonemap_reinhard(linear_rgb: torch.Tensor, **_: Any) -> torch.Tensor:
    """Perform tonemapping using the Reinhard algorithm"""
    return tonemap_forward(linear_rgb, mode="reinhard")


# Registry

_TONEMAP_REGISTRY: Dict[str, Callable] = {
    "none": tonemap_none,
    "reinhard": tonemap_reinhard,
}


def get_tonemapper(name: str) -> Callable:
    """Retrieve a registered tonemapping operator."""
    key = name.lower()
    if key not in _TONEMAP_REGISTRY:
        raise KeyError(
            f"Unknown tonemapper '{name}'. Available: {sorted(_TONEMAP_REGISTRY)}"
        )
    return _TONEMAP_REGISTRY[key]
