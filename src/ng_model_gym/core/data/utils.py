# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
from enum import Enum

import torch

logger = logging.getLogger(__name__)

HDR_MAX = 65504.0


class DataLoaderMode(str, Enum):
    """Data loader mode, either train, validation or test."""

    TRAIN = "train"
    VAL = "validation"
    TEST = "test"


class DatasetType(str, Enum):
    """Dataset type, either .safetensors, .h5, .pt or .tftecord."""

    SAFETENSOR = ".safetensors"
    H5 = ".h5"
    PT = ".pt"
    TFRECORD = ".tfrecord"


class ToneMapperMode(str, Enum):
    """Supported modes for tone mapping."""

    REINHARD = "reinhard"
    KARIS = "karis"
    LOG = "log"
    LOG10 = "log10"
    LOG_NORM = "log_norm"
    ACES = "aces"


def tonemap_forward(
    x: torch.Tensor, max_val: float = 1.0, mode: ToneMapperMode = ToneMapperMode.KARIS
) -> torch.Tensor:
    """
    Reinhard:
    http://behindthepixels.io/assets/files/TemporalAA.pdf#page5

    Karis:
    http://graphicrants.blogspot.com/2013/12/tone-mapping.html
    """
    x = torch.max(x, torch.zeros_like(x))

    # Tonemap
    if mode == ToneMapperMode.REINHARD:
        x = x * (max_val / (1.0 + x))
    elif mode == ToneMapperMode.KARIS:
        # calculate dim based on x number of dimensions.
        luma = torch.max(x, dim=x.ndim - 3, keepdim=True).values
        x = x * (max_val / (1.0 + luma))
    elif mode == ToneMapperMode.LOG10:
        x = torch.log10(x + 1.0)
    elif mode == ToneMapperMode.LOG:
        x = torch.log(x + 1.0)
    elif mode == ToneMapperMode.LOG_NORM:
        log_norm_scale = 1.0 / torch.log(torch.tensor(HDR_MAX + 1.0))
        x = torch.log(x + 1.0) * log_norm_scale
    elif mode == ToneMapperMode.ACES:
        # See ACES in action and its inverse at https://www.desmos.com/calculator/n1lkpc6hwq
        k_aces_a = torch.tensor(2.51, dtype=x.dtype)
        k_aces_b = torch.tensor(0.03, dtype=x.dtype)
        k_aces_c = torch.tensor(2.43, dtype=x.dtype)
        k_aces_d = torch.tensor(0.59, dtype=x.dtype)
        k_aces_e = torch.tensor(0.14, dtype=x.dtype)
        x = (x * (k_aces_a * x + k_aces_b)) / (x * (k_aces_c * x + k_aces_d) + k_aces_e)
    else:
        raise ValueError(f"Tonemap: {mode} unsupported")

    return torch.clamp(x, 0.0, max_val)


def tonemap_inverse(
    x: torch.Tensor, max_val: float = 1.0, mode: ToneMapperMode = ToneMapperMode.KARIS
) -> torch.Tensor:
    """
    Reinhard:
    http://behindthepixels.io/assets/files/TemporalAA.pdf#page5

    Karis:
    http://graphicrants.blogspot.com/2013/12/tone-mapping.html
    """
    x = torch.max(x, torch.zeros_like(x))

    # Tonemap
    if mode == ToneMapperMode.REINHARD:
        x = torch.clamp(
            x,
            0.0,
            tonemap_forward(
                torch.tensor(HDR_MAX, dtype=x.dtype), mode=ToneMapperMode.REINHARD
            ),
        )
        x = x * (max_val / (1.0 - x))
    elif mode == ToneMapperMode.KARIS:
        x = torch.clamp(
            x,
            torch.tensor(0.0),
            tonemap_forward(
                torch.reshape(torch.tensor(HDR_MAX, dtype=x.dtype), (1, 1, 1, 1)),
                mode=ToneMapperMode.KARIS,
            ),
        )
        # calculate dim based on x number of dimensions.
        luma = torch.max(x, dim=x.ndim - 3, keepdim=True).values
        x = x * (max_val / (1.0 - luma))
    elif mode == ToneMapperMode.LOG:
        x = torch.exp(x) - 1.0
    elif mode == ToneMapperMode.LOG10:
        x = torch.pow(10, x) - 1.0
    elif mode == ToneMapperMode.LOG_NORM:
        log_norm_scale = 1.0 / torch.log(torch.tensor(HDR_MAX + 1.0))
        x = torch.exp(x / log_norm_scale) - 1.0
    elif mode == ToneMapperMode.ACES:
        # https://www.desmos.com/calculator/n1lkpc6hwq
        k_aces_a = torch.tensor(2.51, dtype=x.dtype)
        k_aces_b = torch.tensor(0.03, dtype=x.dtype)
        k_aces_c = torch.tensor(2.43, dtype=x.dtype)
        k_aces_d = torch.tensor(0.59, dtype=x.dtype)
        k_aces_e = torch.tensor(0.14, dtype=x.dtype)
        res = k_aces_d * x - k_aces_b
        res += torch.sqrt(
            x * x * (k_aces_d * k_aces_d - 4.0 * k_aces_e * k_aces_c)
            + x * (4.0 * k_aces_e * k_aces_a - 2.0 * k_aces_b * k_aces_d)
            + k_aces_b * k_aces_b
        )
        res /= 2.0 * k_aces_a - 2.0 * k_aces_c * x
        x = res
    else:
        raise ValueError(f"Tonemap: {mode} unsupported")

    return x
