# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import random
from typing import Dict, List, Union

import numpy as np
import torch

TensorData = Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]


def fix_randomness(seed, use_deterministic_cuda=False):
    """Set the seed in Python, PyTorch and Numpy so training is consistent."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Deterministic CUDA operations  (may reduce training performance)
    if torch.cuda.is_available() and use_deterministic_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def lerp_tensor(x: torch.Tensor, y: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Performs: `x * (1 - a) + y * a`"""
    return x * (1.0 - a) + y * a


def clamp_tensor(
    img: torch.Tensor, mini: torch.Tensor, maxi: torch.Tensor
) -> torch.Tensor:
    """Clips between input within range: `[mini, maxi]`"""
    return torch.maximum(torch.minimum(img, maxi), mini)
