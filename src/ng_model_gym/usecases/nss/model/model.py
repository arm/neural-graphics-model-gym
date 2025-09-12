# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn

from ng_model_gym.core.utils.config_model import ConfigModel


def create_model(params: ConfigModel, device: torch.device) -> nn.Module:
    """Create specified model."""

    if params.version == 1:
        from ng_model_gym.usecases.nss.model.model_v1 import (  # pylint: disable=import-outside-toplevel
            create_feedback_model_with_nss,
        )

        return create_feedback_model_with_nss(params, device)

    raise ValueError(f"Model version {params.version} does not exist")
