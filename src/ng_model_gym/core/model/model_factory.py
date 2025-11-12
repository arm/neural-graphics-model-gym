# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Type

import torch

from ng_model_gym.core.model.base_ng_model import BaseNGModel
from ng_model_gym.core.model.base_ng_model_wrapper import BaseNGModelWrapper
from ng_model_gym.core.model.model_registry import get_model_key, MODEL_REGISTRY
from ng_model_gym.core.model.recurrent_model import FeedbackModel
from ng_model_gym.core.utils.config_model import ConfigModel
from ng_model_gym.core.utils.types import TrainEvalMode

logger = logging.getLogger(__name__)


def get_model_from_config(params: ConfigModel) -> Type[BaseNGModel]:
    """Return the registered model class from the model registry,
    using the name and version supplied in the config file."""

    model_name = params.model.name
    model_version = params.model.version

    model_key = get_model_key(model_name, model_version)

    return MODEL_REGISTRY.get(model_key)


def create_model(
    params: ConfigModel, device: torch.device
) -> BaseNGModelWrapper | BaseNGModel:
    """Create specified model."""

    model_cls = get_model_from_config(params)
    model = model_cls(params).to(device)

    # Make a quantized model if doing QAT
    if params.model_train_eval_mode == TrainEvalMode.QAT_INT8:
        model.is_qat_model = True

    elif params.model_train_eval_mode != TrainEvalMode.FP32:
        raise ValueError(f"Unsupported training mode: {params.model_train_eval_mode}")

    # Make FeedbackModel if recurrent_samples is set
    if params.dataset.recurrent_samples:
        logger.info("Creating FeedbackModel for recurrent inference")
        model = FeedbackModel(
            model,
            recurrent_samples=params.dataset.recurrent_samples,
            device=device,
        )

    return model
