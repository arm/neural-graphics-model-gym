# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn

from ng_model_gym.core.config.config_model import ConfigModel
from ng_model_gym.core.data.data_utils import DataLoaderMode
from ng_model_gym.core.data.dataloader import get_dataloader
from ng_model_gym.core.model.base_ng_model import BaseNGModel
from ng_model_gym.core.model.model_factory import create_model
from ng_model_gym.core.model.model_tracer import model_tracer
from ng_model_gym.core.model.qat_checkpoint_compat import prepare_legacy_qat_state_dict
from ng_model_gym.core.utils.enum_definitions import ModelType, TrainEvalMode

logger = logging.getLogger(__name__)


def _prepare_qat_forward_input(
    model: BaseNGModel,
    params: ConfigModel,
) -> Tuple[Any, ...]:
    """Load a sample from train dataloader to trace model before loading real QAT weights."""

    train_path = getattr(params.dataset.path, "train", None)
    if train_path is None:
        raise RuntimeError(
            "Unable to prepare input data for QAT quantization. "
            "Set `dataset.path.train` in the config so a training sample can be loaded."
        )

    dataloader = get_dataloader(
        params,
        num_workers=0,
        prefetch_factor=1,
        loader_mode=DataLoaderMode.TRAIN,
    )

    try:
        batch = next(iter(dataloader))
    except StopIteration as exc:
        raise RuntimeError(
            "Training dataloader yielded no data. Ensure the dataset contains samples."
        ) from exc

    inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
    return model_tracer(model, inputs)


def latest_checkpoint_in_dir(user_checkpoint_dir: Path) -> Path:
    """Returns the path to the latest checkpoint file using file modification time"""

    if user_checkpoint_dir.is_file():
        if user_checkpoint_dir.suffix.lower() != ModelType.PT:
            raise ValueError(
                f"Weight file must have a .pt extension, not: {user_checkpoint_dir.suffix}"
            )
        return user_checkpoint_dir

    if not user_checkpoint_dir.exists() or not user_checkpoint_dir.is_dir():
        raise NotADirectoryError(
            f"Checkpoint directory {user_checkpoint_dir.absolute()} does not exist"
        )

    # Collect all .pt files within the directory tree
    ckpt_files = [p for p in user_checkpoint_dir.rglob("*.pt") if p.is_file()]

    if not ckpt_files:
        raise FileNotFoundError(
            f"Resume training option set but no .pt checkpoints in "
            f"{user_checkpoint_dir.absolute()} to restore from"
        )

    # Find the most recently modified file
    latest_checkpoint_path = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    return latest_checkpoint_path


def load_model_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    prepare_for_weights_load: bool = False,
) -> None:
    """Apply checkpoint compatibility transforms and load the model state."""
    state_dict = prepare_legacy_qat_state_dict(model, state_dict)
    if prepare_for_weights_load:
        prepare_hook = getattr(
            model,
            "prepare_checkpoint_state_dict_for_weights_load",
            None,
        )
        if prepare_hook is not None:
            state_dict = prepare_hook(state_dict)
    model.load_state_dict(state_dict)


def load_checkpoint(model_path: Path, params: ConfigModel, device: torch.device = None):
    """Create a model from checkpoint"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not model_path.exists() or not model_path.is_file():
        raise FileNotFoundError(f"Weight file not found: {model_path}")
    if model_path.suffix.lower() != ModelType.PT:
        raise ValueError(
            f"Weight file must have a .pt extension, not: {model_path.suffix}"
        )

    trained_model: BaseNGModel = create_model(params, device)
    ng_model = trained_model
    checkpoint = torch.load(model_path, weights_only=True)

    if isinstance(trained_model, BaseNGModel):
        ng_model = trained_model
    else:
        raise ValueError("trained_model is not a valid type")

    # If model is QAT, make sure it is in a traced state for loading in weights
    if (
        params.model_train_eval_mode == TrainEvalMode.QAT_INT8
        and not ng_model.is_network_quantized
    ):
        forward_input_data = _prepare_qat_forward_input(trained_model, params)
        ng_model.quantize_modules(forward_input_data)

    logger.info(f"Loading model from checkpoint: {model_path}")
    model_state_dict = checkpoint["model_state_dict"]
    load_model_state_dict(
        trained_model,
        model_state_dict,
        prepare_for_weights_load=True,
    )

    return trained_model
