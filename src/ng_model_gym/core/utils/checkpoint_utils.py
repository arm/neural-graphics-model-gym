# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
import time
from pathlib import Path
from typing import Any, List, Optional

import torch

from ng_model_gym.core.data.dataloader import get_dataloader
from ng_model_gym.core.data.utils import DataLoaderMode
from ng_model_gym.core.model.base_ng_model import BaseNGModel
from ng_model_gym.core.model.base_ng_model_wrapper import BaseNGModelWrapper
from ng_model_gym.core.model.model_factory import create_model
from ng_model_gym.core.model.model_tracer import model_tracer
from ng_model_gym.core.utils.config_model import ConfigModel
from ng_model_gym.core.utils.types import TrainEvalMode

logger = logging.getLogger(__name__)


def replace_prefix_in_state_dict(
    state_dict: dict[str, Any],
    old_prefix: str,
    new_prefix: str,
) -> dict[str, Any]:
    """
    Helper method to replace prefix in a state dict. This borrows from
    `torch.nn.modules.utils.consume_prefix_in_state_dict_if_present`. Statedict modified in place
    but also returned for convenience
    """

    if not isinstance(old_prefix, str) or not isinstance(new_prefix, str):
        raise TypeError("old_prefix and new_prefix must be strings")

    if old_prefix == new_prefix:
        return state_dict

    old_prefix_with_sep = f"{old_prefix}."

    # Rename parameter/buffer keys
    keys = list(state_dict.keys())
    for key in keys:
        if key == old_prefix:
            new_key = new_prefix
        elif key.startswith(old_prefix_with_sep):
            suffix = key[len(old_prefix_with_sep) :]
            new_key = f"{new_prefix}.{suffix}"
        else:
            continue
        state_dict[new_key] = state_dict.pop(key)

    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        meta_data_keys = list(metadata.keys())
        old_base = old_prefix.replace(".", "")
        new_base = new_prefix.replace(".", "")

        for key in meta_data_keys:
            if len(key) == 0:
                continue
            if key == old_base:
                metadata[new_base] = metadata.pop(key)
            elif key.startswith(old_prefix_with_sep):
                suffix = key[len(old_prefix_with_sep) :]
                new_key = f"{new_prefix}.{suffix}"
                metadata[new_key] = metadata.pop(key)

    return state_dict


def remap_feedback_model_state_dict(state_dict):
    """Rename old FeedbackModel prefixes from `nss_model` to `ng_model`."""
    old_prefix = "nss_model"
    new_prefix = "ng_model"

    has_new_prefix = any(
        key == new_prefix or key.startswith(f"{new_prefix}.")
        for key in state_dict.keys()
    )
    has_old_prefix = any(
        key == old_prefix or key.startswith(f"{old_prefix}.")
        for key in state_dict.keys()
    )

    if not has_old_prefix or has_new_prefix:
        return state_dict

    logger.warning(
        "Loading FeedbackModel state dict with old style naming scheme."
        f" Modifying state dict from {old_prefix} namespace to {new_prefix}"
    )

    return replace_prefix_in_state_dict(
        state_dict,
        old_prefix=old_prefix,
        new_prefix=new_prefix,
    )


def is_timestamp(dir_name) -> bool:
    """Checks if dir_name matches timestamp schema"""
    try:
        time.strptime(dir_name, "%y-%m-%d_%H-%M-%S")
        return True
    except ValueError:
        return False


def latest_training_run_dir(checkpoint_dir: Path) -> Path:
    """Find the directory containing the latest training run checkpoints"""

    # Verify directory set in user config exists
    if not checkpoint_dir.is_dir():
        logger.error(
            f"Checkpoint directory {checkpoint_dir.absolute()} set in config does not exist"
        )
        raise NotADirectoryError(
            f"Checkpoint directory {checkpoint_dir.absolute()} set in config does not exist"
        )

    timestamped_directories = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and is_timestamp(item.name):
            timestamped_directories.append(item.name)

    if not timestamped_directories:
        logger.error(
            f"Resume training option set but no training runs in "
            f"{checkpoint_dir.absolute()} matching format '%y-%m-%d_%H-%M-%S' to restore from"
        )
        raise LookupError(
            f"Resume training option set but no training runs in "
            f"{checkpoint_dir.absolute()} matching format '%y-%m-%d_%H-%M-%S' to restore from"
        )

    # Max() on list of string timestamps returns most recent
    latest_checkpoint_directory = Path(
        checkpoint_dir, f"{max(timestamped_directories)}"
    )
    return latest_checkpoint_directory


def latest_checkpoint_path(user_checkpoint_dir: Path):
    """Returns the path to the latest checkpoint file"""

    latest_checkpoint_directory = latest_training_run_dir(user_checkpoint_dir)
    is_checkpoint_file = lambda ckpt: ckpt.is_file() and ckpt.suffix in ".pt"

    # Get list of checkpoint files with pattern "ckpt-XX" where suffix XX is the epoch
    checkpoints_in_directory: List[Optional[int]] = [
        int(str(f.stem).split("-", maxsplit=1)[-1])
        for f in latest_checkpoint_directory.iterdir()
        if is_checkpoint_file(f) and str(f.stem).split("-", maxsplit=1)[-1].isdigit()
    ]
    if not checkpoints_in_directory:
        logger.error(
            f"Resume training option set but no .pt checkpoints in "
            f"{latest_checkpoint_directory.absolute()} to restore from"
        )
        raise FileNotFoundError(
            f"Resume training option set but no .pt checkpoints in "
            f"{latest_checkpoint_directory.absolute()} to restore from"
        )
    latest_checkpoint = Path(
        latest_checkpoint_directory, f"ckpt-{max(checkpoints_in_directory)}.pt"
    )
    return latest_checkpoint


def load_checkpoint(model_path: Path, params: ConfigModel, device: torch.device = None):
    """Create a model from checkpoint"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not model_path.exists() or not model_path.is_file():
        raise FileNotFoundError(f"Weight file not found: {model_path}")
    if model_path.suffix.lower() != ".pt":
        raise ValueError(
            f"Weight file must have a .pt extension, not: {model_path.suffix}"
        )

    trained_model: BaseNGModelWrapper | BaseNGModel = create_model(params, device)
    ng_model = trained_model
    checkpoint = torch.load(model_path, weights_only=True)

    if isinstance(trained_model, BaseNGModelWrapper):
        ng_model = trained_model.get_ng_model()
    elif isinstance(trained_model, BaseNGModel):
        ng_model = trained_model
    else:
        raise ValueError("trained_model is not a valid type")

    # If model is QAT, make sure it is in a traced state for loading in weights
    if (
        params.model_train_eval_mode == TrainEvalMode.QAT_INT8
        and not ng_model.is_network_quantized
    ):
        dataloader = get_dataloader(
            params,
            num_workers=params.dataset.num_workers,
            prefetch_factor=params.dataset.prefetch_factor,
            loader_mode=DataLoaderMode.TRAIN,
            trace_mode=False,
        )

        # Get a real batch from the dataloader
        data = next(iter(dataloader))[0]
        forward_input_data = model_tracer(trained_model, data)

        ng_model.quantize_modules(forward_input_data)

    logger.info(f"Loading model from checkpoint: {model_path}")
    model_state_dict = remap_feedback_model_state_dict(checkpoint["model_state_dict"])
    trained_model.load_state_dict(model_state_dict)

    return trained_model
