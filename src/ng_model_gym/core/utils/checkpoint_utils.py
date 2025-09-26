# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
import time
from pathlib import Path
from typing import List, Optional

import torch

from ng_model_gym.core.utils.config_model import ConfigModel
from ng_model_gym.core.utils.types import TrainEvalMode
from ng_model_gym.usecases.nss.model.model import create_model

logger = logging.getLogger(__name__)


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

    trained_model = create_model(params, device)
    checkpoint = torch.load(model_path, weights_only=True)

    # If model is QAT, make sure it is in a traced state for loading in weights
    if (
        params.model_train_eval_mode == TrainEvalMode.QAT_INT8
        and not trained_model.nss_model.is_network_quantized
    ):
        trained_model.nss_model.quantize_modules(
            (8, trained_model.nss_model.autoencoder.in_channels, 128, 128),
            device=device,
        )

    logger.info(f"Loading model from checkpoint: {model_path}")
    trained_model.load_state_dict(checkpoint["model_state_dict"])

    return trained_model
