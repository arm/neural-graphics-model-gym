# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
import tempfile

import numpy as np

from ng_model_gym.core.utils.config_model import ConfigModel
from ng_model_gym.core.utils.general_utils import create_directory


def clear_loggers() -> None:
    """Close the log handlers."""
    for _, logger in logging.Logger.manager.loggerDict.items():
        if not isinstance(logger, logging.PlaceHolder):
            for handler in logger.handlers:
                handler.close()
                logger.removeHandler(handler)


def get_test_representative_dataset(shape, size=100):
    """Returns a generator, which creates random values based on the shape provided."""

    def dataset():
        for _ in range(size):
            data = np.random.rand(shape)
            yield [data.astype(np.float32)]

    return dataset


def create_simple_params(
    output_dir="./output",
    dataset=None,
    checkpoints="./checkpoints",
    pretrained_weights_path=None,
) -> ConfigModel:
    """
    Returns configuration created with test parameters.

    E.g. create_simple_params(override_params={"train": {"batch_size": 4}}, output_dir="./output")
    """

    # Create temp directory for fields mandating path. Individual tests should override with their
    # own path for these fields.
    temp_dir = tempfile.mkdtemp()

    default_params = {
        "version": 1,
        "processing": {"shader_accurate": False},
        "dataset": {
            "path": {"train": dataset, "validation": dataset, "test": dataset},
            "health_check": True,
            "recurrent_samples": 16,
            "gt_augmentation": True,
            "exposure": 2,
            "tonemapper": "reinhard",
            "num_workers": 4,
            "prefetch_factor": 1,
        },
        "output": {
            "dir": output_dir,
            "export_frame_png": False,
            "tensorboard_output_dir": temp_dir,
            "export": {
                "vgf_output_dir": temp_dir,
                "dynamic_shape": False,
            },
        },
        "train": {
            "batch_size": 8,
            "resume": False,
            "scale": 2.0,
            "seed": 123456,
            "finetune": False,
            "pretrained_weights": pretrained_weights_path,
            "perform_validate": False,
            "fp32": {
                "number_of_epochs": 1,
                "checkpoints": {
                    "dir": checkpoints,
                    "save_frequency": "1",
                },
                "learning_rate": "2e-3",
                "cosine_annealing_scheduler_config": {
                    "warmup_percentage": 0.05,
                    "min_lr": 2e-4,
                },
            },
            "qat": {
                "number_of_epochs": 1,
                "checkpoints": {
                    "dir": f"{checkpoints}/qat_checkpoints",
                    "save_frequency": "1",
                },
                "learning_rate": "6e-4",
                "cosine_annealing_scheduler_config": {
                    "warmup_percentage": 0.05,
                    "min_lr": 1e-5,
                },
            },
        },
        "optimizer": {
            "learning_rate_scheduler": "cosine_annealing",
        },
    }

    # Create output and checkpoint directories
    # Output dir
    create_directory(default_params["output"]["dir"])
    # FP32 checkpoints dir
    create_directory(default_params["train"]["fp32"]["checkpoints"]["dir"])
    # QAT checkpoints dir
    create_directory(default_params["train"]["qat"]["checkpoints"]["dir"])

    return ConfigModel.model_validate(default_params)
