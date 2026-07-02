# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import copy
import logging
import platform
import tempfile

import numpy as np

from ng_model_gym.core.config.config_model import ConfigModel
from ng_model_gym.core.utils.io.file_utils import create_directory

TEST_PARAMS_PRESETS = {
    "nss_v1": {
        "model": {
            "name": "NSS",
            "model_source": "prebuilt",
            "version": "1",
            "scale": 2.0,
            "recurrent_samples": 16,
            "quality": "high",
            "gt_history_augmentation": False,
            "gt_history_augmentation_chance": 30.0,
        },
        "dataset": {
            "name": "NSS",
            "version": "1",
            "path": {
                "train": "",
                "validation": "",
                "test": "",
            },
            "exposure": 2,
            "tonemapper": "reinhard",
            "health_check": True,
            "gt_augmentation": True,
            "num_workers": 0 if platform.system() == "Windows" else 4,
            "prefetch_factor": 1,
        },
        "train": {
            "loss_fn": "loss_v1",
            "loss_args": {
                "temporal_reg_weight": 0.7,
                "alpha_reg_weight": 0.0001,
                "temporal_reg_channels": 1,
                "min_weight": 0.1,
            },
            "fp32": {
                "optimizer": {
                    "optimizer_type": "lars_adam",
                    "learning_rate": "2e-3",
                    "eps": 1e-7,
                },
            },
        },
        "metrics": ["PSNR", "tPSNR", "RecPSNR", "SSIM"],
    },
    "nfru": {
        "model": {
            "name": "NFRU",
            "model_source": "prebuilt",
            "version": "1",
            "scale_factor": 2.0,
            "legacy_nfru_capture_paths": [],
            "dynamic_mask_is_runtime_accurate": False,
            "mv_similarity_threshold": None,
        },
        "dataset": {
            "name": "NFRU",
            "version": "1",
            "path": {"train": "", "validation": "", "test": ""},
            "health_check": True,
            "gt_augmentation": True,
            "num_workers": 0 if platform.system() == "Windows" else 4,
            "prefetch_factor": 1,
            "align_data": True,
            "color_preprocessing": {
                "train": {
                    "pipeline": [
                        ["reinhard"],
                        ["none", "contrast", "saturation", "temperature_tint"],
                    ],
                    "exposure": [1.0, 2.5],
                },
                "validation": {
                    "pipeline": ["reinhard"],
                    "exposure": 2.0,
                },
                "test": {
                    "pipeline": ["reinhard"],
                    "exposure": 2.0,
                },
            },
        },
        "train": {
            "loss_fn": "lpips_spatial_loss_v1",
            "loss_args": {"alpha": 0.1},
            "fp32": {
                "optimizer": {
                    "optimizer_type": "adam",
                    "learning_rate": "1e-3",
                    "eps": 1e-7,
                },
            },
        },
        "metrics": ["PSNR", "SSIM", "STLPIPS"],
    },
}


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


def _deep_merge_dict(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` into ``base`` and return ``base``."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def create_simple_params(
    usecase,
    dataset_path=None,
    output_dir="./output",
    checkpoints="./checkpoints",
) -> ConfigModel:
    """
    Returns configuration created with test parameters dependent on use-case.

    E.g. create_simple_params(usecase="nss_v1",
                            output_dir="./output",
                            dataset_path="path/to/dataset")
    """
    usecase_preset = TEST_PARAMS_PRESETS[usecase.lower()]

    if usecase_preset is None:
        raise ValueError(
            f"Unknown usecase '{usecase}'. Valid options: {TEST_PARAMS_PRESETS.keys()}"
        )

    model = copy.deepcopy(usecase_preset["model"])
    dataset = copy.deepcopy(usecase_preset["dataset"])
    metrics = copy.deepcopy(usecase_preset["metrics"])
    train_overrides = copy.deepcopy(usecase_preset.get("train", {}))

    dataset["path"] = {
        "train": dataset_path,
        "validation": dataset_path,
        "test": dataset_path,
    }

    # Create temp directory for fields mandating path. Individual tests should override with their
    # own path for these fields.
    temp_dir = tempfile.mkdtemp()

    default_params = {
        "model": model,
        "dataset": dataset,
        "output": {
            "dir": output_dir,
            "export_frame_png": False,
            "tensorboard_output_dir": temp_dir,
            "export": {
                "vgf_output_dir": temp_dir,
                "dynamic_shape": True,
            },
        },
        "metrics": metrics,
        "train": {
            "batch_size": 8,
            "seed": 123456,
            "perform_validate": False,
            "fp32": {
                "number_of_epochs": 1,
                "checkpoints": {
                    "dir": checkpoints,
                },
                "lr_scheduler": {
                    "type": "cosine_annealing",
                    "warmup_percentage": 0.05,
                    "min_lr": 2e-4,
                },
            },
            "qat": {
                "number_of_epochs": 1,
                "checkpoints": {
                    "dir": f"{checkpoints}/qat_checkpoints",
                },
                "lr_scheduler": {
                    "type": "cosine_annealing",
                    "warmup_percentage": 0.05,
                    "min_lr": 1e-5,
                },
                "optimizer": {
                    "optimizer_type": "lars_adam",
                    "learning_rate": "2e-3",
                    "eps": 1e-7,
                },
            },
        },
    }

    _deep_merge_dict(default_params["train"], train_overrides)

    # Create output and checkpoint directories
    # Output dir
    create_directory(default_params["output"]["dir"])
    # FP32 checkpoints dir
    create_directory(default_params["train"]["fp32"]["checkpoints"]["dir"])
    # QAT checkpoints dir
    create_directory(default_params["train"]["qat"]["checkpoints"]["dir"])

    return validate_params(default_params)


def validate_params(params: dict) -> ConfigModel:
    """Validate params dict and return ConfigModel object."""
    return ConfigModel.model_validate(params)
