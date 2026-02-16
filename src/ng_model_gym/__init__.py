# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING

# pylint: disable=import-outside-toplevel, too-many-return-statements

# noinspection PyUnresolvedReferences
__all__ = [
    "load_config_file",
    "generate_config_file",
    "logging_config",
    "do_training",
    "do_export",
    "do_evaluate",
    "print_config_options",
    "list_pretrained_models",
    "download_pretrained_model",
    "TrainEvalMode",
    "ExportType",
    "ProfilerType",
]


if TYPE_CHECKING:
    from ng_model_gym.api import (
        do_evaluate,
        do_export,
        do_training,
        ExportType,
        ProfilerType,
        TrainEvalMode,
    )
    from ng_model_gym.core.repos import (
        download_pretrained_model,
        list_pretrained_models,
    )
    from ng_model_gym.core.utils import (
        generate_config_file,
        load_config_file,
        print_config_options,
    )
    from ng_model_gym.core.utils.logging import logging_config
else:

    def __getattr__(attr):
        if attr == "do_training":
            from ng_model_gym.api import do_training

            return do_training
        if attr == "do_evaluate":
            from ng_model_gym.api import do_evaluate

            return do_evaluate
        if attr == "do_export":
            from ng_model_gym.api import do_export

            return do_export
        if attr == "list_pretrained_models":
            from ng_model_gym.core.repos import list_pretrained_models

            return list_pretrained_models
        if attr == "download_pretrained_model":
            from ng_model_gym.core.repos import download_pretrained_model

            return download_pretrained_model
        if attr == "load_config_file":
            from ng_model_gym.core.utils.config_utils import load_config_file

            return load_config_file
        if attr == "print_config_options":
            from ng_model_gym.core.utils.config_utils import print_config_options

            return print_config_options
        if attr == "logging_config":
            from ng_model_gym.core.utils.logging import logging_config

            return logging_config
        if attr == "generate_config_file":
            from ng_model_gym.core.utils.config_utils import generate_config_file

            return generate_config_file

        if attr == "TrainEvalMode":
            from ng_model_gym.api import TrainEvalMode

            return TrainEvalMode

        if attr == "ExportType":
            from ng_model_gym.api import ExportType

            return ExportType

        if attr == "ProfilerType":
            from ng_model_gym.api import ProfilerType

            return ProfilerType

        raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")


def __dir__():
    return list(__all__)
