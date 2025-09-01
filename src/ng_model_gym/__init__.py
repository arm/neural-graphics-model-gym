# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING

__version__ = "0.1.0"
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
    "TrainEvalMode",
    "ExportType",
    "ProfilerType",
    "__version__",
]


if TYPE_CHECKING:
    from ng_model_gym.api import (
        do_evaluate,
        do_export,
        do_training,
        ExportType,
        generate_config_file,
        load_config_file,
        logging_config,
        print_config_options,
        ProfilerType,
        TrainEvalMode,
    )
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
        if attr == "load_config_file":
            from ng_model_gym.api import load_config_file

            return load_config_file
        if attr == "print_config_options":
            from ng_model_gym.api import print_config_options

            return print_config_options
        if attr == "logging_config":
            from ng_model_gym.api import logging_config

            return logging_config
        if attr == "generate_config_file":
            from ng_model_gym.api import generate_config_file

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
