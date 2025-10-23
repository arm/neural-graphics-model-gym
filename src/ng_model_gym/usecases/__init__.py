# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)


def import_usecase_files() -> None:
    """Function to import usecase models and datasets,
    importing all .py modules in the package."""

    for _, name, _ in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
        importlib.import_module(name)
        logger.debug(f"Imported module: {name}")
