# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def create_directory(dir_path: Union[str, Path]):
    """Create directory if it doesn't already exist."""
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory at {dir_path} already exists or has been created.")
    except (FileExistsError, PermissionError, ValueError) as e:
        logger.error(e)
        raise
