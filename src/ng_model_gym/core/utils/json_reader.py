# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import json
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def read_json_file(json_file_path: Path) -> Dict:
    """Create dictionary from json file"""

    if not isinstance(json_file_path, Path):
        raise TypeError("json_file_path must be of type Path from Pathlib")

    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except FileNotFoundError:
        logger.error(f"Config file not found for path: {json_file_path.absolute()}")
        raise
    except json.JSONDecodeError as e:
        logger.error(
            f"Unable to decode JSON in file: {json_file_path.absolute()}\n {e}"
        )
        raise
