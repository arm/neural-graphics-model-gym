# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
import sys
from pathlib import Path

from ng_model_gym.utils.general_utils import create_directory


def setup_logging(
    logger_name,
    log_level=logging.INFO,
    output_dir="",
    log_file_name="output.log",
    stdout: bool = True,
):
    """
    Create logger, if it doesn't exist and set log level and output file.
    If the module name is used as logger_name, then nested modules can access via __name__.
    A log file will be written to output_dir/output.log.
    """
    # Create output directory if it doesn't exist.
    if not Path(output_dir).exists():
        create_directory(output_dir)

    set_log_level(logger_name, log_level)
    add_file_handler(logger_name, output_dir, log_file_name)

    if stdout:
        add_stdout_handler(logger_name)


def set_log_level(logger_name, log_level=logging.INFO):
    """
    Retrieve global logger and set level,
    getLogger will create a logger if it doesn't exist.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)


def add_file_handler(logger_name, output_dir="", log_file_name="output.log"):
    """
    Retrieve global logger and add file handler,
    getLogger will create a logger if it doesn't exist.
    """
    logger = logging.getLogger(logger_name)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_file_path = Path(output_dir, log_file_name)
    file_handler = logging.FileHandler(filename=log_file_path)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)


def add_stdout_handler(logger_name):
    """Add logging to stdout"""
    logger = logging.getLogger(logger_name)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
