# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
import platform
import sys
import warnings
from pathlib import Path

import GPUtil
import psutil
import torch

from ng_model_gym.utils.config_model import ConfigModel
from ng_model_gym.utils.general_utils import create_directory

logger = logging.getLogger(__name__)


def logging_config(
    params: ConfigModel,
    logger_name: str = "ng_model_gym",
    log_level: int = logging.INFO,
) -> None:
    """
    Configure ng_model_gym package logging to file and console.

     Args:
        params (ConfigModel): Configuration model containing `output.dir`.
        logger_name (str): Root logger name (e.g., 'ng_model_gym'). Defaults to empty.
        log_level (int): Logging level (e.g logging.INFO, logging.ERROR). Defaults to logging.INFO.

     Example:
        >>> logging_config(params, "ng_model_gym", logging.INFO)
    """
    # Create output directory if it doesn't exist.
    create_directory(params.output.dir)

    # Setup general logger which will be used throughout all modules.
    # The root name is used to allow nested modules to pick up the same logger using __name__.
    setup_logging(logger_name, log_level, params.output.dir)
    logging.getLogger(logger_name).propagate = False

    if log_level == logging.DEBUG:
        # Setup PyTorch logger, which will write to the same output file.
        setup_logging("torch", logging.INFO, params.output.dir)

    if log_level != logging.DEBUG:
        filter_warnings()


def log_gpu_torch():
    """Log information about GPU used in training."""

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"GPUs available: {num_gpus}")
        for i in range(num_gpus):
            logger.info(f"Using GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("No GPU available. Using CPU for training.")


def filter_warnings() -> None:
    """Filter warnings from logging."""
    warnings.filterwarnings(
        "ignore",
        message=r"The parameter 'pretrained' is deprecated.*",
        category=UserWarning,
        module=r"torchvision\.models\._utils",
    )

    warnings.filterwarnings(
        "ignore",
        message=r"Arguments other than a weight enum or `None` for 'weights' are deprecated.*",
        category=UserWarning,
        module=r"torchvision\.models\._utils",
    )

    warnings.filterwarnings(
        "ignore",
        message=r"TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included.*",
        category=UserWarning,
        module=r"torch\.utils\.cpp_extension",
    )

    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API.*",
        category=UserWarning,
        module=r"slangtorch\.slangtorch",
    )

    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API.*",
        category=UserWarning,
        module=r"executorch\.exir\.dialects\.edge\._ops",
    )

    warnings.filterwarnings(
        "ignore",
        message=(
            r"To copy construct from a tensor, it is recommended to use "
            r"sourceTensor\.detach\(\)\.clone\(\)*"
        ),
        category=UserWarning,
        module=r"executorch\.backends\.arm\.quantizer\.quantization_config",
    )

    warnings.filterwarnings(
        "ignore",
        message=r"erase_node*",
        category=UserWarning,
        module=r"torch\.fx\.graph",
    )


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
    global_logger = logging.getLogger(logger_name)
    global_logger.setLevel(log_level)


def add_file_handler(logger_name, output_dir="", log_file_name="output.log"):
    """
    Retrieve global logger and add file handler,
    getLogger will create a logger if it doesn't exist.
    """
    global_logger = logging.getLogger(logger_name)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_file_path = Path(output_dir, log_file_name)
    file_handler = logging.FileHandler(filename=log_file_path)
    file_handler.setFormatter(formatter)

    global_logger.addHandler(file_handler)


def add_stdout_handler(logger_name):
    """Add logging to stdout"""
    global_logger = logging.getLogger(logger_name)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    global_logger.addHandler(handler)


def log_machine_info():
    """Log info about training machine."""
    sys_lines = [
        "\n-------------- Training machine info --------------",
        f"Name: {platform.uname().node}",
        f"CPU: {platform.processor()}",
        f"Physical cores: {psutil.cpu_count(logical=False)}",
        f"Total cores: {psutil.cpu_count(logical=True)}",
        f"Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB",
        f"System: {platform.uname().system}",
        f"Version: {platform.uname().version}",
        "---------------------------------------------------",
    ]
    logger.info("\n".join(sys_lines))

    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            for i, gpu in enumerate(gpus):
                logger.info(
                    f"GPU {i} memory — total: {gpu.memoryTotal} MB; "
                    f"free: {gpu.memoryFree} Mb; Currently in use: {gpu.memoryUsed} MB"
                )
        else:
            logger.info("No GPUs found.")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.info(f"Something is wrong with GPU drivers: {e}")
