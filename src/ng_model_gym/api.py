# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import re
import shutil
import sys
import warnings
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Callable, List, Tuple, Union

import torch
from pydantic import ValidationError
from rich import print_json
from rich.console import Console
from rich.table import Column, Table
from torch.profiler import schedule

# pylint: disable-next=unused-import
import ng_model_gym.utils.executorch_patcher  # Patch ExecuTorch
from ng_model_gym.nss.evaluators import ModelEvaluator
from ng_model_gym.trainer import Trainer
from ng_model_gym.utils.checkpoint_utils import load_checkpoint
from ng_model_gym.utils.config_model import ConfigModel, OutputDirModel
from ng_model_gym.utils.general_utils import create_directory
from ng_model_gym.utils.gpu_log_decorator import gpu_log_decorator
from ng_model_gym.utils.json_reader import read_json_file
from ng_model_gym.utils.logging import setup_logging
from ng_model_gym.utils.memory_log_decorator import memory_log_decorator
from ng_model_gym.utils.time_decorator import time_decorator
from ng_model_gym.utils.types import ExportType, ProfilerType, TrainEvalMode

logger = logging.getLogger(__name__)

DEFAULT_PATH = "ng_model_gym.nss.configs"


def generate_config_file(save_dir: Union[str, Path, None] = None) -> Tuple[Path, Path]:
    """
    Generate a JSON configuration template and its schema file. This is used to configure training.

    Args:
        save_dir (Union[str, Path, None]): Directory to save config and schema. If None,
         uses current directory.

    Returns:
        Tuple[Path, Path]: Paths to the generated configuration JSON and schema JSON files.

    Example:
        >>> config_output_path, schema_path = generate_config_file(Path('./output'))
    """

    if save_dir:
        output_dir = Path(save_dir)
        if not output_dir.exists() or not output_dir.is_dir():
            raise FileNotFoundError(f"Provided save_dir '{output_dir}' does not exist.")
    else:
        output_dir = Path(".")

    placeholders = {
        "dataset.path.train": "<PATH/TO/TRAIN_DATA>",
        "dataset.path.validation": "<PATH/TO/VALIDATION_DATA>",
        "dataset.path.test": "<PATH/TO/TEST_DATA>",
        "train.fp32.checkpoints.dir": "<OUTPUT/PATH/FOR/CHECKPOINTS>",
        "train.qat.checkpoints.dir": "<OUTPUT/PATH/FOR/CHECKPOINTS>",
    }

    # Load internal default config
    default_config_path = files(DEFAULT_PATH) / "default.json"
    default_config: dict = read_json_file(default_config_path)

    for json_loc, placeholder_text in placeholders.items():
        dict_key_path: List[str] = json_loc.split(".")
        config_section = default_config

        # Iterate over dict keys e.g dict_key_path = ["dataset", "path", "train"]
        for dict_key in dict_key_path[:-1]:
            # Check the json path provided in placeholders is valid
            if dict_key not in config_section or not isinstance(
                config_section[dict_key], dict
            ):
                raise KeyError(
                    f"Invalid config path {json_loc} to replace with a placeholder"
                )
            # Access sub dict e.g config_section["dataset"]
            config_section = config_section[dict_key]

        final_key = dict_key_path[-1]
        if final_key not in config_section:
            raise KeyError(
                f"Invalid config path {json_loc} to replace with a placeholder"
            )

        config_section[final_key] = placeholder_text

    file_name = "config"
    suffix = ".json"
    config_output_path = output_dir / f"{file_name}{suffix}"

    # If a file already exists "config.json", create config_1.json etc
    if config_output_path.exists():
        count = 1
        file_path = output_dir / f"{file_name}_{count}{suffix}"
        while file_path.exists():
            count += 1
            file_path = output_dir / f"{file_name}_{count}{suffix}"
        config_output_path = file_path

    # Write the config file
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(default_config, f, indent=4)

    # Copy schema_config.json as well
    schema_config_path = files(DEFAULT_PATH) / "schema_config.json"
    shutil.copy(src=schema_config_path, dst=output_dir)

    schema_path = output_dir / "schema_config.json"
    return config_output_path, schema_path


def load_config_file(user_config_path: Path) -> ConfigModel:
    """
    Load and validate a JSON configuration file. The created config object is used as an argument
    to other API functions.

    Note: SystemExit - if validation fails, after printing errors, exits with status code 1

    Args:
        user_config_path (Path): Path to a user-provided JSON config file

    Returns:
        ConfigModel: Parsed and validated configuration model.

    Example:
        >>> params = load_config_file(Path("config.json"))
    """

    user_config: dict = read_json_file(user_config_path)

    try:
        return ConfigModel.model_validate(user_config)
    except ValidationError as e:
        validation_errors = e.errors()

        # Set up logging for validating config file, logging to file only
        # Try to get the user's specified output_dir, fall back to ./output if not valid
        output_dir = user_config.get("output", {}).get("dir")

        try:
            output_dir = OutputDirModel(dir=output_dir).dir

        except ValidationError:
            output_dir = Path("./output")

        setup_logging(
            logger_name=f"{__name__}.config_validation",
            log_level=logging.ERROR,
            output_dir=output_dir,
            stdout=False,
        )

        config_logger = logging.getLogger(f"{__name__}.config_validation")
        config_logger.propagate = False

        # Format validation errors into a table
        table = Table(
            Column(
                header="Validation Issue",
                justify="center",
                style="red",
                no_wrap=False,
                overflow="fold",
            ),
            Column(header="Details", justify="center", no_wrap=False, overflow="fold"),
            Column(
                header="Your input",
                justify="center",
                style="blue",
                no_wrap=False,
                overflow="fold",
            ),
            Column(
                header="Location in JSON",
                justify="center",
                style="green",
                no_wrap=False,
                overflow="fold",
            ),
            show_lines=True,
            expand=True,
        )

        config_logger.error("Config validation errors:")

        # Create table rows sorted by type e.g. missing, int_parsing
        for error_ctx in sorted(validation_errors, key=lambda e: e["type"]):
            val_type = error_ctx["type"]
            message = error_ctx["msg"]

            user_input = "" if val_type == "missing" else str(error_ctx["input"])

            # Location is a tuple, turn into list of strings and use dot instead of space
            location_in_json = ".".join(map(str, [*error_ctx["loc"]]))

            table.add_row(val_type, message, user_input, location_in_json)

            # Log errors
            config_logger.error(
                f"[{val_type}] {message} (input={user_input}, location={location_in_json})"
            )

        # Instantiate Rich console and print table
        console = Console()
        console.print(table)

        num_errors = len(validation_errors)

        error_str = (
            f"\n[bold]Configuration has [red]{num_errors}[/red] "
            f"{'issues' if num_errors > 1 else 'issue'}[/bold]"
        )

        console.print(error_str)

        error_str = re.sub(r"\[/?[^\]]+\]", "", error_str).lstrip("\n")
        config_logger.error(error_str)

        sys.exit(1)


def print_config_options() -> None:
    """
    Print the JSON configuration schema, listing each parameter with its type and description

    Example:
        >>> print_config_options()
    """
    schema_config_path = files(DEFAULT_PATH) / "schema_config.json"
    with schema_config_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    print_json(json.dumps(data))


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

    if log_level != logging.DEBUG:
        # ignore the pretrained deprecation warning in torchvision.models._utils
        warnings.filterwarnings(
            "ignore",
            message=r"The parameter 'pretrained' is deprecated.*",
            category=UserWarning,
            module=r"torchvision\.models\._utils",
        )

        # ignore the weights deprecation warning in torchvision.models._utils
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

    if log_level == logging.DEBUG:
        # Setup PyTorch logger, which will write to the same output file.
        setup_logging("torch", logging.INFO, params.output.dir)


def log_gpu_torch():
    """Log information about GPU used in training."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"GPUs available: {num_gpus}")
        for i in range(num_gpus):
            logger.info(f"Using GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("No GPU available. Using CPU for training.")


def _trace_profiler_wrapper(func: Callable, *args, trace_output_dir: Path):
    """Wraps a function call with PyTorch profiler and saves trace output."""
    # This schedule was introduced to circumvent a bug in PyTorch
    # https://github.com/pytorch/pytorch/issues/109969
    # Specifically, the repeat=1 solves the issue
    prof_sched = schedule(
        skip_first=5,  # skip 5 iterations
        wait=5,  # then wait 5 steps
        warmup=2,  # warm up for 2 steps
        active=2,  # record 2 steps
        repeat=1,  # run this cycle exactly once
    )
    with torch.profiler.profile(
        activities=(torch.profiler.supported_activities()),
        with_flops=True,
        with_stack=True,
        profile_memory=True,
        record_shapes=True,
        schedule=prof_sched,
    ) as prof:
        logger.info("PyTorch profiler is enabled")
        func(*args, prof)

    timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    trace_path = Path(trace_output_dir) / f"trace-{timestamp}.json"
    logger.info("Writing profiler trace to disk...")
    prof.export_chrome_trace(str(trace_path))
    logger.info(f"Profiler trace saved to {str(trace_path.absolute())}")
    logger.info("JSON trace file can be viewed at https://ui.perfetto.dev/")


def _cuda_profiler_wrapper(func: Callable, *args, trace_output_dir: Path):
    """Wraps a function call with PyTorch's CUDA memory profiling and saves a snapshot."""
    torch.cuda.memory._record_memory_history()
    logger.info("CUDA memory profiler is enabled")
    func(*args)
    timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    cuda_trace_path = Path(trace_output_dir) / f"cuda_profiler-{timestamp}.pickle"
    logger.info("Writing CUDA memory snapshot to disk...")
    torch.cuda.memory._dump_snapshot(str(cuda_trace_path))
    logger.info(f"CUDA memory profile saved to {str(cuda_trace_path.absolute())}")
    logger.info("Pickle file can be viewed at https://pytorch.org/memory_viz")


@memory_log_decorator
@time_decorator
@gpu_log_decorator(enabled=True, log_level=logging.DEBUG)
def do_training(
    params: ConfigModel,
    training_mode: TrainEvalMode,
    profile_setting: ProfilerType = ProfilerType.DISABLED,
):
    """
    Run training on the model specified in the configuration.

    Args:
        params (ConfigModel): Configuration model from `load_config_file()`.
        training_mode (TrainEvalMode): Select FP32 training or QAT INT8
        profile_setting (Optional, ProfilerType, ): Profiling strategy to use during training:
          - `ProfilerType.DISABLED` (default): No profiling
          - `ProfilerType.TRACE`: CPU tracing
          - `ProfilerType.GPU_MEMORY`: GPU memory profiling
    Returns:
        Tuple[torch.nn.Module, Path]: Trained model and path to its checkpoint.

    Example:
       >>> model, checkpoint_path = do_training(params, TrainEvalMode.FP32)
    """

    if not isinstance(training_mode, TrainEvalMode):
        raise ValueError("training_mode parameter is not set correctly")

    if params.dataset.path.train is None:
        raise ValueError("Config error: No path specified for the train dataset path")

    if params.train.perform_validate and params.dataset.path.validation is None:
        raise ValueError(
            "Config error: Perform validate is enabled and no path specified for validation dataset"
        )

    params.model_train_eval_mode = training_mode
    log_gpu_torch()

    trained_model = Trainer(params)

    # Train the model (with optional profiling)
    if profile_setting == ProfilerType.TRACE:
        _trace_profiler_wrapper(
            trained_model.train, trace_output_dir=trained_model.model_save_path
        )
    elif profile_setting == ProfilerType.GPU_MEMORY:
        _cuda_profiler_wrapper(
            trained_model.train, trace_output_dir=trained_model.model_save_path
        )
    else:
        trained_model.train()

    return trained_model.model, trained_model.latest_model_save_path


@memory_log_decorator
@time_decorator
def do_evaluate(
    params: ConfigModel,
    model_path: Union[str, Path],
    model_type: TrainEvalMode,
    profile_setting: ProfilerType = ProfilerType.DISABLED,
):
    """
    Evaluate a trained PyTorch model on a test dataset. Prints and saves metrics to a
    JSON file in the output directory location found in the config.

    Args:
        params (ConfigModel): Configuration object obtained via `load_config_file(path)`.
        model_path (str or Path): Filesystem path to the model checkpoint (.pt or .pth file).
        model_type (TrainEvalMode): Model type e.g. `TrainEvalMode.FP32` or `TrainEvalMode.QAT`.
        profile_setting (Optional, ProfilerType, ): Profiling strategy to use during evaluation:
            - `ProfilerType.DISABLED` (default): No profiling
            - `ProfilerType.TRACE`: CPU tracing
            - `ProfilerType.GPU_MEMORY`: GPU memory profiling

    Example:
        >>> params = load_config_file("config.json")
        >>> do_evaluate(
        ...     params=params,
        ...     model_path="/checkpoints/model_latest.pt",
        ...     model_type=TrainEvalMode.FP32,
        ... )
    """

    logger.info("Evaluating the trained model...")

    # Copy original params, so we can modify freely
    # (prevent side-effects modifying original params passed by user)
    params_for_eval = params.model_copy(deep=True)

    if params_for_eval.dataset.path.test is None:
        raise ValueError("Config error: No test dataset path provided for evaluation")

    model_path = Path(model_path)
    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"Expected a .pt file, got {model_path.name}")

    model_path = Path(model_path)

    # Load user's specified model
    params_for_eval.model_train_eval_mode = model_type
    model = load_checkpoint(model_path, params_for_eval)

    # Create Evaluator object
    # Set temporary values for parameters to allow evaluation of test set
    logger.debug("Temporarily setting recurrent_samples to 1")
    params_for_eval.dataset.recurrent_samples = 1
    model_evaluator = ModelEvaluator(model, params_for_eval)

    # Evaluate model (with optional profiling)
    if profile_setting == ProfilerType.TRACE:
        _trace_profiler_wrapper(
            model_evaluator.evaluate, trace_output_dir=params.output.dir
        )
    elif profile_setting == ProfilerType.GPU_MEMORY:
        _cuda_profiler_wrapper(
            model_evaluator.evaluate, trace_output_dir=params.output.dir
        )
    else:
        model_evaluator.evaluate()


@memory_log_decorator
@time_decorator
def do_export(
    params: ConfigModel, model_path: Union[str, Path], export_type: ExportType
) -> None:
    """
    Export a trained .pt model checkpoint to a VGF file.

    Note: The .pt model type should correspond to an export_type as shown below

    - FP32 -> ExportType.FP32, ExportType.PTQ_INT8
    - QAT -> ExportType.QAT_INT8

    Args:
        params (ConfigModel): Configuration object containing export settings
        model_path (str or Path): Path to the input .pt model file
        export_type (ExportType): The exported model type e.g ExportType.FP32, ExportType.QAT_INT8

    Example:
        >>> params = load_config_file(Path("config.json"))
        >>> do_export(
        ...     params=params,
        ...     model_path="/path/model_to_export.pt",
        ...     export_type=ExportType.FP32,
        ...)
    """

    # Fix for torchao adding BasicHandler if none was present on root
    logging.getLogger().addHandler(logging.NullHandler())
    from ng_model_gym.utils.export_utils import (  # pylint: disable=import-outside-toplevel
        executorch_vgf_export,
    )

    # Create output directory if it doesn't exist.
    create_directory(params.output.dir)
    create_directory(params.output.export.vgf_output_dir)

    model_path = Path(model_path)
    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"Expected a .pt file, got {model_path.name}")

    if export_type in ExportType:
        executorch_vgf_export(params, export_type, model_path)
    else:
        raise ValueError(f"Unsupported export type: {export_type}.")
