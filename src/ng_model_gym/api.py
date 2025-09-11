# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Union

import torch
from torch.profiler import schedule

from ng_model_gym.nss.evaluators import ModelEvaluator
from ng_model_gym.trainer import Trainer
from ng_model_gym.utils.checkpoint_utils import load_checkpoint
from ng_model_gym.utils.config_model import ConfigModel
from ng_model_gym.utils.general_utils import create_directory
from ng_model_gym.utils.gpu_log_decorator import gpu_log_decorator
from ng_model_gym.utils.logging import log_gpu_torch
from ng_model_gym.utils.memory_log_decorator import memory_log_decorator
from ng_model_gym.utils.time_decorator import time_decorator
from ng_model_gym.utils.types import ExportType, ProfilerType, TrainEvalMode

logger = logging.getLogger(__name__)


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
