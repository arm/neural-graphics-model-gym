# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
from enum import Enum


class ExportType(str, Enum):
    """Enum of export types."""

    # fp32 weights; fp32 expected input type
    FP32 = "fp32"

    # int8 model weights.
    # Run quantization-aware-training/finetuning on a representative dataset.
    QAT_INT8 = "qat_int8"

    # int8 model weights. Run basic post-training quantization.
    PTQ_INT8 = "ptq_int8"


class ProfilerType(str, Enum):
    """
    Available profilers:
    trace - Detailed execution trace of training
    gpu_memory - GPU memory allocation graph
    """

    DISABLED = "disabled"
    TRACE = "trace"
    GPU_MEMORY = "gpu_memory"


class TrainEvalMode(str, Enum):
    """Indicate if training/evaluation is standard FP32 or QAT"""

    FP32 = "fp32"
    QAT_INT8 = "qat_int8"


class LearningRateScheduler(str, Enum):
    """Enum of supported learning rate scheduler types."""

    COSINE_ANNEALING = "cosine_annealing"
    EXPONENTIAL_STEP = "exponential_step"
    STATIC = "static"


class LossFn(str, Enum):
    """Enum of supported loss functions."""

    LOSS_V1 = "loss_v1"


class OptimizerType(str, Enum):
    """Enum of supported optimizers."""

    ADAM_W = "adam_w"
    LARS_ADAM = "lars_adam"


class HistoryBufferResetFunction(str, Enum):
    """Enum of supported history buffer reset functions."""

    IDENTITY = "identity"
    ZEROS = "zeros"
    ONES = "ones"
    RESET_LR = "reset_lr"
    RESET_HR = "reset_hr"


class ExportSpec(str, Enum):
    """Enum of TOSA export specifications."""

    TOSA_INT = "TOSA-1.00+INT"
    TOSA_FP = "TOSA-1.00+FP"


class ModelType(str, Enum):
    """Enum of model file extensions."""

    PT = ".pt"
