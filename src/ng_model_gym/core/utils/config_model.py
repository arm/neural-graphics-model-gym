# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import pathlib
from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    StrictFloat,
)
from pydantic_core import PydanticCustomError

from ng_model_gym.core.data.utils import ToneMapperMode
from ng_model_gym.core.utils.types import LearningRateScheduler, TrainEvalMode

# pylint: disable=line-too-long


class OutputDirModel(BaseModel):
    """Mini model to validate output directory."""

    dir: pathlib.Path


class PydanticConfigModel(BaseModel):
    """Base Pydantic configuration other models should inherit from"""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    @model_validator(mode="before")
    @classmethod
    def _empty_strings_to_none(cls, values: dict):
        """
        Empty strings are validated as None.
        This is to avoid empty string dir/file paths treated as valid inputs
        """
        return {k: None if v == "" else v for k, v in values.items()}

    @field_validator("*", mode="before")
    @classmethod
    def _reject_placeholder_values(cls, field, info):
        """Forbid "<>" placeholders in the config"""
        if isinstance(field, (str, pathlib.Path)):
            s = str(field)
            if "<" in s or ">" in s:
                raise PydanticCustomError(
                    "Placeholder",
                    f"Field `{info.field_name}` contains placeholder {s!r}. Please supply a real value",
                )
        return field


class Path(PydanticConfigModel):
    """Paths to train, test and validation dataset"""

    train: Optional[pathlib.Path] = Field(description="Train dataset directory path")
    validation: Optional[pathlib.Path] = Field(
        description="Validation dataset directory path"
    )
    test: Optional[pathlib.Path] = Field(description="Test dataset directory path")


class Processing(PydanticConfigModel):
    """Processing specific configuration"""

    shader_accurate: bool = Field(
        description="Use slang shaders that match deployment shaders"
    )


class Dataset(PydanticConfigModel):
    """Dataset configuration"""

    path: Path
    exposure: float = Field(ge=0.0, description="Training dataset exposure value")
    tonemapper: ToneMapperMode = Field(description="Tonemapping method for dataset")
    health_check: bool = Field(description="Run health check on given dataset")
    recurrent_samples: int = Field(gt=1, description="Number of recurrent samples")
    gt_augmentation: bool = Field(
        description="Enable dataset augmentations e.g flips, rotations"
    )
    num_workers: int = Field(ge=1, description="Number of dataloader workers to use")
    prefetch_factor: int = Field(
        ge=1,
        description="Number of batches loaded in advance by each dataloader worker",
    )


class Export(PydanticConfigModel):
    """Model export configuration"""

    dynamic_shape: bool = Field(
        description="Enable dynamic input shapes for the exported model "
    )
    vgf_output_dir: pathlib.Path = Field(
        description="Output directory for the VGF file"
    )


class Output(PydanticConfigModel):
    """Training output configuration"""

    dir: pathlib.Path = Field(description="Directory path for storing training output")
    export_frame_png: bool = Field(
        description="Export frames to PNG (for visualization) during model evaluation"
    )
    tensorboard_output_dir: Optional[pathlib.Path] = Field(
        description="Output directory for tensorboard logs. If null is passed, disable tensorboard"
    )
    export: Export


class Checkpoints(PydanticConfigModel):
    """Checkpoints configuration"""

    dir: pathlib.Path = Field(description="Save directory for checkpoints")
    save_frequency: int = Field(
        gt=0, description="How often to save checkpoints during training"
    )


class CosineAnnealingSchedulerConfig(PydanticConfigModel):
    """Cosine annealing config"""

    warmup_percentage: float = Field(
        ge=0.0,
        lt=1.0,
        description="Proportion of training steps to linearly warm up the learning rate",
    )
    min_lr: float = Field(
        ge=0.0, le=1.0, description="Minimum learning rate reached after cosine decay."
    )


class TrainingConfig(PydanticConfigModel):
    """Configuration for FP32 or QAT"""

    number_of_epochs: int = Field(ge=1, description="Number of epochs")
    checkpoints: Checkpoints
    learning_rate: float = Field(
        gt=0.0,
        le=1.0,
        description="The learning rate. Scheduler may override this value",
    )
    cosine_annealing_scheduler_config: CosineAnnealingSchedulerConfig


class Train(PydanticConfigModel):
    """Training configuration"""

    batch_size: int = Field(
        ge=1,
        description="Number of samples processed together in one pass before updating model weights",
    )
    resume: bool = Field(
        description="Resume training from the most recent saved checkpoint"
    )
    scale: StrictFloat = Field(
        2.0,
        ge=2.0,
        le=2.0,
        description="Upscale parameter for the NSS model. Note only 2x for now is supported in this version",
    )
    seed: int = Field(ge=0, description="Seed for random number generation")
    finetune: bool = Field(description="Fine-tune using pretrained_weights")
    pretrained_weights: Optional[pathlib.Path] = Field(
        description="Path to the weights of the pretrained model"
    )
    perform_validate: bool = Field(
        description="Perform validation at the end of each training epoch"
    )
    fp32: TrainingConfig
    qat: TrainingConfig


class ExponentialSchedulerConfig(PydanticConfigModel):
    """Exponential scheduler config"""

    decay: float = Field(ge=0.0, le=1.0, description="Learning rate decay factor")
    decay_step: int = Field(ge=1, description="Steps between learning rate decay")


class Optimizer(PydanticConfigModel):
    """Optimizer configuration"""

    learning_rate_scheduler: LearningRateScheduler = Field(
        description="Learning rate scheduler to use when training"
    )


class ConfigModel(PydanticConfigModel):
    """Pydantic model representing configuration file"""

    version: int = Field(gt=0, description="Model version")
    dataset: Dataset
    output: Output
    train: Train
    optimizer: Optimizer
    processing: Processing
    model_train_eval_mode: Optional[TrainEvalMode] = Field(
        default=None, exclude=True
    )  # Hidden from user. Internal param.
