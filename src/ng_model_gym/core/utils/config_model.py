# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import pathlib
from typing import Annotated, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    PositiveInt,
    StrictFloat,
)
from pydantic_core import PydanticCustomError

from ng_model_gym.core.data.utils import ToneMapperMode
from ng_model_gym.core.utils.types import (
    LearningRateScheduler,
    LossFn,
    OptimizerType,
    TrainEvalMode,
)

# pylint: disable=line-too-long

# Pydantic models representing the configuration file structure.
# For fields which are not core to all model types (e.g. recurrent_samples),
# ensure they are marked as Optional with a default of None.


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

    @field_validator("*", mode="after")
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

    train: Optional[pathlib.Path] = Field(
        description="Train dataset directory path", default=None
    )
    validation: Optional[pathlib.Path] = Field(
        description="Validation dataset directory path", default=None
    )
    test: Optional[pathlib.Path] = Field(
        description="Test dataset directory path", default=None
    )


class Processing(PydanticConfigModel):
    """Processing specific configuration"""

    shader_accurate: bool = Field(
        description="Use slang shaders that match deployment shaders"
    )


class Model(PydanticConfigModel):
    """Model configuration"""

    name: str = Field(description="Model name")
    version: Optional[str] = Field(description="Model version", default=None)


class Dataset(PydanticConfigModel):
    """Dataset configuration"""

    name: str = Field(description="Dataset name")
    version: Optional[str] = Field(description="Dataset version", default=None)
    path: Path
    exposure: Optional[float] = Field(
        ge=0.0, description="Training dataset exposure value", default=None
    )
    tonemapper: Optional[ToneMapperMode] = Field(
        description="Tonemapping method for dataset", default=None
    )
    health_check: bool = Field(
        description="Run health check on given dataset. health_check() must be implemented in the Dataset."
    )
    recurrent_samples: Optional[int] = Field(
        gt=1, description="Number of recurrent samples", default=None
    )
    gt_augmentation: bool = Field(
        description="Enable dataset augmentations e.g flips, rotations"
    )
    num_workers: int = Field(ge=0, description="Number of dataloader workers to use")
    prefetch_factor: int = Field(
        ge=0,
        description="Number of batches loaded in advance by each dataloader worker. "
        "Used only if num_workers > 0",
    )
    extension: Optional[str] = Field(
        description="File extension for dataset files to be found (grep) and loaded, e.g. '.safetensors', '.jpg'. Defaults to .safetensors.",
        default=".safetensors",
    )


class Export(PydanticConfigModel):
    """Model export configuration"""

    dynamic_shape: bool = Field(
        description="Enable dynamic input shapes for the exported model "
    )

    vgf_static_input_shape: (
        Annotated[
            list[Annotated[list[PositiveInt], Field(min_length=4, max_length=4)]],
            Field(
                min_length=1,
                description="One or more static VGF input shape definition (4 positive ints each)",
            ),
        ]
        | None
    ) = None

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
        description="Output directory for tensorboard logs. If None is passed, disable tensorboard",
        default=None,
    )
    export: Export


class Checkpoints(PydanticConfigModel):
    """Checkpoints configuration"""

    dir: pathlib.Path = Field(description="Save directory for checkpoints")


class BaseSchedulerConfig(PydanticConfigModel):
    """Base scheduler config with discriminator to signify the type of scheduler"""

    # Act as a discriminator for SchedulerConfig union members
    type: str


class CosineAnnealingConfig(BaseSchedulerConfig):
    """Configuration for the cosine annealing learning rate scheduler"""

    type: Literal[LearningRateScheduler.COSINE_ANNEALING]
    warmup_percentage: float = Field(
        ge=0.0,
        lt=1.0,
        description="Proportion of training steps to linearly warm up the learning rate",
    )
    min_lr: float = Field(
        ge=0.0, le=1.0, description="Minimum learning rate reached after cosine decay."
    )


class ExponentialStepSchedulerConfig(BaseSchedulerConfig):
    """Configuration for the exponential learning rate scheduler"""

    type: Literal[LearningRateScheduler.EXPONENTIAL_STEP]
    decay_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Learning rate decay rate. Decay applied every step_size steps.",
    )
    decay_factor: int = Field(
        ge=1,
        description="Divisor for total number of iterations. The larger the value, the quicker the decay i.e. step_size = (epochs * dataset_size) / decay_factor",
    )


class StaticSchedulerConfig(BaseSchedulerConfig):
    """Configuration for a static learning rate (no scheduler)"""

    type: Literal[LearningRateScheduler.STATIC]


SchedulerConfig = Annotated[
    Union[CosineAnnealingConfig, ExponentialStepSchedulerConfig, StaticSchedulerConfig],
    Field(
        discriminator="type",
        description=f"The Learning Rate Scheduler used during training. Can be one of: {', '.join([e.value for e in LearningRateScheduler])}",
    ),
]


class Optimizer(PydanticConfigModel):
    """Optimizer configuration"""

    optimizer_type: str = Field(
        description="Optimizer type to use during training. Choose from: "
        + ", ".join([e.value for e in OptimizerType]),
    )
    learning_rate: float = Field(
        gt=0.0,
        le=1.0,
        description="The learning rate. Scheduler may override this value",
    )


class TrainingConfig(PydanticConfigModel):
    """Configuration for FP32 or QAT"""

    number_of_epochs: int = Field(ge=1, description="Number of epochs")
    checkpoints: Checkpoints
    lr_scheduler: SchedulerConfig
    optimizer: Optimizer


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
        description="Path to the weights of the pretrained model", default=None
    )
    perform_validate: bool = Field(
        description="Perform validation at the end of specific training epochs, as set by the validate_frequency field."
    )
    validate_frequency: Union[int, List[int]] = Field(
        default=1,
        description="For a single int, N, validate every N epochs. For a List of ints, run validation after each epoch in the List.",
    )
    fp32: TrainingConfig
    qat: TrainingConfig
    loss_fn: Optional[str] = Field(
        default=None,
        description="Loss function to use. Choose from: "
        + ", ".join([e.value for e in LossFn]),
    )


class ConfigModel(PydanticConfigModel):
    """Pydantic model representing configuration file"""

    model: Model
    dataset: Dataset
    output: Output
    train: Train
    processing: Processing
    metrics: Optional[List[str]] = Field(
        default=None,
        description="Metric names to instantiate. Temporal metrics are replaced with streaming variants during evaluation.",
    )
    model_train_eval_mode: Optional[TrainEvalMode] = Field(
        default=None, exclude=True
    )  # Hidden from user. Internal param.
