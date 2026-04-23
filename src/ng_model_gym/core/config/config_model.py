# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import pathlib
import tempfile
from numbers import Real
from typing import Annotated, Any, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    PositiveInt,
    StrictFloat,
)
from pydantic.json_schema import SkipJsonSchema
from pydantic_core import PydanticCustomError

from ng_model_gym.core.utils.enum_definitions import (
    LearningRateScheduler,
    LossFn,
    OptimizerType,
    ToneMapperMode,
    TrainEvalMode,
)

# pylint: disable=line-too-long

CONFIG_SCHEMA_VERSION = "5"

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


class Paths(PydanticConfigModel):
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


ColourPipelineStage = Union[str, dict[str, Any]]
ColourPipelineGroup = list[ColourPipelineStage]
ColourPipelineConfig = list[Union[ColourPipelineStage, ColourPipelineGroup]]
ColourExposureRange = Annotated[list[float], Field(min_length=2, max_length=2)]
ColourExposureConfig = Union[float, Literal["auto"], ColourExposureRange]


class ColourPreprocessingSplitConfig(PydanticConfigModel):
    """Per-split colour preprocessing config using lowercase user-facing keys."""

    pipeline: ColourPipelineConfig = Field(
        default_factory=list,
        description=(
            "Ordered colour stages. Nested lists indicate mutually exclusive stages "
            "to sample from during random-effects training."
        ),
    )
    exposure: ColourExposureConfig = Field(
        default=2.0,
        description='Fixed exposure, "auto", or a two-item range for resampling.',
    )
    auto_exposure_key_value: float = Field(
        default=1.0,
        description="Key value used when exposure is set to auto.",
    )
    auto_exposure_variance: Optional[dict[str, float]] = Field(
        default=None,
        description="Optional per-time-index multiplier applied to auto exposure.",
    )


class ColourPreprocessingConfig(PydanticConfigModel):
    """Per-split colour preprocessing for NFRU datasets."""

    train: Optional[ColourPreprocessingSplitConfig] = Field(
        default=None, description="Colour preprocessing applied to training data."
    )
    validation: Optional[ColourPreprocessingSplitConfig] = Field(
        default=None, description="Colour preprocessing applied to validation data."
    )
    test: Optional[ColourPreprocessingSplitConfig] = Field(
        default=None, description="Colour preprocessing applied to test data."
    )


class Processing(PydanticConfigModel):
    """Processing specific configuration"""

    shader_accurate: bool = Field(
        description="Use slang shaders that match deployment shaders"
    )


class MetricsConfig(PydanticConfigModel):
    """Metrics configuration for train/val/test splits."""

    train: Optional[List[str]] = Field(
        default=None, description="Metric names to instantiate during training."
    )
    val: Optional[List[str]] = Field(
        default=None, description="Metric names to instantiate during validation."
    )
    test: Optional[List[str]] = Field(
        default=None, description="Metric names to instantiate during evaluation/test."
    )


class BaseModelSettings(PydanticConfigModel):
    """Model configuration"""

    name: str = Field(description="Model name")
    version: Optional[str] = Field(description="Model version", default=None)


class PrebuiltModelSettingsBase(BaseModelSettings):
    """
    Our own models we provide should extend this class and fill out the prebuilt_models_settings
    discriminated union
    """

    model_source: Literal["prebuilt"]


class CustomModelSettings(BaseModelSettings):
    """
    If a user specifies the model_source to be custom,
    allow any json fields into the model section of the config
    """

    model_source: Literal["custom"]
    model_config = ConfigDict(extra="allow", revalidate_instances="always")


class NSSModelSettings(PrebuiltModelSettingsBase):
    """NSS model settings"""

    name: Literal["nss"]
    scale: StrictFloat = Field(
        2.0,
        ge=2.0,
        le=2.0,
        description="Upscale parameter for the NSS model. Note, for now only 2x is supported in this version",
    )
    recurrent_samples: int = Field(gt=1, description="Number of recurrent samples")


class NFRUModelSettings(PrebuiltModelSettingsBase):
    """NFRU model settings"""

    name: Literal["nfru"]
    scale_factor: StrictFloat = Field(
        default=2.0,
        ge=2.0,
        le=2.0,
        description="Interpolation scale factor for the NFRU model. Only 2 (or 2.0) is currently supported.",
    )
    legacy_nfru_capture_paths: List[str] = Field(
        default_factory=list,
        description="Path substrings identifying legacy NFRU captures that should use the old window stride behavior.",
    )

    @field_validator("scale_factor", mode="before")
    @classmethod
    def _validate_nfru_scale_factor(cls, value):
        """Require NFRU scale_factor to be the numeric value 2 or 2.0."""
        if isinstance(value, bool) or not isinstance(value, Real):
            raise PydanticCustomError(
                "InvalidScaleFactorType",
                "model.scale_factor must be provided as the numeric value 2 or 2.0.",
            )

        if float(value) != 2.0:
            raise PydanticCustomError(
                "InvalidScaleFactor",
                "model.scale_factor must be 2 or 2.0. NFRU currently only supports 2x interpolation.",
            )

        return 2.0


# DO NOT FORGET TO ADD NEW MODEL SETTINGS HERE
prebuilt_models_settings = Annotated[
    Union[NSSModelSettings, NFRUModelSettings],
    Field(discriminator="name"),
]


ModelSettings = Annotated[
    Union[prebuilt_models_settings, CustomModelSettings],
    Field(discriminator="model_source"),
]


class Dataset(PydanticConfigModel):
    """Dataset configuration"""

    name: str = Field(description="Dataset name")
    version: Optional[str] = Field(description="Dataset version", default=None)
    path: Paths
    colour_preprocessing: Optional[ColourPreprocessingConfig] = Field(
        default=None,
        description=(
            "Required lowercase colour pipeline configuration for train, "
            "validation, and test."
        ),
    )
    exposure: Optional[float] = Field(
        ge=0.0, description="Training dataset exposure value", default=None
    )
    tonemapper: Optional[ToneMapperMode] = Field(
        description="Tonemapping method for dataset", default=None
    )
    health_check: bool = Field(
        description="Run health check on given dataset. health_check() must be implemented in the Dataset."
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
    align_data: bool = Field(
        default=True,
        description="Whether to align loaded dataset features to the model input set.",
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

    @field_validator("vgf_output_dir", mode="after")
    @classmethod
    def _validate_vgf_output_dir(cls, value: pathlib.Path) -> pathlib.Path:
        """Disallow temp paths for vgf output dir"""
        temp_root = pathlib.Path(tempfile.gettempdir()).resolve(strict=False)
        path = value.expanduser().resolve(strict=False)

        try:
            path.relative_to(temp_root)
        except ValueError:
            return path  # Success, path is not in a tmp dir

        # If path.relative_to() did not raise an error, that means path is in /tmp folder
        raise PydanticCustomError(
            "TempDir",
            "vgf_output_dir must not be under the system temp directory. "
            "Choose a path to a persistent directory ",
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
    eps: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Optional epsilon value for optimizers that support it.",
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
    resume: SkipJsonSchema[Optional[pathlib.Path]] = Field(
        default=None, exclude=True
    )  # Hidden from user
    seed: int = Field(ge=0, description="Seed for random number generation")
    finetune: SkipJsonSchema[str | pathlib.Path | None] = Field(
        default=None,
        exclude=True,
        union_mode="left_to_right",
    )  # Hidden from user
    perform_validate: bool = Field(
        description="Perform validation at the end of specific training epochs, as set by the validate_frequency field."
    )
    validate_frequency: Union[int, List[int]] = Field(
        default=1,
        description=(
            "For a single int, N, validate every N epochs. "
            "For a List of ints, run validation after each epoch in the List. "
            "If using with `perform_validate=true`, ensure that at least one validation pass "
            "will run within the total number of epochs, otherwise a ValueError is raised."
        ),
    )
    fp32: TrainingConfig
    qat: TrainingConfig
    loss_fn: Optional[str] = Field(
        default=None,
        description="Loss function to use. Choose from: "
        + ", ".join([e.value for e in LossFn]),
    )
    loss_args: Optional[dict] = Field(
        default=None,
        description="Optional dictionary of arguments to pass to the loss function",
    )

    @model_validator(mode="after")
    def _validate_validation_schedule(self):
        """
        Warn early if validation is enabled but won't run within the configured epochs.
        """
        if not self.perform_validate:
            return self

        total_epochs = (
            self.fp32.number_of_epochs
        )  # fp32 and qat share validation config shape
        freq = self.validate_frequency

        def will_run(start_epoch: int, epochs: int, frequency):
            if isinstance(frequency, int):
                return frequency > 0 and any(
                    epoch % frequency == 0 for epoch in range(start_epoch, epochs + 1)
                )
            if isinstance(frequency, list):
                return any(
                    epoch in frequency for epoch in range(start_epoch, epochs + 1)
                )
            return False

        if not will_run(1, total_epochs, freq):
            raise ValueError(
                "Validation is enabled (`perform_validate=true`) but no validation pass "
                "will run with the current settings "
                f"(validate_frequency={freq}, number_of_epochs={total_epochs}). "
                "Increase number_of_epochs or adjust validate_frequency."
            )

        return self


class ConfigModel(PydanticConfigModel):
    """Pydantic model representing configuration file"""

    @model_validator(mode="before")
    @classmethod
    def _normalise_model_name(cls, values: dict):
        """Normalise pre-built model names to lowercase."""

        if isinstance(values, dict):
            model = values.get("model")

            if isinstance(model, dict) and model.get("model_source") == "prebuilt":
                name = model.get("name")

                if isinstance(name, str):
                    model["name"] = name.lower()

        return values

    config_schema_version: str = Field(
        CONFIG_SCHEMA_VERSION,
        description="Config schema version. Used to check compatibility.",
    )
    model: ModelSettings
    dataset: Dataset
    output: Output
    train: Train
    processing: Processing
    metrics: Optional[Union[List[str], MetricsConfig]] = Field(
        default=None,
        description=(
            "Metric names to instantiate. Provide a list to use the same metrics for"
            " train/val/test, or an object with train/val/test lists. Temporal metrics"
            " are replaced with streaming variants during evaluation."
        ),
    )
    model_train_eval_mode: SkipJsonSchema[Optional[TrainEvalMode]] = Field(
        default=None, exclude=True
    )  # Hidden from user. Internal param.

    @model_validator(mode="after")
    def _validate_nfru_colour_preprocessing(self):
        """Require explicit colour-preprocessing config for NFRU v1."""
        if getattr(self.model, "name", None) != "nfru":
            return self

        colour_preprocessing = self.dataset.colour_preprocessing
        if colour_preprocessing is None:
            raise PydanticCustomError(
                "NFRUColourPreprocessingRequired",
                "NFRU requires dataset.colour_preprocessing.train, "
                "dataset.colour_preprocessing.validation, and "
                "dataset.colour_preprocessing.test.",
            )

        required_splits = ("train", "validation", "test")
        missing_or_invalid_splits = [
            split
            for split in required_splits
            if getattr(colour_preprocessing, split, None) is None
        ]
        if missing_or_invalid_splits:
            raise PydanticCustomError(
                "NFRUColourPreprocessingMissingSplit",
                "NFRU dataset.colour_preprocessing must define object configurations for "
                "train, validation, and test. Missing or invalid splits: "
                f"{', '.join(missing_or_invalid_splits)}.",
            )

        return self
