# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

"""Colour-pipeline builders for NFRU preprocessing and augmentation."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import torch

from ng_model_gym.usecases.nfru.utils.autoexposure import KeyValueAE
from ng_model_gym.usecases.nfru.utils.colour_grading import _COLOUR_GRADING_REGISTRY
from ng_model_gym.usecases.nfru.utils.constants import (
    _HALF_FLOAT_MAX,
    _RANDOM_CONTRAST_STRENGTH_RANGE,
    _RANDOM_SATURATION_STRENGTH_RANGE,
    _RANDOM_TEMPERATURE_RANGE,
    _RANDOM_TINT_RANGE,
)
from ng_model_gym.usecases.nfru.utils.tonemapping import _TONEMAP_REGISTRY

# Stage lookup spans both tonemapping and post-tonemap grading operators.
_COMBINED_REGISTRY: Dict[str, Callable] = {
    **_TONEMAP_REGISTRY,
    **_COLOUR_GRADING_REGISTRY,
}


class ColourPipeline:
    """Apply exposure and ordered colour stages to linear RGB inputs.

    Exposure can be supplied in three ways:
    - ``"auto"`` uses key-value auto-exposure.
    - A numeric value applies a fixed exposure to every sample.
    - ``None`` falls back to per-frame ``exposure_{time_index}`` tensors in ``x_in``.
    """

    def __init__(
        self,
        pipeline_stages: List[Dict[str, Any]],
        exposure: float | str | None,
        auto_exposure_key_value: float,
        auto_exposure_variance: Optional[Dict[str, float]] = None,
    ):
        self.pipeline_stages = pipeline_stages
        self.exposure = None if isinstance(exposure, str) else exposure
        self.use_auto_exposure = (
            isinstance(exposure, str) and exposure.lower() == "auto"
        )
        self.auto_exposure_key_value = auto_exposure_key_value
        self.auto_exposure_variance = auto_exposure_variance

    def _parse_exposure(
        self, x_in: Dict[str, torch.Tensor], time_index: str, rgb_linear: torch.Tensor
    ) -> torch.Tensor:
        """Resolve exposure from auto, fixed, or per-frame dataset metadata."""
        if self.use_auto_exposure:
            key_value = self.auto_exposure_key_value
            if (
                self.auto_exposure_variance is not None
                and time_index in self.auto_exposure_variance
            ):
                key_value *= self.auto_exposure_variance[time_index]
            return KeyValueAE(rgb_linear, key_value=key_value)

        if self.exposure is not None:
            return torch.tensor(
                self.exposure, device=rgb_linear.device, dtype=rgb_linear.dtype
            )

        exposure = x_in.get(f"exposure_{time_index}")
        if exposure is None:
            raise ValueError(f"'exposure_{time_index}' not found in x_in")
        return exposure.to(device=rgb_linear.device, dtype=rgb_linear.dtype)

    def __call__(
        self,
        rgb_linear: torch.Tensor,
        x_in: Dict[str, torch.Tensor],
        time_index: str = "",
    ) -> torch.Tensor:
        """Run the configured exposure step and stage sequence for one frame."""
        exposure = self._parse_exposure(x_in, time_index, rgb_linear)
        rgb = rgb_linear * torch.exp(exposure)
        rgb = torch.clamp(rgb, min=0.0, max=_HALF_FLOAT_MAX)

        for stage in self.pipeline_stages:
            rgb = stage["op"](rgb, x_in=x_in, time_index=time_index, **stage["params"])

        return rgb


class RandomEffectsPipeline(ColourPipeline):
    """Pipeline variant that resamples stage groups and numeric ranges per batch.

    Each pipeline group contributes exactly one stage when ``resample_effects`` is
    called. Exposure and selected stage parameters may also be drawn from ranges.
    """

    PARAM_RANGES = {
        "contrast": {"strength": _RANDOM_CONTRAST_STRENGTH_RANGE},
        "saturation": {"strength": _RANDOM_SATURATION_STRENGTH_RANGE},
        "temperature_tint": {
            "temperature": _RANDOM_TEMPERATURE_RANGE,
            "tint": _RANDOM_TINT_RANGE,
        },
    }

    def __init__(
        self,
        pipeline_stage_groups: List[List[Dict[str, Any]]],
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.pipeline_stage_groups = pipeline_stage_groups
        self.param_ranges = {**self.PARAM_RANGES}
        self.auto_exposure_key_value_range = self._normalise_range(
            self.auto_exposure_key_value
        )
        self.exposure_range = self._normalise_range(self.exposure)
        self.resample_effects()

    def resample_effects(self) -> None:
        """Resample one stage per group and refresh any ranged exposure values."""
        stages = []
        for group in self.pipeline_stage_groups:
            stage = self._sample_from_group(group)
            stages.append(
                {
                    "name": stage["name"],
                    "op": stage["op"],
                    "params": self._randomize_params(stage["name"], stage["params"]),
                }
            )

        self.pipeline_stages = stages

        if self.auto_exposure_key_value_range:
            self.auto_exposure_key_value = self._sample_from_range(
                self.auto_exposure_key_value_range
            )
        elif self.exposure_range:
            self.exposure = self._sample_from_range(self.exposure_range)

    def _randomize_params(
        self, stage_name: str, stage_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fill in any omitted randomisable parameters for the chosen stage."""
        if stage_name not in self.param_ranges:
            return dict(stage_params)

        randomized = dict(stage_params)
        for param_name, bounds in self.param_ranges[stage_name].items():
            if param_name not in randomized:
                randomized[param_name] = self._sample_from_range(bounds)
        return randomized

    @staticmethod
    def _sample_from_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pick one stage config from a mutually exclusive stage group."""
        if len(group) == 1:
            return group[0]
        idx = torch.randint(0, len(group), (1,)).item()
        return group[idx]

    @staticmethod
    def _normalise_range(value: Any) -> Optional[tuple[float, float]]:
        """Convert two-item list or tuple configs into float bounds."""
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (float(value[0]), float(value[1]))
        return None

    @staticmethod
    def _sample_from_range(bounds: tuple[float, float]) -> float:
        """Sample uniformly between the provided float bounds."""
        low, high = bounds
        return torch.rand(1).item() * (high - low) + low


def _get_stage_operator(stage_name: str) -> Callable:
    """Resolve a configured stage name to its registered callable."""
    if stage_name in _COMBINED_REGISTRY:
        return _COMBINED_REGISTRY[stage_name]
    raise KeyError(
        f"Unknown pipeline stage '{stage_name}'. "
        f"Available operators: {sorted(_COMBINED_REGISTRY.keys())}"
    )


def _parse_single_stage(stage_config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Parse one stage entry from string or single-key dict config syntax."""
    if isinstance(stage_config, str):
        stage_name = stage_config.lower()
        stage_params = {}
    elif isinstance(stage_config, dict):
        if len(stage_config) != 1:
            raise ValueError(
                f"Each pipeline stage dict must have exactly one key, got: {stage_config}"
            )
        stage_name = list(stage_config.keys())[0].lower()
        stage_params_raw = stage_config[stage_name] or {}
        stage_params = {k.lower(): v for k, v in stage_params_raw.items()}
    else:
        raise ValueError(f"Invalid pipeline stage config: {stage_config}")

    return {
        "name": stage_name,
        "op": _get_stage_operator(stage_name),
        "params": stage_params,
    }


def _parse_pipeline(
    pipeline_list: List[Union[str, Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Parse a deterministic pipeline where each item becomes one fixed stage."""
    return [_parse_single_stage(stage_config) for stage_config in pipeline_list]


def _parse_random_pipeline(
    pipeline_list: List[Union[str, Dict[str, Any], List[Any]]]
) -> List[List[Dict[str, Any]]]:
    """Parse a pipeline where nested lists represent mutually exclusive stage groups."""
    stage_groups = []
    for item in pipeline_list:
        if isinstance(item, list):
            stage_groups.append([_parse_single_stage(stage) for stage in item])
        else:
            stage_groups.append([_parse_single_stage(item)])
    return stage_groups


def build_colour_pipeline(colour_config: Any) -> ColourPipeline:
    """Build a colour pipeline from config.

    Expected config keys:
    - ``pipeline``: ordered stage list, where nested lists request random choice.
    - ``exposure``: fixed scalar, ``"auto"``, or a two-item range.
    - ``auto_exposure_key_value`` / ``auto_exposure_variance``: controls for
      key-value auto exposure.

    Returns ``RandomEffectsPipeline`` when the config contains random stage groups
    or an exposure range; otherwise returns ``ColourPipeline``.
    """
    exposure = colour_config["exposure"]
    auto_exposure_key_value = colour_config["auto_exposure_key_value"]
    auto_exposure_variance = colour_config["auto_exposure_variance"]

    pipeline_cfg = colour_config.get("pipeline", [])
    # Nested stage groups or ranged exposure opt into per-batch resampling.
    randomisation_requested = any(isinstance(item, list) for item in pipeline_cfg) or (
        isinstance(exposure, (list, tuple)) and len(exposure) == 2
    )

    if randomisation_requested:
        return RandomEffectsPipeline(
            pipeline_stage_groups=_parse_random_pipeline(pipeline_cfg),
            pipeline_stages=[],
            exposure=exposure,
            auto_exposure_key_value=auto_exposure_key_value,
            auto_exposure_variance=auto_exposure_variance,
        )

    return ColourPipeline(
        pipeline_stages=_parse_pipeline(pipeline_cfg),
        exposure=exposure,
        auto_exposure_key_value=auto_exposure_key_value,
        auto_exposure_variance=auto_exposure_variance,
    )
