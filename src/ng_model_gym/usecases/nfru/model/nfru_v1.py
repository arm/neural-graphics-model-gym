# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn

from ng_model_gym.core.config.config_model import ConfigModel, NFRUV1ModelSettings
from ng_model_gym.core.model.base_ng_model import BaseNGModel, QATQuantizationProfile
from ng_model_gym.core.model.graphics_utils import normalize_mvs
from ng_model_gym.core.model.model_registry import register_model
from ng_model_gym.core.model.shaders.slang_utils import load_slang_module, SlangOutput
from ng_model_gym.usecases.nfru.model.constants import (
    _DEFAULT_SCALE_FACTOR,
    _FLOW_DOWNSAMPLE_SCALE,
    _FLOW_RESIZE_FACTOR,
    _NEAREST_INTERPOLATION,
    _NFRU_AUTOENCODER_INPUT_CHANNELS,
    _NFRU_DEPTH_PARAM_CHANNELS,
    _QAT_FAKE_QUANT_EPS,
    _RANDOM_SEED_MAX,
)
from ng_model_gym.usecases.nfru.model.nfru_v1_nn import NFRUAutoEncoder
from ng_model_gym.usecases.nfru.model.optical_flow.blockmatch_v321 import (
    BlockMatchV321,
    upscale_and_dilate_flow,
)
from ng_model_gym.usecases.nfru.utils.color_pipeline import build_color_pipeline
from ng_model_gym.usecases.nfru.utils.constants import (
    _BITS_EXP,
    _BITS_X,
    _BITS_Y,
    _MAX_VAL,
)
from ng_model_gym.usecases.nfru.utils.down_sampling_2d import DownSampling2D
from ng_model_gym.usecases.nfru.utils.up_sampling_2d import UpSampling2D

logger = logging.getLogger(__name__)

_SHADER_DIR = "ng_model_gym.usecases.nfru.model.shaders"
_SHADER_FILE = "nfru_v1_sa.slang"
_FLOW_METHOD = "blockmatch_v321"
_REQUIRED_COLOR_SPLITS = ("train", "validation", "test")
_MV_SIMILARITY_THRESHOLD_DYNAMIC_MASK = 0.01
_MV_SIMILARITY_THRESHOLD_DYNAMIC_MASK_RUNTIME_ACCURATE = 0.3
_MV_SIMILARITY_NOISE_THRESHOLD_DYNAMIC_MASK = 0.001
_MV_SIMILARITY_NOISE_THRESHOLD_DYNAMIC_MASK_RUNTIME_ACCURATE = 1.0


def _get_color_config(params: ConfigModel) -> dict[str, dict]:
    color_preprocessing = getattr(params.dataset, "color_preprocessing", None)
    if color_preprocessing is None:
        raise ValueError(
            "NFRU requires dataset.color_preprocessing.train, "
            "dataset.color_preprocessing.validation, and "
            "dataset.color_preprocessing.test. "
        )

    if hasattr(color_preprocessing, "model_dump"):
        color_preprocessing = color_preprocessing.model_dump(mode="json")

    if not isinstance(color_preprocessing, dict) or not color_preprocessing:
        raise ValueError(
            "NFRU requires dataset.color_preprocessing.train, "
            "dataset.color_preprocessing.validation, and "
            "dataset.color_preprocessing.test. "
        )

    missing_or_invalid_splits = tuple(
        split
        for split in _REQUIRED_COLOR_SPLITS
        if not isinstance(color_preprocessing.get(split), dict)
    )
    if missing_or_invalid_splits:
        missing_split_names = ", ".join(missing_or_invalid_splits)
        raise ValueError(
            "NFRU dataset.color_preprocessing must define object configurations for "
            "train, validation, and test. Missing or invalid splits: "
            f"{missing_split_names}."
        )

    return {split: dict(color_preprocessing[split]) for split in _REQUIRED_COLOR_SPLITS}


@register_model(name="NFRU-v1")
class NFRUv1(BaseNGModel):
    """NFRU v1 exposed through the BaseNGModel interface."""

    def __init__(self, params: ConfigModel):
        super().__init__(params)
        if not isinstance(self.params.model, NFRUV1ModelSettings):
            raise TypeError(
                "model section in parameter is not of type NFRUV1ModelSettings"
            )
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        quant_params = {
            "max_val": _MAX_VAL,
            "bits_exp": _BITS_EXP,
            "bits_x": _BITS_X,
            "bits_y": _BITS_Y,
        }
        color_config = _get_color_config(params)

        self.network = NFRUv1Core(
            color_config=color_config,
            quant_params=quant_params.copy(),
            device=self.device,
            scale_factor=self.params.model.scale_factor,
            dynamic_mask_is_runtime_accurate=(
                self.params.model.dynamic_mask_is_runtime_accurate
            ),
            mv_similarity_threshold=self.params.model.mv_similarity_threshold,
        )
        self.quant_params = self.network.quant_params

    def get_neural_network(self) -> nn.Module:
        return self.network.auto_encoder

    def set_neural_network(self, neural_network: nn.Module) -> None:
        self.network.auto_encoder = neural_network

    def get_qat_quantization_profile(self) -> QATQuantizationProfile:
        return QATQuantizationProfile(
            per_channel_weight_quantization=False,
            use_global_quantization_config=False,
            use_fixed_sigmoid_output_qparams=False,
            activation_fake_quant_eps=_QAT_FAKE_QUANT_EPS,
            weight_fake_quant_eps=_QAT_FAKE_QUANT_EPS,
            preserve_qat_qparams=False,
        )

    def on_after_batch_transfer(
        self, batch: tuple[Dict[str, torch.Tensor], torch.Tensor]
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        inputs_dataset, ground_truth_data = batch
        color_pipeline = getattr(self.network, "color_pipeline", None)
        if color_pipeline is None:
            logger.warning(
                "Color pipeline not configured. Returning unprocessed ground truth."
            )
            return inputs_dataset, ground_truth_data

        if self.training and hasattr(color_pipeline, "resample_effects"):
            color_pipeline.resample_effects()

        colored = color_pipeline(ground_truth_data, inputs_dataset, time_index="m1")
        if not isinstance(colored, torch.Tensor):
            colored = torch.from_numpy(colored)

        return inputs_dataset, colored.to(
            device=ground_truth_data.device, dtype=torch.float32
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass of the NFRU core."""
        return self.network(inputs)

    def on_train_epoch_start(self) -> None:
        """Select the training color-preprocessing pipeline."""
        self.network.set_color_pipeline("train")

    def on_validation_start(self) -> None:
        """Select the validation color-preprocessing pipeline."""
        self.network.set_color_pipeline("validation")

    def on_evaluation_start(self) -> None:
        """Select the evaluation/test color-preprocessing pipeline."""
        self.network.set_color_pipeline("test")

    def define_dynamic_export_model_input(self) -> tuple[Dict[int, Any]]:
        """Describe the dynamic batch and spatial dimensions for export."""
        batch_size = torch.export.Dim("batch", min=1)
        input_height_over_2 = torch.export.Dim("input_height_over_2", min=1)
        input_width_over_2 = torch.export.Dim("input_width_over_2", min=1)
        input_height = 2 * input_height_over_2
        input_width = 2 * input_width_over_2
        # Single tensor input to the autoencoder: one tuple entry per positional arg.
        dynamic_shape = ({0: batch_size, 2: input_height, 3: input_width},)

        return dynamic_shape


class NFRUv1Core(nn.Module):
    """Shader-accurate NFRU v1 wrapper around the unchanged autoencoder."""

    def __init__(
        self,
        color_config: Dict[str, Dict],
        quant_params: Optional[Dict[str, int]] = None,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        scale_factor: int = _DEFAULT_SCALE_FACTOR,
        dynamic_mask_is_runtime_accurate: bool = False,
        mv_similarity_threshold: Optional[float] = None,
    ):
        """If mv_similarity_threshold isn't supplied, a default will be used."""
        super().__init__()
        self.device = device
        self.slang: Optional[object] = None
        self.flow_method = _FLOW_METHOD
        self.dynamic_flow = True
        self.of_540 = False
        self.dynamic_mask_is_runtime_accurate = dynamic_mask_is_runtime_accurate
        self.mv_similarity_threshold = self._get_mv_similarity_threshold(
            mv_similarity_threshold, dynamic_mask_is_runtime_accurate
        )
        self.mv_similarity_noise_threshold = (
            self._get_default_mv_similarity_noise_threshold(
                dynamic_mask_is_runtime_accurate
            )
        )
        self.scale_factor = scale_factor

        self.in_ch = _NFRU_AUTOENCODER_INPUT_CHANNELS
        self.auto_encoder = NFRUAutoEncoder()
        self.quant_params = quant_params.copy() if quant_params else {}

        self.available_color_pipeline = {
            split: build_color_pipeline(color_config[split])
            for split in _REQUIRED_COLOR_SPLITS
        }
        self.set_color_pipeline("train")
        self._validate_scale_factor()

        self.dynamic_flow_model = BlockMatchV321()
        self.flow_downsampler = DownSampling2D()
        self.flow_upsampler = UpSampling2D(
            size=_FLOW_RESIZE_FACTOR, interpolation=_NEAREST_INTERPOLATION
        )
        self.coeff_softmax = nn.Softmax(dim=1)

    def _get_slang(self):
        """Lazily load the NFRU v1 Slang module."""
        if self.slang is None:
            self.slang = load_slang_module(
                _SHADER_DIR,
                _SHADER_FILE,
                autograd=True,
            )
        return self.slang

    def set_color_pipeline(self, split: str) -> None:
        """Select the configured color pipeline for the requested split."""
        pipeline = self.available_color_pipeline.get(split)
        if pipeline is None:
            available_splits = ", ".join(self.available_color_pipeline)
            raise ValueError(
                f"Color pipeline split '{split}' is not configured. "
                f"Available splits: {available_splits}."
            )
        self.color_pipeline = pipeline

    def _validate_scale_factor(self) -> None:
        """Ensure the runtime scale factor yields at least one interpolation step."""
        if self.scale_factor <= 1:
            raise ValueError(
                "model.scale_factor must be > 1. Values <= 1 yield no interpolation "
                "timesteps."
            )

    def _get_timestep_range(self) -> np.ndarray:
        """Return the timesteps to evaluate between the input frames."""
        self._validate_scale_factor()
        return np.arange(0, 1, 1 / self.scale_factor)[1:]

    def _resolve_flow(
        self,
        inputs: Dict[str, torch.Tensor],
        rgb_m1: torch.Tensor,
        rgb_p1: torch.Tensor,
        depth_m1: torch.Tensor,
    ) -> torch.Tensor:
        flow_key = f"flow_m1_f30_p1@{self.flow_method}"
        flow_xx_f30_xx = inputs.get(flow_key)

        if flow_xx_f30_xx is None:
            with torch.no_grad():
                input_mv = upscale_and_dilate_flow(
                    inputs["sy_m1_f30_p1"], depth_m1, scale=1.0
                ).contiguous()
                flow_result = self.dynamic_flow_model(
                    {"img_t": rgb_p1, "img_tm1": rgb_m1, "input_mv": input_mv}
                )["output"]
                flow_xx_f30_xx = (
                    self.flow_upsampler(flow_result) * _FLOW_RESIZE_FACTOR
                ).contiguous()
            flow_xx_f30_xx = flow_xx_f30_xx.to(rgb_p1.dtype)
            inputs[flow_key] = flow_xx_f30_xx

        if not self.of_540:
            flow_xx_f30_xx = (
                self.flow_downsampler(flow_xx_f30_xx) * _FLOW_DOWNSAMPLE_SCALE
            )
        return flow_xx_f30_xx

    def warp_mv(
        self,
        depth_m1: torch.Tensor,
        depth_p1: torch.Tensor,
        mv_p1_f30_m1: torch.Tensor,
        dynamic_mask: torch.Tensor,
        motion_mat_tm1: torch.Tensor,
        motion_mat_tp1: torch.Tensor,
        scale: float,
        batch: int,
        out_dims: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Warp motion vectors and return the filled flow plus hole masks."""
        slang = self._get_slang()
        out_packed_mv, out_dynamic_mask, out_holes_t, out_holes_tm1 = slang.warp_mv(
            in_depth=depth_p1,
            in_depth_m1=depth_m1,
            in_motion=mv_p1_f30_m1,
            in_dynamic_mask=dynamic_mask,
            in_motion_mat_m1p1=motion_mat_tm1,
            in_motion_mat_p1m1=motion_mat_tp1,
            in_timestep=scale,
            in_dynamic_mask_is_runtime_accurate=self.dynamic_mask_is_runtime_accurate,
            in_mv_similarity_threshold=self.mv_similarity_threshold,
            in_mv_similarity_noise_threshold=self.mv_similarity_noise_threshold,
            out_constructors={
                "out_packed_mv": SlangOutput(
                    shape=(batch, 1, *out_dims),
                    dtype=torch.int32,
                    device=str(depth_m1.device),
                ),
                "out_dynamic_mask": SlangOutput(
                    shape=(batch, 1, *out_dims), device=str(depth_m1.device)
                ),
                "out_holes_t": SlangOutput(
                    shape=(batch, 1, *out_dims), device=str(depth_m1.device)
                ),
                "out_holes_tm1": SlangOutput(
                    shape=(batch, 1, *out_dims), device=str(depth_m1.device)
                ),
            },
            dispatch_size=[batch, *out_dims],
        )
        out_motion = slang.fill_mv(
            in_packed_mv=out_packed_mv,
            out_constructors={
                "out_motion": SlangOutput(
                    shape=(batch, 2, *out_dims), device=str(depth_m1.device)
                ),
            },
            dispatch_size=[batch, *out_dims],
        )
        return out_motion, out_dynamic_mask, out_holes_t, out_holes_tm1

    def warp_flow(
        self,
        depth: torch.Tensor,
        mv: torch.Tensor,
        scale: float,
        batch: int,
        out_dims: list[int],
    ) -> torch.Tensor:
        """Warp the dense flow field to the requested intermediate timestep."""
        slang = self._get_slang()
        out_packed_mv = slang.warp_flow(
            in_depth=depth,
            in_motion=mv,
            in_timestep=scale,
            out_constructors={
                "out_packed_mv": SlangOutput(
                    shape=(batch, 1, *out_dims),
                    dtype=torch.int32,
                    device=str(depth.device),
                ),
            },
            dispatch_size=[batch, *out_dims],
        )
        return slang.fill_mv(
            in_packed_mv=out_packed_mv,
            out_constructors={
                "out_motion": SlangOutput(
                    shape=(batch, 2, *out_dims), device=str(depth.device)
                ),
            },
            dispatch_size=[batch, *out_dims],
        )

    def _pre_process(
        self,
        flow_t_f30_xx: torch.Tensor,
        mv_t_f30_m1: torch.Tensor,
        rgb_m1: torch.Tensor,
        rgb_p1: torch.Tensor,
        depth_m1: torch.Tensor,
        depth_p1: torch.Tensor,
        depth_p1_warp_t: torch.Tensor,
        depth_p1_warp_p1: torch.Tensor,
        motion_mat_m1p1: torch.Tensor,
        motion_mat_p1m1: torch.Tensor,
        depth_params: torch.Tensor,
        timestep: float,
        random_seed: Optional[int] = None,
    ) -> torch.Tensor:
        slang = self._get_slang()
        batch = mv_t_f30_m1.shape[0]
        out_dims = [flow_t_f30_xx.shape[2], flow_t_f30_xx.shape[3]]

        if random_seed is None:
            random_seed = torch.randint(
                0, _RANDOM_SEED_MAX, (1,), device=rgb_m1.device
            ).item()

        return slang.preprocess(
            in_flow_t_f30_xx=flow_t_f30_xx,
            in_mv_t_f30_m1=mv_t_f30_m1,
            in_rgb_m1=rgb_m1,
            in_rgb_p1=rgb_p1,
            in_depth_m1=depth_m1,
            in_depth_p1=depth_p1,
            in_depth_p1_warp_t=depth_p1_warp_t,
            in_depth_p1_warp_p1=depth_p1_warp_p1,
            in_motion_mat_m1p1=motion_mat_m1p1,
            in_motion_mat_p1m1=motion_mat_p1m1,
            in_depth_params=depth_params,
            in_timestep=timestep,
            in_random_seed=random_seed,
            out_constructors={
                "network_in": SlangOutput(
                    shape=(batch, self.in_ch, *out_dims), device=str(rgb_m1.device)
                )
            },
            dispatch_size=[batch, *out_dims],
        )

    def _post_process(
        self,
        flow_t_f30_xx: torch.Tensor,
        mv_t_f30_m1: torch.Tensor,
        rgb_m1: torch.Tensor,
        rgb_p1: torch.Tensor,
        learnt_params: torch.Tensor,
        timestep: float,
    ) -> torch.Tensor:
        slang = self._get_slang()
        output = slang.postprocess(
            in_flow_t_f30_xx=flow_t_f30_xx,
            in_mv_t_f30_m1=mv_t_f30_m1,
            in_rgb_m1=rgb_m1,
            in_rgb_p1=rgb_p1,
            in_params=learnt_params,
            in_timestep=timestep,
            out_constructors={
                "out_color": SlangOutput(shape=rgb_m1.shape, device=str(rgb_m1.device))
            },
            dispatch_size=[rgb_m1.shape[0], rgb_m1.shape[2], rgb_m1.shape[3]],
        )
        return torch.nan_to_num(output)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run the full NFRU interpolation pipeline for all output timesteps."""
        self._validate_scale_factor()
        batch = inputs["rgb_linear_m1"].shape[0]
        motion_mat = inputs["MotionMat"]

        rgb_m1 = self.color_pipeline(inputs["rgb_linear_m1"], inputs, "m1")
        if not isinstance(rgb_m1, torch.Tensor):
            rgb_m1 = torch.from_numpy(rgb_m1)
        rgb_m1 = rgb_m1.to(self.device)

        rgb_p1 = self.color_pipeline(inputs["rgb_linear_p1"], inputs, "p1")
        if not isinstance(rgb_p1, torch.Tensor):
            rgb_p1 = torch.from_numpy(rgb_p1)
        rgb_p1 = rgb_p1.to(self.device)

        depth_m1 = inputs["depth_m1"]
        depth_p1 = inputs["depth_p1"]
        flow_xx_f30_xx = self._resolve_flow(inputs, rgb_m1, rgb_p1, depth_m1)
        mv_p1_f30_m1 = inputs["mv_p1_f30_m1"]
        mv_m1_f30_m3 = inputs["mv_m1_f30_m3"]
        depth_params = inputs["DepthParams_p1"].reshape(
            batch, _NFRU_DEPTH_PARAM_CHANNELS, 1, 1
        )

        motion_mat_m3 = inputs["ViewProj_m3"][:, 0, ...] @ torch.linalg.inv(
            inputs["ViewProj_m1"][:, 0, ...]
        )

        flow_xx_f30_xx = normalize_mvs(flow_xx_f30_xx)
        mv_p1_f30_m1 = normalize_mvs(mv_p1_f30_m1)
        mv_m1_f30_m3 = normalize_mvs(mv_m1_f30_m3)

        dims_540 = [depth_m1.shape[2], depth_m1.shape[3]]
        dims_270 = [flow_xx_f30_xx.shape[2], flow_xx_f30_xx.shape[3]]

        func_previous_dynamic_mask = self._get_previous_dynamic_mask_fn(
            self.dynamic_mask_is_runtime_accurate
        )
        in_dynamic_mask = func_previous_dynamic_mask(
            tv_depth=depth_m1,
            tv_mv_m1_f30_m3=mv_m1_f30_m3,
            tv_motion_mat_tm1=motion_mat_m3,
            mv_similarity_threshold=float(self.mv_similarity_threshold),
            mv_similarity_noise_threshold=float(self.mv_similarity_noise_threshold),
            out_constructors={
                "tv_dynamic_mask_p1": SlangOutput(
                    shape=depth_m1.shape, device=str(depth_m1.device)
                ),
            },
        )

        output_mfg = []
        timestep_range = self._get_timestep_range()

        for timestep in timestep_range:
            mv_t_f30_m1, _, out_holes_t, out_holes_tm1 = self.warp_mv(
                depth_m1,
                depth_p1,
                mv_p1_f30_m1,
                in_dynamic_mask,
                motion_mat[:, 1, :, :],
                motion_mat[:, 0, :, :],
                timestep,
                batch,
                dims_540,
            )
            flow_t_f30_xx = self.warp_flow(
                depth_m1,
                flow_xx_f30_xx,
                1.0 - timestep,
                batch,
                dims_270,
            )

            network_in = self._pre_process(
                flow_t_f30_xx=flow_t_f30_xx,
                mv_t_f30_m1=mv_t_f30_m1,
                rgb_m1=rgb_m1,
                rgb_p1=rgb_p1,
                depth_m1=depth_m1,
                depth_p1=depth_p1,
                depth_p1_warp_t=out_holes_t,
                depth_p1_warp_p1=out_holes_tm1,
                motion_mat_m1p1=motion_mat[:, 1, :, :],
                motion_mat_p1m1=motion_mat[:, 0, :, :],
                depth_params=depth_params,
                timestep=timestep,
            )

            learnt_params = self.auto_encoder(network_in)
            output = self._post_process(
                flow_t_f30_xx=flow_t_f30_xx,
                mv_t_f30_m1=mv_t_f30_m1,
                rgb_m1=rgb_m1,
                rgb_p1=rgb_p1,
                learnt_params=learnt_params,
                timestep=timestep,
            )
            output_mfg.append(output)

        return {
            "output": output,
            "coeffs": self.coeff_softmax(learnt_params),
            "output_mfg": output_mfg,
        }

    def _get_previous_dynamic_mask_fn(self, dynamic_mask_is_runtime_accurate: bool):
        slang = self._get_slang()
        return (
            slang.calculate_previous_dynamic_mask_runtime_accurate
            if dynamic_mask_is_runtime_accurate
            else slang.calculate_previous_dynamic_mask
        )

    @staticmethod
    def _get_mv_similarity_threshold(
        configured_mv_similarity_threshold: float,
        dynamic_mask_is_runtime_accurate: bool,
    ) -> float:
        return (
            configured_mv_similarity_threshold
            if configured_mv_similarity_threshold is not None
            else _MV_SIMILARITY_THRESHOLD_DYNAMIC_MASK_RUNTIME_ACCURATE
            if dynamic_mask_is_runtime_accurate
            else _MV_SIMILARITY_THRESHOLD_DYNAMIC_MASK
        )

    @staticmethod
    def _get_default_mv_similarity_noise_threshold(
        dynamic_mask_is_runtime_accurate: bool,
    ) -> float:
        return (
            _MV_SIMILARITY_NOISE_THRESHOLD_DYNAMIC_MASK_RUNTIME_ACCURATE
            if dynamic_mask_is_runtime_accurate
            else _MV_SIMILARITY_NOISE_THRESHOLD_DYNAMIC_MASK
        )
