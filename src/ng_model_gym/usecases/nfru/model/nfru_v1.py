# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn

from ng_model_gym.core.config.config_model import ConfigModel, NFRUModelSettings
from ng_model_gym.core.model.base_ng_model import BaseNGModel, QATQuantizationProfile
from ng_model_gym.core.model.graphics_utils import normalize_mvs
from ng_model_gym.core.model.model_registry import register_model
from ng_model_gym.core.model.shaders.slang_utils import load_slang_module, SlangOutput
from ng_model_gym.usecases.nfru.model.blockmatch_v311 import (
    BlockMatchV311,
    upscale_and_dilate_flow,
)
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
from ng_model_gym.usecases.nfru.model.nfru_v1_ne import NFRUAutoEncoder
from ng_model_gym.usecases.nfru.utils.colour_pipeline import build_colour_pipeline
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
_FLOW_METHOD = "blockmatch_v311"
_REQUIRED_COLOUR_SPLITS = ("train", "validation", "test")

m = load_slang_module(_SHADER_DIR, _SHADER_FILE, autograd=True)


def _get_colour_config(params: ConfigModel) -> dict[str, dict]:
    colour_preprocessing = getattr(params.dataset, "colour_preprocessing", None)
    if colour_preprocessing is None:
        raise ValueError(
            "NFRU requires dataset.colour_preprocessing.train, "
            "dataset.colour_preprocessing.validation, and "
            "dataset.colour_preprocessing.test. "
        )

    if hasattr(colour_preprocessing, "model_dump"):
        colour_preprocessing = colour_preprocessing.model_dump(mode="json")

    if not isinstance(colour_preprocessing, dict) or not colour_preprocessing:
        raise ValueError(
            "NFRU requires dataset.colour_preprocessing.train, "
            "dataset.colour_preprocessing.validation, and "
            "dataset.colour_preprocessing.test. "
        )

    missing_or_invalid_splits = tuple(
        split
        for split in _REQUIRED_COLOUR_SPLITS
        if not isinstance(colour_preprocessing.get(split), dict)
    )
    if missing_or_invalid_splits:
        missing_split_names = ", ".join(missing_or_invalid_splits)
        raise ValueError(
            "NFRU dataset.colour_preprocessing must define object configurations for "
            "train, validation, and test. Missing or invalid splits: "
            f"{missing_split_names}."
        )

    return {
        split: dict(colour_preprocessing[split]) for split in _REQUIRED_COLOUR_SPLITS
    }


@register_model(name="NFRU", version="1")
class NFRUv1(BaseNGModel):
    """NFRU v1 exposed through the BaseNGModel interface."""

    def __init__(self, params: ConfigModel):
        super().__init__(params)
        if not isinstance(self.params.model, NFRUModelSettings):
            raise TypeError(
                "model section in parameter is not of type NFRUModelSettings"
            )
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        quant_params = {
            "max_val": _MAX_VAL,
            "bits_exp": _BITS_EXP,
            "bits_x": _BITS_X,
            "bits_y": _BITS_Y,
        }
        colour_config = _get_colour_config(params)

        self.network = NFRUv1Core(
            colour_config=colour_config,
            quant_params=quant_params.copy(),
            device=self.device,
            scale_factor=self.params.model.scale_factor,
            shader_accurate=self.params.processing.shader_accurate,
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
        )

    def on_after_batch_transfer(
        self, batch: tuple[Dict[str, torch.Tensor], torch.Tensor]
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        inputs_dataset, ground_truth_data = batch
        colour_pipeline = getattr(self.network, "colour_pipeline", None)
        if colour_pipeline is None:
            logger.warning(
                "Colour pipeline not configured. Returning unprocessed ground truth."
            )
            return inputs_dataset, ground_truth_data

        if self.training and hasattr(colour_pipeline, "resample_effects"):
            colour_pipeline.resample_effects()

        coloured = colour_pipeline(ground_truth_data, inputs_dataset, time_index="m1")
        if not isinstance(coloured, torch.Tensor):
            coloured = torch.from_numpy(coloured)

        return inputs_dataset, coloured.to(
            device=ground_truth_data.device, dtype=torch.float32
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass of the NFRU core."""
        return self.network(inputs)

    def on_train_epoch_start(self) -> None:
        """Select the training colour-preprocessing pipeline."""
        self.network.set_colour_pipeline("train")

    def on_validation_start(self) -> None:
        """Select the validation colour-preprocessing pipeline."""
        self.network.set_colour_pipeline("validation")

    def on_evaluation_start(self) -> None:
        """Select the evaluation/test colour-preprocessing pipeline."""
        self.network.set_colour_pipeline("test")

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
        colour_config: Dict[str, Dict],
        quant_params: Optional[Dict[str, int]] = None,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        scale_factor: int = _DEFAULT_SCALE_FACTOR,
        shader_accurate: bool = False,
    ):
        super().__init__()
        self.device = device
        self.flow_method = _FLOW_METHOD
        self.dynamic_flow = True
        self.of_540 = False
        self.new_dynamic_mask = False
        self.shader_accurate = shader_accurate
        self.scale_factor = scale_factor

        self.in_ch = _NFRU_AUTOENCODER_INPUT_CHANNELS
        self.auto_encoder = NFRUAutoEncoder()
        self.quant_params = quant_params.copy() if quant_params else {}

        self.available_colour_pipeline = {
            split: build_colour_pipeline(colour_config[split])
            for split in _REQUIRED_COLOUR_SPLITS
        }
        self.set_colour_pipeline("train")
        self._validate_scale_factor()

        self.dynamic_flow_model = BlockMatchV311()
        self.flow_downsampler = DownSampling2D()
        self.flow_upsampler = UpSampling2D(
            size=_FLOW_RESIZE_FACTOR, interpolation=_NEAREST_INTERPOLATION
        )
        self.coeff_softmax = nn.Softmax(dim=1)

    def set_colour_pipeline(self, split: str) -> None:
        """Select the configured colour pipeline for the requested split."""
        pipeline = self.available_colour_pipeline.get(split)
        if pipeline is None:
            available_splits = ", ".join(self.available_colour_pipeline)
            raise ValueError(
                f"Colour pipeline split '{split}' is not configured. "
                f"Available splits: {available_splits}."
            )
        self.colour_pipeline = pipeline

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
                flow_result = self.dynamic_flow_model(rgb_p1, rgb_m1, input_mv)
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
        out_packed_mv, out_dynamic_mask, out_holes_t, out_holes_tm1 = m.warp_mv(
            in_depth=depth_p1,
            in_depth_m1=depth_m1,
            in_motion=mv_p1_f30_m1,
            in_dynamic_mask=dynamic_mask,
            in_motion_mat_m1p1=motion_mat_tm1,
            in_motion_mat_p1m1=motion_mat_tp1,
            in_timestep=scale,
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
        out_motion = m.fill_mv(
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
        out_packed_mv = m.warp_flow(
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
        return m.fill_mv(
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
        batch = mv_t_f30_m1.shape[0]
        out_dims = [flow_t_f30_xx.shape[2], flow_t_f30_xx.shape[3]]

        if random_seed is None:
            random_seed = torch.randint(
                0, _RANDOM_SEED_MAX, (1,), device=rgb_m1.device
            ).item()

        return m.preprocess(
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
        output = m.postprocess(
            in_flow_t_f30_xx=flow_t_f30_xx,
            in_mv_t_f30_m1=mv_t_f30_m1,
            in_rgb_m1=rgb_m1,
            in_rgb_p1=rgb_p1,
            in_params=learnt_params,
            in_timestep=timestep,
            out_constructors={
                "out_colour": SlangOutput(shape=rgb_m1.shape, device=str(rgb_m1.device))
            },
            dispatch_size=[rgb_m1.shape[0], rgb_m1.shape[2], rgb_m1.shape[3]],
        )
        return torch.nan_to_num(output)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run the full NFRU interpolation pipeline for all output timesteps."""
        self._validate_scale_factor()
        batch = inputs["rgb_linear_m1"].shape[0]
        motion_mat = inputs["MotionMat"]

        rgb_m1 = self.colour_pipeline(inputs["rgb_linear_m1"], inputs, "m1")
        if not isinstance(rgb_m1, torch.Tensor):
            rgb_m1 = torch.from_numpy(rgb_m1)
        rgb_m1 = rgb_m1.to(self.device)

        rgb_p1 = self.colour_pipeline(inputs["rgb_linear_p1"], inputs, "p1")
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

        func_dynamic_mask = self._get_dynamic_mask_fn()
        in_dynamic_mask = func_dynamic_mask(
            tv_depth=depth_m1,
            tv_mv_m1_f30_m3=mv_m1_f30_m3,
            tv_motion_mat_tm1=motion_mat_m3,
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

    def _get_dynamic_mask_fn(self):
        return (
            m.calculate_previous_dynamic_mask_v5
            if self.new_dynamic_mask
            else m.calculate_previous_dynamic_mask
        )
