# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
"""NSS v1 model for FP32 training and evaluation."""

import logging
from typing import Dict, Optional

import torch
from torch import nn

from ng_model_gym.core.config.config_model import ConfigModel, NSSModelSettings
from ng_model_gym.core.data.data_utils import tonemap_forward
from ng_model_gym.core.model.base_ng_model import BaseNGModel
from ng_model_gym.core.model.graphics_utils import generate_lr_to_hr_lut
from ng_model_gym.core.model.model_registry import register_model
from ng_model_gym.core.model.shaders.slang_utils import load_slang_module, SlangOutput
from ng_model_gym.core.utils.enum_definitions import TrainEvalMode
from ng_model_gym.usecases.nss.model.model_blocks_v1 import AutoEncoderV1
from ng_model_gym.usecases.nss.model.quality_modes import (
    NSSV1Quality,
    resolve_nss_v1_quality,
)

logger = logging.getLogger(__name__)

_NSS_V1_SHADER_QUALITY_DEFINES = {
    NSSV1Quality.LOW: 0,
    NSSV1Quality.MID: 1,
    NSSV1Quality.HIGH: 2,
}


@register_model(name="NSS", version="1")
class NSSV1Model(BaseNGModel):
    """NSS v1 model for high-quality training and evaluation flows."""

    def __init__(self, params: ConfigModel):
        """Set up the NSS v1 model."""
        super().__init__(params)

        if not isinstance(self.params.model, NSSModelSettings):
            raise TypeError(
                "model section in parameter is not of type NSSModelSettings"
            )

        if self.params.model_train_eval_mode == TrainEvalMode.QAT_INT8:
            raise NotImplementedError(
                "NSS-v1 QAT is planned follow-up work. "
                "The initial NSS-v1 release supports FP32 training/evaluation only."
            )

        self.quality = resolve_nss_v1_quality(self.params.model.quality)
        self.scale = self.params.model.scale
        self.recurrent_samples = self.params.model.recurrent_samples
        self.gt_history_augmentation = bool(self.params.model.gt_history_augmentation)
        self.gt_history_augmentation_chance = float(
            self.params.model.gt_history_augmentation_chance
        )
        self.tonemapper = self.params.dataset.tonemapper
        self.autoencoder = AutoEncoderV1(batch_norm=False, kpn_size=(6, 6))
        self.history_buffers = self.init_history_buffers()

        # NSS v1 training and evaluation configs use shader-accurate NSS with
        # low-resolution motion vectors.
        self.shader_accurate = True
        self.slang_shader_dir = "ng_model_gym.usecases.nss.model.shaders"
        self.slang_shader_file = "nss_v1.slang"
        self.slang: Optional[object] = None

        self.preprocess_half_res_input = False
        self.depth_scatter_quarter_res_input = False
        self.use_sparse_filter_2x2 = False
        self.use_history_catmull = True
        self.packed_nearest_offset_quad = False
        self.required_multiple = (8, 8)
        self.filter_kernel_size = 3
        self.filter_kernel_taps = self.filter_kernel_size * self.filter_kernel_size

    def get_neural_network(self) -> nn.Module:
        """Return the trainable NSS v1 neural network."""
        return self.autoencoder

    def set_neural_network(self, neural_network: nn.Module) -> None:
        """Replace the trainable NSS v1 neural network."""
        self.autoencoder = neural_network

    def validate_export_supported(self, export_type: object) -> None:
        """Raise until NSS v1 export support is implemented."""

        del export_type

        raise NotImplementedError(
            "NSS-v1 export is planned follow-up work. "
            "The initial NSS-v1 release supports FP32 training/evaluation only."
        )

    def define_dynamic_export_model_input(self) -> tuple[object, ...]:
        """Raise until NSS v1 export support is implemented."""

        self.validate_export_supported(export_type=None)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run recurrent NSS v1 high-quality forward pass.

        Input tensors are channel-first recurrent tensors with shape
        ``(N, T, C, H, W)``.
        """

        sequence_length = min(self.recurrent_samples, x["colour_linear"].shape[1])
        if sequence_length < 1:
            raise ValueError("NSS-v1 forward requires at least one recurrent frame.")

        inputs_over_time = self._split_inputs_over_time(
            x, sequence_length=sequence_length
        )
        if not inputs_over_time:
            raise ValueError("NSS-v1 forward requires at least one recurrent frame.")

        outputs: Dict[str, list[torch.Tensor]] = {}
        for t, inputs in enumerate(inputs_over_time):
            inputs = self.set_buffers(inputs)
            inputs = self._maybe_apply_gt_history_augmentation(inputs, time_index=t)
            y_pred = self.core_forward(inputs)
            y_pred.pop("motion", None)
            y_pred.pop("reset_event", None)
            y_pred["reset_event"] = inputs["reset_event"]
            self.update_buffers(inputs, y_pred)

            for key, value in y_pred.items():
                outputs.setdefault(key, []).append(value)

        stacked_outputs = {
            key: torch.stack(value, dim=1) for key, value in outputs.items()
        }
        # TODO: This avoids duplicating the full motion sequence for the loss, but
        # uses the original input motion rather than reset-zeroed per-step motion.
        # That is fine for the current pre-cropped training windows because the loss
        # ignores frame-0 motion and windows should not contain mid-window resets.
        # Longer term, move recurrent loss context out of model outputs and let the
        # trainer pass processed context without stacking a second full sequence.
        stacked_outputs["motion"] = x["motion"]
        return stacked_outputs

    def core_forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run a one-frame high-quality NSS v1 forward pass."""

        self._require_cuda_for_slang_forward(inputs)
        slang = self._get_slang()
        (
            input_shape,
            process_shape,
            hr_shape,
            pad_shape,
            depth_shape,
        ) = self._calculate_dispatch_dims(inputs)
        reset_occurred = 1.0 - (inputs["reset_event"] == 0.0).float()
        device = str(inputs["colour_linear"].device)

        depth_tm1 = slang.depth_scatter(
            in_motion=inputs["motion_lr"],
            in_depth=inputs["depth"],
            in_render_size=inputs["render_size"],
            out_constructors={
                "out_tensor": SlangOutput(
                    init="full",
                    shape=depth_shape,
                    value=torch.iinfo(torch.int32).max,
                    dtype=torch.int32,
                    device=device,
                ),
            },
            dispatch_size=[depth_shape[0], depth_shape[2], depth_shape[3]],
        )

        (
            input_tensor,
            derivative,
            disocclusion_mask,
            nearest_depth_offset,
        ) = slang.pre_process(
            in_colour=inputs["colour_linear"],
            in_history=inputs["history"],
            in_motion=inputs["motion_lr"],
            in_depth=inputs["depth"],
            in_depth_tm1=depth_tm1,
            in_jitter=inputs["jitter"],
            in_jitter_tm1=inputs["jitter_tm1"],
            in_feedback_tm1=inputs["temporal_params_tm1"],
            in_derivative_tm1=inputs["derivative_tm1"],
            in_depth_params=inputs["depth_params"],
            in_exposure=inputs["exposure"],
            in_render_size=inputs["render_size"],
            out_constructors={
                "out_tensor": SlangOutput(
                    shape=pad_shape,
                    channel_dim=12,
                    device=device,
                ),
                "out_luma_derivative": SlangOutput(
                    shape=input_shape,
                    channel_dim=2,
                    device=device,
                ),
                "out_disocclusion_mask": SlangOutput(
                    shape=process_shape,
                    channel_dim=2,
                    device=device,
                ),
                "out_nearest_depth_off": SlangOutput(
                    shape=process_shape,
                    channel_dim=1,
                    device=device,
                ),
            },
            dispatch_size=[pad_shape[0], pad_shape[2], pad_shape[3]],
        )

        kpn_params, temporal_params = self.autoencoder(input_tensor)

        offset_lut, idx_modulo = self._generate_offset_lut(inputs["jitter"])
        output_linear, out_filtered_linear = slang.post_process(
            in_colour=inputs["colour_linear"],
            in_history=inputs["history"],
            in_kpn_params=kpn_params,
            in_temporal_params=temporal_params,
            in_motion=inputs["motion_lr"],
            in_nearest_depth_off=nearest_depth_offset,
            in_exposure=inputs["exposure"],
            in_jitter=inputs["jitter"],
            in_offset_lut=offset_lut,
            in_idx_modulo=idx_modulo,
            in_reset=reset_occurred,
            out_constructors={
                "out_colour": SlangOutput(shape=hr_shape, device=device),
                "out_colour_filtered": SlangOutput(shape=hr_shape, device=device),
            },
            dispatch_size=[hr_shape[0], hr_shape[2], hr_shape[3]],
        )

        output_tm = tonemap_forward(
            output_linear * inputs["exposure"], mode=self.tonemapper
        )
        out_filtered_tm = tonemap_forward(
            out_filtered_linear * inputs["exposure"], mode=self.tonemapper
        )

        return {
            "output": output_tm,
            "output_linear": output_linear,
            "out_filtered": out_filtered_tm,
            "temporal_params": temporal_params,
            "disocclusion_mask": disocclusion_mask,
            "derivative": derivative,
            "ground_truth": tonemap_forward(
                inputs["ground_truth_linear"] * inputs["exposure"],
                mode=self.tonemapper,
            ),
            "input_color": tonemap_forward(
                inputs["colour_linear"] * inputs["exposure"],
                mode=self.tonemapper,
            ),
        }

    def init_history_buffers(self) -> Dict[str, Optional[torch.Tensor]]:
        """Return version-local NSS v1 history buffers."""

        return {
            "history": None,
            "jitter_tm1": None,
            "temporal_params_tm1": None,
            "derivative_tm1": None,
            "reset_event": None,
        }

    def reset_history_buffers(self) -> None:
        """Reset NSS v1 history buffers."""

        self.history_buffers = self.init_history_buffers()

    def set_buffers(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Attach NSS v1 history buffers to one-frame inputs."""

        input_tensors = {
            key: tensor.to(dtype=torch.float32, device=self._model_device())
            for key, tensor in x.items()
        }

        initial_buffers = self._create_zero_history_buffers(input_tensors)
        reset_event = self.history_buffers["reset_event"]
        if reset_event is None:
            for key, value in initial_buffers.items():
                input_tensors[key] = value
            self._zero_reset_motion(
                input_tensors,
                torch.zeros_like(input_tensors["seq"]),
            )
            return input_tensors

        seq = input_tensors["seq"]
        same_sequence = seq == reset_event
        for key, reset_value in initial_buffers.items():
            buffered_value = self.history_buffers[key]
            if buffered_value is None:
                input_tensors[key] = reset_value
            else:
                input_tensors[key] = torch.where(
                    same_sequence,
                    buffered_value,
                    reset_value,
                )
        input_tensors["reset_event"] = torch.where(
            same_sequence,
            reset_event,
            torch.zeros_like(seq),
        )
        self._zero_reset_motion(input_tensors, input_tensors["reset_event"])
        return input_tensors

    def _maybe_apply_gt_history_augmentation(
        self,
        inputs: Dict[str, torch.Tensor],
        *,
        time_index: int,
    ) -> Dict[str, torch.Tensor]:
        """Randomly initialize first-frame history from ground truth in training."""

        if (
            not self.training
            or not self.gt_history_augmentation
            or self.gt_history_augmentation_chance <= 0.0
            or time_index != 0
        ):
            return inputs

        gt_frame = inputs.get("ground_truth_linear")
        if gt_frame is None:
            return inputs

        if gt_frame.ndim != 4:
            return inputs

        gt_frame = gt_frame.to(
            device=inputs["history"].device,
            dtype=inputs["history"].dtype,
        )
        if gt_frame.shape != inputs["history"].shape:
            raise ValueError(
                "GT history augmentation shape mismatch: expected "
                f"{tuple(inputs['history'].shape)}, got {tuple(gt_frame.shape)}."
            )

        threshold = self.gt_history_augmentation_chance / 100.0
        mask = (
            torch.rand(
                (inputs["history"].shape[0], 1, 1, 1),
                device=inputs["history"].device,
            )
            < threshold
        )

        inputs["history"] = torch.where(mask, gt_frame, inputs["history"])
        # TODO: Revisit immediate recurrent-state detaches
        self.history_buffers["history"] = inputs["history"].detach()

        if "reset_event" in inputs:
            reset_mask = mask
            while reset_mask.ndim < inputs["reset_event"].ndim:
                reset_mask = reset_mask.unsqueeze(-1)
            reset_mask = reset_mask.to(device=inputs["reset_event"].device)
            inputs["reset_event"] = torch.where(
                reset_mask,
                torch.ones_like(inputs["reset_event"]),
                inputs["reset_event"],
            )

        return inputs

    def update_buffers(
        self,
        inputs: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
    ) -> None:
        """Update NSS v1 history buffers from a one-frame output."""

        # TODO: Revisit immediate recurrent-state detaches
        self.history_buffers["history"] = outputs["output_linear"].detach()
        self.history_buffers["jitter_tm1"] = inputs["jitter"].detach()
        self.history_buffers["temporal_params_tm1"] = outputs[
            "temporal_params"
        ].detach()
        self.history_buffers["derivative_tm1"] = outputs["derivative"].detach()
        self.history_buffers["reset_event"] = inputs["seq"].detach()

    def detach_buffers(self) -> None:
        """Detach NSS v1 history buffers."""

        for key, value in self.history_buffers.items():
            if value is not None:
                self.history_buffers[key] = value.detach()

    def on_train_epoch_start(self) -> None:
        """Reset history buffers at the start of each epoch."""

        self.reset_history_buffers()

    def on_train_batch_end(self) -> None:
        """Detach history buffers after each training batch."""

        self.detach_buffers()

    def on_train_end(self) -> None:
        """Reset history buffers after training completes."""

        self.reset_history_buffers()

    def on_validation_start(self) -> None:
        """Reset history buffers at the start of validation."""

        self.reset_history_buffers()

    def on_validation_end(self) -> None:
        """Reset history buffers at the end of validation."""

        self.reset_history_buffers()

    def on_evaluation_start(self) -> None:
        """Reset history buffers at the start of evaluation."""

        self.reset_history_buffers()

    def _get_input_data_at_t(
        self, data: Dict[str, torch.Tensor], t: int
    ) -> Dict[str, torch.Tensor]:
        """Retrieve data at time step ``t`` from recurrent tensors."""

        return {key: tensor[:, t, :, :, :] for key, tensor in data.items()}

    def _split_inputs_over_time(
        self, data: Dict[str, torch.Tensor], sequence_length: int
    ) -> tuple[Dict[str, torch.Tensor], ...]:
        """Pre-split recurrent tensors into per-frame input dictionaries."""

        if sequence_length < 1:
            return ()

        frames: list[Dict[str, torch.Tensor]] = [{} for _ in range(sequence_length)]
        for key, tensor in data.items():
            if tensor.ndim >= 5:
                if tensor.shape[1] < sequence_length:
                    raise ValueError(
                        f"Input '{key}' has {tensor.shape[1]} recurrent frames, "
                        f"but {sequence_length} are required."
                    )
                chunks = tensor.split(split_size=1, dim=1)
                for frame, chunk in zip(frames, chunks[:sequence_length]):
                    frame[key] = chunk.squeeze(1)
            else:
                for frame in frames:
                    frame[key] = tensor

        return tuple(frames)

    def _model_device(self) -> torch.device:
        """Return the device that owns the trainable NSS v1 network."""

        return next(self.autoencoder.parameters()).device

    @property
    def device(self) -> torch.device:
        """Return the device that owns the trainable NSS v1 network."""

        return self._model_device()

    def _require_cuda_for_slang_forward(self, inputs: Dict[str, torch.Tensor]) -> None:
        """Raise clearly if the Slang-backed forward path is not on CUDA."""

        if (
            self._model_device().type == "cuda"
            and inputs["colour_linear"].device.type == "cuda"
        ):
            return

        raise RuntimeError("NSS-v1 Slang-backed forward requires CUDA.")

    def _get_slang(self):
        """Lazily load the NSS v1 Slang module."""

        if self.slang is None:
            slang_defines = {
                "NSS_QUALITY": _NSS_V1_SHADER_QUALITY_DEFINES[self.quality],
                "NSS_QUALITY_LOW": _NSS_V1_SHADER_QUALITY_DEFINES[NSSV1Quality.LOW],
                "NSS_QUALITY_MEDIUM": _NSS_V1_SHADER_QUALITY_DEFINES[NSSV1Quality.MID],
                "NSS_QUALITY_HIGH": _NSS_V1_SHADER_QUALITY_DEFINES[NSSV1Quality.HIGH],
                "NSS_PREPROCESS_HALF_RES_INPUT": int(self.preprocess_half_res_input),
                "NSS_DEPTH_SCATTER_QUARTER_RES_INPUT": int(
                    self.depth_scatter_quarter_res_input
                ),
                "NSS_USE_SPARSE_2X2_FILTER": int(self.use_sparse_filter_2x2),
                "NSS_USE_HISTORY_CATMULL": int(self.use_history_catmull),
                "NSS_PACKED_NEAREST_OFFSET_QUAD": int(self.packed_nearest_offset_quad),
                "FILTER_COLOUR_KERNEL_SZ": int(self.filter_kernel_taps),
            }
            if self.shader_accurate or self.preprocess_half_res_input:
                slang_defines["SHADER_ACCURATE"] = True
            self.slang = load_slang_module(
                self.slang_shader_dir,
                self.slang_shader_file,
                autograd=True,
                defines=slang_defines,
            )
        return self.slang

    def _create_zero_history_buffers(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Create reset-state history buffers for high-quality NSS v1."""

        batch = inputs["colour_linear"].shape[0]
        lr_h, lr_w = inputs["colour_linear"].shape[2:]
        device = inputs["colour_linear"].device
        dtype = inputs["colour_linear"].dtype

        return {
            "history": torch.zeros_like(inputs["ground_truth_linear"]),
            "jitter_tm1": torch.zeros_like(inputs["jitter"]),
            "temporal_params_tm1": torch.zeros(
                batch,
                4,
                *self._derive_process_and_padded_spatial(lr_h, lr_w)[2:],
                device=device,
                dtype=dtype,
            ),
            "derivative_tm1": torch.zeros(
                batch,
                2,
                lr_h,
                lr_w,
                device=device,
                dtype=dtype,
            ),
            "reset_event": torch.zeros_like(inputs["seq"]),
        }

    def _zero_reset_motion(
        self,
        inputs: Dict[str, torch.Tensor],
        reset_event: torch.Tensor,
    ) -> None:
        """Zero motion inputs for frames with freshly reset recurrent state."""

        reset_mask = reset_event == 0.0
        for key in ("motion", "motion_lr"):
            if key in inputs:
                inputs[key] = torch.where(
                    reset_mask,
                    torch.zeros_like(inputs[key]),
                    inputs[key],
                )

    def _calculate_dispatch_dims(
        self, inputs: Dict[str, torch.Tensor]
    ) -> tuple[
        tuple[int, int, int, int],
        tuple[int, int, int, int],
        tuple[int, int, int, int],
        tuple[int, int, int, int],
        tuple[int, int, int, int],
    ]:
        """Calculate Slang output and dispatch dimensions."""

        input_shape = tuple(inputs["colour_linear"].shape)
        (
            process_h,
            process_w,
            padded_h,
            padded_w,
        ) = self._derive_process_and_padded_spatial(input_shape[2], input_shape[3])
        hr_shape = (
            input_shape[0],
            input_shape[1],
            int(input_shape[2] * self.scale),
            int(input_shape[3] * self.scale),
        )
        process_shape = (input_shape[0], input_shape[1], process_h, process_w)
        pad_shape = (input_shape[0], input_shape[1], padded_h, padded_w)
        depth_shape = (
            input_shape[0],
            1,
            input_shape[2] // 2,
            input_shape[3] // 2,
        )
        return input_shape, process_shape, hr_shape, pad_shape, depth_shape

    def _calculate_pad(self, size: int, multiple: int) -> int:
        """Return padding required to align ``size`` to ``multiple``."""

        if multiple <= 1:
            return 0
        return (multiple - (size % multiple)) % multiple

    def _derive_process_and_padded_spatial(
        self,
        input_h: int,
        input_w: int,
    ) -> tuple[int, int, int, int]:
        """Return high-quality process and padded spatial dimensions."""

        process_h = input_h // 2 if self.preprocess_half_res_input else input_h
        process_w = input_w // 2 if self.preprocess_half_res_input else input_w
        padded_h = process_h + self._calculate_pad(
            process_h,
            self.required_multiple[0],
        )
        padded_w = process_w + self._calculate_pad(
            process_w,
            self.required_multiple[1],
        )
        return process_h, process_w, padded_h, padded_w

    @torch.compiler.disable
    def _generate_offset_lut(
        self, jitter: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate the NSS v1 post-process offset LUT and modulo tensor."""

        base_lut, idx_mod = generate_lr_to_hr_lut(self.scale, jitter)
        idx_mod_pair = idx_mod.repeat(2) if idx_mod.ndim == 0 else idx_mod
        offset_lut = self._compute_lut(base_lut, idx_mod_pair)
        idx_modulo = idx_mod_pair.reshape(1, 1, 1, 2).to(
            dtype=torch.float32,
            device=jitter.device,
        )
        return offset_lut, idx_modulo

    def _compute_lut(
        self,
        base_lut: torch.Tensor,
        idx_mod: torch.Tensor,
    ) -> torch.Tensor:
        """Build the compact KPN sampling LUT used by the v1 post-process shader."""

        idx_mod_h = int(idx_mod[..., 0].item())
        idx_mod_w = int(idx_mod[..., 1].item())
        kernel_window = 6
        taps_needed = self.filter_kernel_taps
        num_offsets = kernel_window * kernel_window
        tile_count = idx_mod_h * idx_mod_w

        device = base_lut.device
        dtype = base_lut.dtype
        window_radius = kernel_window // 2
        start = -window_radius + 1

        dx_vals = torch.arange(
            start,
            start + kernel_window,
            device=device,
            dtype=torch.long,
        )
        dy_vals = torch.arange(
            start,
            start + kernel_window,
            device=device,
            dtype=torch.long,
        )
        dx_offsets = dx_vals.repeat_interleave(kernel_window)
        dy_offsets = dy_vals.repeat(kernel_window)
        tap_linear = torch.arange(num_offsets, device=device, dtype=torch.long)

        dy_offsets_f = dy_offsets.to(dtype)
        dx_offsets_f = dx_offsets.to(dtype)
        distance_squared = dy_offsets_f.pow(2) + dx_offsets_f.pow(2)
        sigma = kernel_window / 3.0
        weights = torch.exp(-0.5 * distance_squared / (sigma * sigma))

        eps = torch.finfo(dtype).eps * 16.0
        order_bias = (num_offsets - tap_linear.to(dtype)) * eps
        weights_with_bias = weights + order_bias

        base_y = torch.arange(idx_mod_h, device=device, dtype=torch.long)
        base_x = torch.arange(idx_mod_w, device=device, dtype=torch.long)
        by_grid, bx_grid = torch.meshgrid(base_y, base_x, indexing="ij")
        by_flat = by_grid.reshape(-1)
        bx_flat = bx_grid.reshape(-1)

        tap_y = (by_flat.unsqueeze(1) + dy_offsets.unsqueeze(0)) % idx_mod_h
        tap_x = (bx_flat.unsqueeze(1) + dx_offsets.unsqueeze(0)) % idx_mod_w
        tap_linear_idx = tap_y * idx_mod_w + tap_x

        dy_map = base_lut[:, 0].reshape(base_lut.shape[0], -1)
        dx_map = base_lut[:, 1].reshape(base_lut.shape[0], -1)
        valid_map = base_lut[:, 2].reshape(base_lut.shape[0], -1)

        tap_linear_idx_exp = tap_linear_idx.unsqueeze(0).expand(
            base_lut.shape[0],
            -1,
            -1,
        )

        def gather_from_map(map_flat: torch.Tensor) -> torch.Tensor:
            return torch.gather(
                map_flat.unsqueeze(1).expand(-1, tile_count, -1),
                2,
                tap_linear_idx_exp,
            )

        dy_vals_map = gather_from_map(dy_map)
        dx_vals_map = gather_from_map(dx_map)
        valid_vals = gather_from_map(valid_map)

        weights_expanded = weights_with_bias.view(1, 1, num_offsets)
        score = weights_expanded * valid_vals
        _, topk_idx = torch.topk(score, k=taps_needed, dim=-1)

        def gather_candidates(tensor: torch.Tensor) -> torch.Tensor:
            return torch.gather(tensor, -1, topk_idx)

        dy_selected = gather_candidates(dy_vals_map)
        dx_selected = gather_candidates(dx_vals_map)
        valid_selected = gather_candidates(valid_vals)
        tap_dy_selected = gather_candidates(
            dy_offsets_f.view(1, 1, -1).expand(
                base_lut.shape[0],
                tile_count,
                -1,
            )
        )
        tap_dx_selected = gather_candidates(
            dx_offsets_f.view(1, 1, -1).expand(
                base_lut.shape[0],
                tile_count,
                -1,
            )
        )
        tap_linear_selected = gather_candidates(
            tap_linear.view(1, 1, -1).expand(
                base_lut.shape[0],
                tile_count,
                -1,
            )
        ).to(dtype)

        mask = (valid_selected > 0).to(dtype)
        out = torch.zeros(
            (base_lut.shape[0], 6, tile_count, taps_needed),
            dtype=dtype,
            device=device,
        )
        out[:, 0] = dy_selected * mask
        out[:, 1] = dx_selected * mask
        out[:, 2] = mask
        out[:, 3] = tap_dy_selected * mask
        out[:, 4] = tap_dx_selected * mask
        out[:, 5] = tap_linear_selected * mask
        return out
