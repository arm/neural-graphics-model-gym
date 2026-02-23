# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code
import logging
from typing import Tuple

import torch
from torch import nn, Tensor

from ng_model_gym.core.data.utils import tonemap_forward
from ng_model_gym.core.history_buffer import HistoryBuffer
from ng_model_gym.core.model.base_ng_model import BaseNGModel
from ng_model_gym.core.model.graphics_utils import (
    compute_jitter_tile_offset,
    generate_lr_to_hr_lut,
)
from ng_model_gym.core.model.model_registry import register_model
from ng_model_gym.core.utils.config_model import ConfigModel
from ng_model_gym.core.utils.tensor_types import TensorData
from ng_model_gym.core.utils.types import HistoryBufferResetFunction
from ng_model_gym.usecases.nss.model.model_blocks import AutoEncoderV1
from ng_model_gym.usecases.nss.model.nss_padding_utils import (
    NSSPaddingPolicy,
    Resolution,
)
from ng_model_gym.usecases.nss.model.post_processing import (
    PostProcessV1,
    PostProcessV1_ShaderAccurate,
)
from ng_model_gym.usecases.nss.model.pre_processing import (
    PreProcessV1,
    PreProcessV1_ShaderAccurate,
)

logger = logging.getLogger(__name__)


@register_model(name="NSS", version="1")
class NSSModel(BaseNGModel):
    """NSS Model"""

    def __init__(self, params: ConfigModel):
        """Set up the model."""
        super().__init__(params)

        self.feedback_ch = 4

        self.shader_accurate = self.params.processing.shader_accurate

        self.tonemapper = self.params.dataset.tonemapper

        self.autoencoder = AutoEncoderV1(feedback_ch=self.feedback_ch, batch_norm=True)

        self.scale = self.params.train.scale

        self.dm_scale_on_no_motion = nn.Parameter(
            torch.tensor([0.5]), requires_grad=True
        )

        self.slang_shader_dir = "ng_model_gym.usecases.nss.model.shaders"
        self.slang_shader_file = "nss_v1.slang"

        self.padding_policy: NSSPaddingPolicy | None = None

        self.recurrent_samples = params.dataset.recurrent_samples
        self.unpad = True
        self.history_buffers = self.init_history_buffers()
        self.device = torch.device("cuda")

    def get_neural_network(self) -> nn.Module:
        return self.autoencoder

    def set_neural_network(self, neural_network: nn.Module) -> None:
        self.autoencoder = neural_network

    def forward(self, x):
        """Run forward pass for the NSS model.
        Input is in channel-first format (N, T, C, H, W).
        """

        # Clear padding_policy if it was set previously
        self.padding_policy = None

        # Get Input Data for t=0
        inputs = self._get_input_data_at_t(x, t=0)

        # Initialise History Buffers Prev
        inputs = self.set_buffers(inputs)

        # Run first inference and initialize output `dict`
        y_pred = self.core_forward(inputs)
        outputs = {}

        # Update history buffers and unpad model predictions.
        y_pred_unpadded = self.update_buffers(inputs, y_pred)

        # Turn outputs into a dict containing lists of tensors.
        for key, value in y_pred_unpadded.items():
            outputs[key] = [value]

        # Compute predictions for the sequence.
        for t in range(1, self.recurrent_samples):
            inputs = self._get_input_data_at_t(x, t=t)
            inputs = self.set_buffers(inputs)
            y_pred = self.core_forward(inputs)
            y_pred_unpadded = self.update_buffers(inputs, y_pred)
            for key, value in y_pred_unpadded.items():
                outputs[key].append(value)

        # Convert dict values from a list of (N, C, H, W) tensors into
        # single tensors with time dimension i.e. (N, T, C, H, W).
        for key, value in outputs.items():
            outputs[key] = torch.stack(value, axis=1)

        return outputs

    def core_forward(self, inputs):
        """Autoencoder neural network forward pass"""

        input_tensor, derivative, depth_dilated = self.preprocess(inputs)

        kernels, temporal_params, feedback = self.autoencoder(input_tensor)

        outputs = self.postprocess(
            kernels, inputs, temporal_params, depth_dilated, derivative, feedback
        )
        if isinstance(inputs, dict):
            outputs.setdefault(
                "motion", inputs["motion"]
            )  # Add input motion vector to outputs

        return outputs

    def preprocess(self, preprocess_input):
        """Create the autoencoder input from preprocessing"""

        preprocess_input["dm_scale"] = torch.clamp(
            self.dm_scale_on_no_motion, min=0.0, max=1.0
        )

        preprocess_input = {
            key: tensor.to(dtype=torch.float32, device=torch.device("cuda"))
            for key, tensor in preprocess_input.items()
        }

        if self.shader_accurate:
            # NOTE: depth_dilated here is actually the offset,
            # Maintain same name as before to reuse history buffer
            # from original code, so depth_tm1 ends up becoming offsets
            (
                input_tensor,
                derivative,
                depth_dilated,
            ) = PreProcessV1_ShaderAccurate.apply(
                preprocess_input["colour_linear"],
                preprocess_input["history"],
                preprocess_input["motion_lr"],
                preprocess_input["depth"],
                preprocess_input["true_depth_tm1"],
                preprocess_input["depth_tm1"],  # nearest offset tm1
                preprocess_input["jitter"],
                preprocess_input["jitter_tm1"],
                preprocess_input["feedback_tm1"],
                preprocess_input["derivative_tm1"],
                preprocess_input["depth_params"],
                preprocess_input["exposure"],
                preprocess_input["render_size"],
                preprocess_input["dm_scale"],
                self.slang_shader_dir,
                self.slang_shader_file,
            )

        else:
            input_tensor, derivative, depth_dilated = PreProcessV1.apply(
                preprocess_input["colour_linear"],
                preprocess_input["history"],
                preprocess_input["motion"],
                preprocess_input["depth"],
                preprocess_input["depth_tm1"],
                preprocess_input["jitter"],
                preprocess_input["jitter_tm1"],
                preprocess_input["feedback_tm1"],
                preprocess_input["derivative_tm1"],
                preprocess_input["depth_params"],
                preprocess_input["exposure"],
                preprocess_input["render_size"],
                preprocess_input["dm_scale"],
                self.slang_shader_dir,
                self.slang_shader_file,
            )

        return input_tensor, derivative, depth_dilated

    def postprocess(
        self, kernels, inputs, temporal_params, depth_dilated, derivative, feedback
    ):
        """Postprocess neural network outputs"""

        # For convenience, we'll combine these kernels into the same buffer
        # (in deployment these are kept as separate textures)
        kpn_params = torch.concatenate(kernels, dim=1)

        # Ignore network temporal decisions
        # When a sequence change has occurred -> force history reset
        temporal_params = torch.where(
            inputs["reset_event"] == 0.0,
            torch.zeros_like(temporal_params),
            temporal_params,
        )

        inputs["motion"] = torch.where(
            inputs["reset_event"] == 0.0,
            torch.ones_like(inputs["motion"]) * 20_000.0,
            inputs["motion"],
        )

        # 3) Reconstruct high-res output
        scale_tensor = torch.tensor(
            self.scale, dtype=torch.float32, device=inputs["jitter"].device
        )

        if self.shader_accurate:
            _, idx_modulo = generate_lr_to_hr_lut(self.scale, inputs["jitter"])
            offset_lut = compute_jitter_tile_offset(
                inputs["jitter"], scale_tensor, idx_modulo
            )

            output_linear, out_filtered = PostProcessV1_ShaderAccurate.apply(
                inputs["colour_linear"],
                inputs["history"],
                kpn_params.to(dtype=torch.float32),
                temporal_params.to(dtype=torch.float32),
                inputs["motion_lr"],
                depth_dilated,
                inputs["exposure"],
                inputs["jitter"],
                offset_lut,
                scale_tensor,
                idx_modulo,
                self.slang_shader_dir,
                self.slang_shader_file,
            )

        else:
            offset_lut, idx_modulo = generate_lr_to_hr_lut(self.scale, inputs["jitter"])

            output_linear, out_filtered = PostProcessV1.apply(
                inputs["colour_linear"],
                inputs["history"],
                kpn_params.to(dtype=torch.float32),
                temporal_params.to(dtype=torch.float32),
                inputs["motion"],
                inputs["exposure"],
                inputs["jitter"],
                offset_lut,
                scale_tensor,
                idx_modulo,
                self.slang_shader_dir,
                self.slang_shader_file,
            )

        # TM for loss function / visualisation
        output_tm = tonemap_forward(
            output_linear * inputs["exposure"], mode=self.tonemapper
        )
        out_filtered_tm = tonemap_forward(
            out_filtered * inputs["exposure"], mode=self.tonemapper
        )
        outputs = {
            "output": output_tm,
            "output_linear": output_linear,
            "out_filtered": out_filtered_tm,
            "feedback": feedback,
            "derivative": derivative,
            "depth_dilated": depth_dilated,  # on shader accurate this is actually offsets
        }
        return outputs

    def set_buffers(self, x: TensorData) -> TensorData:
        """Set or retrieve history buffers"""
        input_tensors = {}

        if not self.padding_policy:
            self.padding_policy = self.create_padding_policy(x)

        for key, value in x.items():
            input_tensors[key] = self._pad(value)

        # Add history buffers to input tensors.
        for key, buffer in self.history_buffers.items():
            if not buffer.initialised:
                input_tensors[buffer.name] = buffer.set(
                    input_tensors[buffer.reset_key], x["seq"]
                )
            else:
                input_tensors[buffer.name] = buffer.get(
                    input_tensors[buffer.reset_key], x["seq"]
                )

        return input_tensors

    def update_buffers(
        self, inputs: TensorData, outputs: TensorData
    ) -> dict[list[torch.tensor]]:
        """Update history buffers"""
        for _, buffer in self.history_buffers.items():
            if buffer.update_key in outputs:
                buffer.update(outputs[buffer.update_key])
            elif buffer.update_key in inputs:
                buffer.update(inputs[buffer.update_key])
            else:
                raise KeyError(
                    f"{buffer.update_key} not found in inputs or outputs of network"
                )

        # Now unpad outputs
        final_outputs = {}
        for k, v in outputs.items():
            final_outputs[k] = self._unpad(v)

        return final_outputs

    def _get_pad_sz(
        self, height: int, width: int, is_unpad: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """Return padding size - new resolutions need to be added to table"""

        pad_h, pad_w = self.padding_policy.calculate_padding(
            height,
            width,
            is_unpad=is_unpad,
        )
        return torch.tensor(pad_h), torch.tensor(pad_w)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[2], x.shape[3]
        pad_h, pad_w = self._get_pad_sz(height, width)
        padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return padded

    def _unpad(self, x: torch.Tensor) -> torch.Tensor:
        """Removes Padding"""
        if not self.unpad:
            return x
        height, width = x.shape[2], x.shape[3]
        pad_h, pad_w = self._get_pad_sz(height, width, is_unpad=True)
        return x[..., : height - pad_h, : width - pad_w]

    def _get_input_data_at_t(self, data: TensorData, t: int) -> TensorData:
        """Retrieve data at time step `t` from 5D Recurrent Tensor"""
        input_tensors = {}
        for key in data.keys():
            input_tensors[key] = data[key][:, t, :, :, :]
        return input_tensors

    def define_dynamic_export_model_input(self):
        """
        Description of dynamic dimensions of input tensor

        Note: Dynamic shape constraints e.g. dimension is a multiple of 8 or certain
        range of values is an optional check during export. Exported TOSA/VGF model do not
        capture these constraints in the network graph for the dynamic dimensions. This should be
        handled in the preprocessing stage when running the model.
        """

        # Dynamic batch size - ensure when exporting, this dim in the config is ≥ 2
        batch_size = torch.export.Dim("batch")  # Batch size can be anything

        # Input width/height is a multiple of 8 (because of the resizing layers)
        input_height_over_8 = torch.export.Dim("input_height_over_8", min=1)
        input_width_over_8 = torch.export.Dim("input_width_over_8", min=1)

        # Optional constraints enforced at model export. Can be omitted
        input_height = 8 * input_height_over_8
        input_width = 8 * input_width_over_8

        # NCHW - tuple contents match forward tensor input
        dynamic_shape = ({0: batch_size, 2: input_height, 3: input_width},)

        return dynamic_shape

    def init_history_buffers(self):
        return {
            "history": HistoryBuffer(
                name="history",
                # NOTE: motion is currently expected to be output resolution
                reset_key="motion",
                reset_func=HistoryBufferResetFunction.ZEROS,
                update_key="output_linear",
                channel_dim=3,
            ),
            "depth_tm1": HistoryBuffer(
                name="depth_tm1",
                reset_key="depth",
                reset_func=HistoryBufferResetFunction.ZEROS,
                update_key="depth_dilated",
            ),
            "true_depth_tm1": HistoryBuffer(
                name="true_depth_tm1",
                reset_key="depth",
                update_key="depth",
            ),
            "jitter_tm1": HistoryBuffer(
                name="jitter_tm1", reset_key="jitter", update_key="jitter"
            ),
            "feedback_tm1": HistoryBuffer(
                name="feedback_tm1",
                reset_key="depth",
                reset_func=HistoryBufferResetFunction.ZEROS,
                update_key="feedback",
                channel_dim=self.feedback_ch,
            ),
            "derivative_tm1": HistoryBuffer(
                name="derivative_tm1",
                reset_key="depth",
                reset_func=HistoryBufferResetFunction.ZEROS,
                update_key="derivative",
                channel_dim=2,
            ),
            "reset_event": HistoryBuffer(
                name="reset_event",
                reset_key="seq",
                reset_func=HistoryBufferResetFunction.ZEROS,
                update_key="seq",
            ),
        }

    def get_additional_constants(self):
        """Return additional constants the model learns as a dict."""
        return {
            "dm_scale_on_no_motion": self.dm_scale_on_no_motion.detach()
            .numpy()
            .tolist(),
        }

    def create_padding_policy(self, tensor_data: TensorData):
        """Find the lr and hr resolutions from dataset and create padding policy"""
        tensor_resolutions: set[Tuple[int, int]] = set()

        for tensor in tensor_data.values():
            if not isinstance(tensor, torch.Tensor):
                continue

            if tensor.ndim != 4:
                raise ValueError("Tensor does not have 4 dimensions")

            height, width = tensor.shape[2], tensor.shape[3]

            # Skip scalar tensors
            if height == 1 and width == 1:
                continue

            tensor_resolutions.add((height, width))

        if len(tensor_resolutions) != 2:
            raise ValueError(
                f"Expected the presence of high resolution and low resolution tensors but found {tensor_resolutions} tensors"  # pylint: disable=line-too-long
            )

        # Calculate area to figure out lr and hr values
        lr_h, lr_w = min(tensor_resolutions, key=lambda hw: hw[0] * hw[1])
        hr_h, hr_w = max(tensor_resolutions, key=lambda hw: hw[0] * hw[1])

        # Create named tuples
        lr_res = Resolution(height=lr_h, width=lr_w)
        hr_res = Resolution(height=hr_h, width=hr_w)

        scale = int(self.params.train.scale)
        multiple = 8
        return NSSPaddingPolicy(lr_res, hr_res, multiple, scale)

    def reset_history_buffers(self):
        """Reset history buffers"""
        self.history_buffers = self.init_history_buffers()

    def detach_buffers(self) -> None:
        """Detach history buffers"""
        for _, buffer in self.history_buffers.items():
            buffer.detach()

    def on_train_epoch_start(self) -> None:
        """Reset history buffers at the start of each epoch"""
        self.reset_history_buffers()

    def on_train_batch_end(self) -> None:
        """Detach history buffers after each training batch"""
        self.detach_buffers()

    def on_train_end(self) -> None:
        """Reset history buffers after training completes"""
        self.reset_history_buffers()

    def on_validation_start(self) -> None:
        """Reset history buffers at the start of validation"""
        self.reset_history_buffers()

    def on_validation_end(self) -> None:
        """Reset history buffers at the end of validation"""
        self.reset_history_buffers()

    def on_evaluation_start(self) -> None:
        """Reset history buffers and set recurrence to 1 at the start of eval"""
        logger.debug("Temporarily setting recurrent_samples to 1")
        self.recurrent_samples = 1
        self.reset_history_buffers()
