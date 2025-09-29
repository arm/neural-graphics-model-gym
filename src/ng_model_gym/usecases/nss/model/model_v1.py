# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code
import logging
from typing import Optional

import torch
from torch import nn

from ng_model_gym.core.data.utils import tonemap_forward
from ng_model_gym.core.model.base_ng_model import BaseNGModel
from ng_model_gym.core.model.graphics_utils import (
    compute_jitter_tile_offset,
    generate_lr_to_hr_lut,
)
from ng_model_gym.core.model.model_registry import register_model
from ng_model_gym.core.utils.config_model import ConfigModel
from ng_model_gym.core.utils.types import HistoryBufferResetFunction, TrainEvalMode
from ng_model_gym.usecases.nss.history_buffer import HistoryBuffer
from ng_model_gym.usecases.nss.model.model_blocks import AutoEncoderV1
from ng_model_gym.usecases.nss.model.post_processing import (
    PostProcessV1,
    PostProcessV1_ShaderAccurate,
)
from ng_model_gym.usecases.nss.model.pre_processing import (
    PreProcessV1,
    PreProcessV1_ShaderAccurate,
)
from ng_model_gym.usecases.nss.model.recurrent_model import FeedbackModel

logger = logging.getLogger(__name__)


@register_model(name="NSS", version="1")
class NSSModel(BaseNGModel):
    """NSS Model"""

    def __init__(
        self,
        params: ConfigModel,
        feedback_ch: Optional[int] = 4,
    ):
        """Set up the model."""
        super().__init__()

        self.model_name = "nss_v1"
        self.model_params = {
            "gt_augmentation": params.dataset.gt_augmentation,
            "recurrent_samples": params.dataset.recurrent_samples,
        }

        self.shader_accurate = params.processing.shader_accurate

        self.feedback_ch = feedback_ch
        self.tonemapper = params.dataset.tonemapper

        self.autoencoder = AutoEncoderV1(feedback_ch=self.feedback_ch, batch_norm=True)

        # Store input tensor to autoencoder for future use when tracing model is required
        self.autoencoder_input_tensor: Optional[torch.Tensor] = None

        self.scale = params.train.scale

        self.dm_scale_on_no_motion = nn.Parameter(
            torch.tensor([0.5]), requires_grad=True
        )

    def get_neural_network(self) -> nn.Module:
        return self.autoencoder

    def set_neural_network(self, neural_network: nn.Module) -> None:
        self.autoencoder = neural_network

    def forward(self, inputs):
        """Forward pass."""

        input_tensor, derivative, depth_dilated = self.preprocess(inputs)

        kernels, temporal_params, feedback = self.autoencoder(input_tensor)

        outputs = self.postprocess(
            kernels, inputs, temporal_params, depth_dilated, derivative, feedback
        )

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
            "ground_truth": tonemap_forward(
                inputs["ground_truth_linear"] * inputs["exposure"], mode=self.tonemapper
            ),
            "input_color": tonemap_forward(
                inputs["colour_linear"] * inputs["exposure"], mode=self.tonemapper
            ),
        }
        return outputs

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


def initialize_nss_model_core(params: ConfigModel, device: torch.device) -> NSSModel:
    """Return NSS model v1."""

    match params.model_train_eval_mode:
        case TrainEvalMode.FP32:
            model = NSSModel(params).to(device)

        case TrainEvalMode.QAT_INT8:
            model = NSSModel(params).to(device)
            model.is_qat_model = True

        case other:
            raise ValueError(f"Unsupported training mode: {other}")

    return model


def create_feedback_model_with_nss(
    params: ConfigModel, device: torch.device
) -> nn.Module:
    """Creates and returns the complete Feedback model with NSS model"""

    created_feedback_model = FeedbackModel(
        initialize_nss_model_core(params, device),
        recurrent_samples=params.dataset.recurrent_samples,
        device=device,
    )

    return created_feedback_model
