# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code
import logging
from typing import Any, Dict, Optional, Tuple

import torch
from executorch.backends.arm.quantizer.arm_quantizer import (
    QuantizationConfig,
    TOSAQuantizer,
)
from executorch.backends.arm.tosa_specification import TosaSpecification
from torch import nn
from torch.fx import GraphModule
from torch.nn.modules.module import T
from torchao.quantization.pt2e import (
    move_exported_model_to_eval,
    move_exported_model_to_train,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e
from torchao.quantization.pt2e.quantizer import (
    FixedQParamsQuantizationSpec,
    QuantizationSpec,
)

from ng_model_gym.core.data.utils import tonemap_forward
from ng_model_gym.core.model.graphics_utils import (
    compute_jitter_tile_offset,
    generate_lr_to_hr_lut,
)
from ng_model_gym.core.quantization.observers import (
    enable_all_observers,
    freeze_all_observers,
    FusedMovingAvgObsFakeQuantizeFix,
)
from ng_model_gym.core.utils.config_model import ConfigModel
from ng_model_gym.core.utils.tensor_types import TensorData
from ng_model_gym.core.utils.types import (
    ExportSpec,
    HistoryBufferResetFunction,
    TrainEvalMode,
)
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


class NSSModel(nn.Module):
    """NSS Model"""

    def __init__(
        self,
        params,
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

        self.history_buffers = self._init_history_buffers()

        self.autoencoder = AutoEncoderV1(feedback_ch=self.feedback_ch, batch_norm=True)

        # Store input tensor to autoencoder for future use when tracing model is required
        self.autoencoder_input_tensor: Optional[torch.Tensor] = None

        self.scale = params.train.scale

        self.dm_scale_on_no_motion = nn.Parameter(
            torch.tensor([0.5]), requires_grad=True
        )

    def _init_history_buffers(self):
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

    def reset_history_buffers(self):
        """Reset history buffers"""
        self.history_buffers = self._init_history_buffers()

    def forward(self, inputs):
        """Forward pass."""
        # 1) PreProcess - Construct Input Tensor
        input_tensor, derivative, depth_dilated = self.preprocess(inputs)

        # 2) Network Dispatch to estimate NSS Params
        kernels, temporal_params, feedback = self.autoencoder(input_tensor)

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

    def get_additional_constants(self):
        """Return additional constants the model learns as a dict."""
        return {
            "dm_scale_on_no_motion": self.dm_scale_on_no_motion.detach()
            .numpy()
            .tolist(),
        }


class QATNSSModel(NSSModel):
    """QAT instrumented NSS model"""

    def __init__(self, params, feedback_ch: Optional[int] = 4):
        super().__init__(params, feedback_ch)
        self.modules_quantized: bool = False

    def train(self: T, mode: bool = True):  # pylint: disable=unused-argument
        """Overrides PyTorch mode hint, model.train()"""

        if not self.modules_quantized:
            raise RuntimeError(
                "Something went wrong. "
                "quantize_modules method not called on QAT model before training"
            )
        if mode:
            move_exported_model_to_train(self.autoencoder)
            enable_all_observers(self.autoencoder)
        else:
            self._eval()

    def _eval(self: T):
        """Quant specific model.eval(), we freeze observers and move to eval mode."""

        if not self.modules_quantized:
            raise RuntimeError(
                "Something went wrong. "
                "quantize_modules method not called on QAT model before eval"
            )
        freeze_all_observers(self.autoencoder)
        move_exported_model_to_eval(self.autoencoder)

    def quantize_modules(
        self,
        input_shape: Tuple[int],
        device: torch.device,
        tosa_spec: Optional[str] = ExportSpec.TOSA_INT,
    ) -> None:
        """Trace module and insert FakeQuantizer nodes. Must be done before training starts"""

        logger.info("Preparing model for QAT")

        # Configure TOSA Quantizer
        quantizer = TOSAQuantizer(TosaSpecification.create_from_string(tosa_spec))

        # Activations get asymmetric per-tensor with moving average of min/max
        extra_args: Dict[str, Any] = {"eps": 2e-12}
        extra_args["observer"] = MovingAverageMinMaxObserver.with_args(
            # PyTorch uses `1e-2` as default which seems a little aggressive
            averaging_constant=1e-5,
            dtype=torch.int8,
            reduce_range=False,
            quant_min=-128,
            quant_max=127,
        )
        qspec = QuantizationSpec(
            dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            qscheme=torch.per_tensor_affine,
            is_dynamic=False,
            observer_or_fake_quant_ctr=FusedMovingAvgObsFakeQuantizeFix.with_args(
                **extra_args,
            ),
        )

        # Weights get symmetric per-channel with true min/max
        extra_args: Dict[str, Any] = {"eps": 2e-12}
        extra_args["observer"] = MovingAveragePerChannelMinMaxObserver.with_args(
            # TODO: work out the correct quantizer to use for this
            # `FakeQuantize` on upstream doesn't work
            # Setting to `1.0` is equivalent to not using a moving average
            averaging_constant=1.0,
            dtype=torch.int8,
            reduce_range=False,
            quant_min=-127,
            quant_max=127,
        )
        weight_quantization_spec = QuantizationSpec(
            dtype=torch.int8,
            quant_min=-127,
            quant_max=127,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
            is_dynamic=False,
            observer_or_fake_quant_ctr=FusedMovingAvgObsFakeQuantizeFix.with_args(
                **extra_args
            ),
        )

        # Bias is assumed to be fine without simulated quantization, because it pre-populates the
        # accumulate register, it's only quantized to int32, with negligible precision drop
        default_qconfig = QuantizationConfig(
            input_activation=qspec,
            output_activation=qspec,
            weight=weight_quantization_spec,
            bias=None,
        )
        quantizer.set_global(quantization_config=default_qconfig)

        # Special case `sigmoid` output nodes, where we use fixed params
        # to ensure table generates full [0, 1] range, also,
        # because we alias this as an SNORM texture to use linear interpolation via HW sampler,
        # we use symmetric [-127, 127] value range, because SNORM disregards -128
        q_min = -127
        q_max = 127
        scale = 1 / (q_max - q_min)
        sigmoid_qspec = FixedQParamsQuantizationSpec(
            dtype=torch.int8,
            scale=scale,
            zero_point=q_min,
            quant_min=q_min,
            quant_max=q_max,
            qscheme=torch.per_tensor_affine,
        )
        sigmoid_qconfig = QuantizationConfig(
            input_activation=default_qconfig.input_activation,
            output_activation=sigmoid_qspec,
            weight=None,
            bias=None,
        )
        quantizer.set_module_type(nn.Sigmoid, sigmoid_qconfig)

        def trace_and_quantize_module(
            module: nn.Module, inputs: Tuple[TensorData]
        ) -> GraphModule:
            # Trace graph
            aten_dialect = torch.export.export_for_training(module, inputs).module()
            # Insert FakeQuantizer nodes
            return prepare_qat_pt2e(aten_dialect, quantizer)

        # Check layer has not already been quantized
        if isinstance(self.autoencoder, GraphModule):
            raise RuntimeError(
                "Trying to quantize module that is already a GraphModule"
            )

        # Grab the relevant layer to quantize
        self.autoencoder = trace_and_quantize_module(
            self.autoencoder, (torch.randn(*input_shape, device=device),)
        )

        self.modules_quantized = True
        logger.info("Model preparations finished for QAT")


def initialize_nss_model_core(params: ConfigModel, device: torch.device) -> NSSModel:
    """Return NSS model v1."""

    match params.model_train_eval_mode:
        case TrainEvalMode.FP32:
            model = NSSModel(params).to(device)

        case TrainEvalMode.QAT_INT8:
            model = QATNSSModel(params).to(device)

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
