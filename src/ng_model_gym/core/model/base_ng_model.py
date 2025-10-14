# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
from executorch.backends.arm.quantizer.arm_quantizer import (
    QuantizationConfig,
    TOSAQuantizer,
)
from executorch.backends.arm.tosa.specification import TosaSpecification
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

from ng_model_gym.core.quantization.observers import (
    enable_all_observers,
    freeze_all_observers,
    FusedMovingAvgObsFakeQuantizeFix,
)
from ng_model_gym.core.utils.tensor_types import TensorData
from ng_model_gym.core.utils.types import ExportSpec

logger = logging.getLogger(__name__)


class BaseNGModel(nn.Module, ABC):
    """
    Base class for creating neural-graphics models.

    Subclasses should:
        * Implement getter/setter methods for the core neural network
        * Write the model `forward()` pass
        * [Optional] For recurrent models, override `init_history_buffers`

    Example::

        from torch import nn

        class ExampleNGModel(BaseNGModel):
            def __init__(self):
                super().__init__()
                self.neural_network: nn.Module = ExampleNeuralNetwork()

            def get_neural_network(self) -> nn.Module:
                return self.neural_network

            def set_neural_network(self, neural_network: nn.Module):
                self.neural_network = neural_network

            def forward(self, input_data):
                x = self.preprocessing(input_data)
                x = self.neural_network(x)
                x = self.postprocessing(x)
                return x
    """

    def __init__(self) -> None:
        """Initialise PyTorch nn.Module"""
        super().__init__()
        self.is_qat_model = False
        self.is_network_quantized = False

    @abstractmethod
    def get_neural_network(self) -> nn.Module:
        """
        Get the model's neural network. It should be the same network every call.

        Returns:
            nn.Module: The neural network performing the core forward computation.
        """
        raise NotImplementedError

    @abstractmethod
    def set_neural_network(self, neural_network: nn.Module) -> None:
        """
        Set the model's neural network.
        Args:
            neural_network (nn.Module): Neural network to set.
        """
        raise NotImplementedError

    def train(self: T, mode: bool = True):  # pylint: disable=unused-argument
        """Overrides PyTorch mode hint, model.train()"""

        # Call through PyTorch's default .train() method if doing FP32 training
        if not self.is_qat_model:
            super().train(mode=mode)
            return

        if not self.is_network_quantized:
            raise RuntimeError(
                "Internal quantize_modules method not called on model marked "
                "with is_qat_model=True before model training"
            )
        if mode:
            current_network = self.get_neural_network()
            move_exported_model_to_train(current_network)
            enable_all_observers(current_network)
        else:
            self._eval()

    def _eval(self: T):
        """Quant specific model.eval(), we freeze observers and move to eval mode."""

        if not self.is_network_quantized:
            raise RuntimeError(
                "Something went wrong. "
                "Internal quantize_modules method not called on QAT model before eval"
            )
        current_network = self.get_neural_network()
        freeze_all_observers(current_network)
        move_exported_model_to_eval(current_network)

    def quantize_modules(
        self,
        input_data: Tuple[Any, ...],
        tosa_spec: Optional[ExportSpec] = ExportSpec.TOSA_INT,
    ) -> None:
        """
        Trace neural network and insert FakeQuantizer nodes for QAT.
        Must be done before training starts.
        """

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
        current_network = self.get_neural_network()
        if isinstance(current_network, GraphModule):
            raise RuntimeError(
                "Attempting to quantize network that is already of type GraphModule"
            )

        # Grab the relevant layer to quantize
        quantized_network = trace_and_quantize_module(current_network, input_data)

        self.set_neural_network(quantized_network)

        self.is_network_quantized = True
        logger.info("Model preparations finished for QAT")

    def init_history_buffers(self):
        """Override this method if the model is recurrent"""
        return {}
