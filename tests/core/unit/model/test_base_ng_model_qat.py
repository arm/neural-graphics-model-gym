# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torchao.quantization.pt2e import FusedMovingAvgObsFakeQuantize, PlaceholderObserver
from torchao.quantization.pt2e.fake_quantize import FixedQParamsFakeQuantize
from torchao.quantization.pt2e.quantizer import QuantizationSpec

from ng_model_gym.core.model import base_ng_model, BaseNGModel
from ng_model_gym.core.quantization.observers import (
    FixedQParamsFakeQuantizeFix,
    FusedMovingAvgObsFakeQuantizeFix,
)
from ng_model_gym.usecases.nfru.model.nfru_v1 import NFRUv1
from ng_model_gym.usecases.nss.model.model_v1 import NSSV1Model
from tests.testing_utils import create_simple_params

# pylint: disable=missing-function-docstring


class _TestingNeuralNetwork(nn.Module):
    """Small neural network"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        x = self.linear(x)
        return x


class _TestingNGModel(BaseNGModel):
    """BaseNGModel for testing"""

    def __init__(self, params):
        super().__init__(params)
        self.network = _TestingNeuralNetwork()

    def get_neural_network(self) -> nn.Module:
        return self.network

    def set_neural_network(self, neural_network: nn.Module) -> None:
        self.network = neural_network

    def forward(self, x):
        return self.network(x)


class TestBaseNGModelQAT(unittest.TestCase):
    """Test the QAT functionality of BaseNGModel"""

    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cpu")
        self.mock_forward_input = torch.randn((4, 4), device=self.device)
        self.mock_forward_input_trace = (self.mock_forward_input,)
        self.params = create_simple_params(usecase="nss-v1")

    def test_qat_train_raises_if_not_quantized(self):
        """Test model raises error if not prepared with fake quant observers before QAT training"""
        model = _TestingNGModel(self.params)
        model.is_qat_model = True
        with self.assertRaises(RuntimeError):
            model.train(True)

    def test_graphmodule_created_from_quantize_modules(self):
        """Test quantize_modules method successfully creates a GraphModule for QAT"""
        model = _TestingNGModel(self.params)
        model.is_qat_model = True
        model.quantize_modules(self.mock_forward_input_trace)
        self.assertTrue(model.is_network_quantized)
        self.assertTrue(isinstance(model.get_neural_network(), torch.fx.GraphModule))
        out = model(self.mock_forward_input)
        self.assertEqual(out.shape, (4, 4))

    def test_quantize_modules_uses_non_composable_tosa_quantizer(self):
        """Bundled QAT checkpoints require the legacy TOSA graph layout."""
        model = _TestingNGModel(self.params)

        with patch.object(
            base_ng_model,
            "TOSAQuantizer",
            wraps=base_ng_model.TOSAQuantizer,
        ) as quantizer_constructor:
            model.quantize_modules(self.mock_forward_input_trace)

        self.assertFalse(
            quantizer_constructor.call_args.kwargs["use_composable_quantizer"]
        )

    def test_qat_config_derives_int32_conv_bias_for_each_weight_scheme(self):
        """ExecuTorch derives convolution bias quantization from activation and weight."""
        for model_type, usecase, channels, expected_qscheme in (
            (NSSV1Model, "nss-v1", 12, torch.per_channel_symmetric),
            (NFRUv1, "nfru-v1", 16, torch.per_tensor_symmetric),
        ):
            with self.subTest(usecase=usecase):
                model = model_type(create_simple_params(usecase=usecase))
                with patch.object(
                    base_ng_model,
                    "QuantizationConfig",
                    wraps=base_ng_model.QuantizationConfig,
                ) as config_constructor:
                    model.quantize_modules((torch.randn(2, channels, 16, 16),))

                config_call = next(
                    call
                    for call in config_constructor.call_args_list
                    if call.kwargs.get("weight") is not None
                )
                qconfig = base_ng_model.QuantizationConfig(
                    *config_call.args, **config_call.kwargs
                )

                self.assertIsInstance(qconfig.bias, QuantizationSpec)
                self.assertEqual(qconfig.bias.dtype, torch.float)
                self.assertIs(
                    qconfig.bias.observer_or_fake_quant_ctr, PlaceholderObserver
                )

                non_conv_node = SimpleNamespace(target=torch.ops.aten.add.Tensor)
                self.assertIs(qconfig.get_bias_qspec(non_conv_node), qconfig.bias)

                conv_node = SimpleNamespace(
                    target=torch.ops.aten.conv2d.default,
                    args=("activation", "weight"),
                )
                derived_bias = qconfig.get_bias_qspec(conv_node)
                self.assertEqual(derived_bias.dtype, torch.int32)
                self.assertEqual(derived_bias.qscheme, expected_qscheme)
                self.assertEqual(derived_bias.ch_axis, qconfig.weight.ch_axis)

    def test_qparam_policy_selects_fake_quantizer(self):
        """Test NSS uses the fake quantizer fix whilst NFRU uses standard torchAO fake quantizer"""
        for model_type, usecase, channels, expected_fake_quantizer_type in (
            (NSSV1Model, "nss-v1", 12, FusedMovingAvgObsFakeQuantizeFix),
            (NFRUv1, "nfru-v1", 16, FusedMovingAvgObsFakeQuantize),
        ):
            with self.subTest(usecase=usecase):
                model = model_type(create_simple_params(usecase=usecase))
                model.quantize_modules((torch.randn(2, channels, 16, 16),))
                self.assertEqual(
                    {
                        type(module)
                        for module in model.get_neural_network().modules()
                        if isinstance(module, FusedMovingAvgObsFakeQuantize)
                    },
                    {expected_fake_quantizer_type},
                )
                fixed_qparams = [
                    module
                    for module in model.get_neural_network().modules()
                    if isinstance(module, FixedQParamsFakeQuantize)
                ]
                if usecase == "nss-v1":
                    self.assertTrue(
                        any(
                            isinstance(module, FixedQParamsFakeQuantizeFix)
                            for module in fixed_qparams
                        )
                    )
                    self.assertFalse(
                        any(
                            isinstance(module, FixedQParamsFakeQuantize)
                            and not isinstance(module, FixedQParamsFakeQuantizeFix)
                            for module in fixed_qparams
                        )
                    )
                else:
                    self.assertEqual(fixed_qparams, [])

    def test_double_quantize_raises(self):
        """Test if attempting to quantize modules twice raises"""
        model = _TestingNGModel(self.params)
        model.is_qat_model = True
        model.quantize_modules(self.mock_forward_input_trace)
        with self.assertRaises(RuntimeError):
            model.quantize_modules(self.mock_forward_input_trace)

    def test_fp32_train_eval_works(self):
        """Test FP32 train and eval modes haven't been changed by overriding .train() method"""
        fp32_model = _TestingNGModel(self.params)
        self.assertIs(fp32_model.train(), fp32_model)
        self.assertTrue(fp32_model.training)
        self.assertIs(fp32_model.eval(), fp32_model)
        self.assertFalse(fp32_model.training)

    def test_qat_train_eval_updates_training_mode_and_returns_model(self):
        """Test QAT train and eval preserve the PyTorch module lifecycle contract."""
        model = _TestingNGModel(self.params)
        model.is_qat_model = True
        model.quantize_modules(self.mock_forward_input_trace)

        self.assertIs(model.eval(), model)
        self.assertFalse(model.training)
        self.assertIs(model.train(), model)
        self.assertTrue(model.training)
