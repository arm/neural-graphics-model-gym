# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch
from torch import nn

from ng_model_gym.core.model.base_ng_model import BaseNGModel
from tests.testing_utils import create_simple_params

# pylint: disable=missing-function-docstring


class TestingNeuralNetwork(nn.Module):
    """Small neural network"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        x = self.linear(x)
        return x


class TestingNGModel(BaseNGModel):
    """BaseNGModel for testing"""

    def __init__(self, params):
        super().__init__(params)
        self.network = TestingNeuralNetwork()

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
        self.params = create_simple_params()

    def test_qat_train_raises_if_not_quantized(self):
        """Test model raises error if not prepared with fake quant observers before QAT training"""
        model = TestingNGModel(self.params)
        model.is_qat_model = True
        with self.assertRaises(RuntimeError):
            model.train(True)

    def test_graphmodule_created_from_quantize_modules(self):
        """Test quantize_modules method successfully creates a GraphModule for QAT"""
        model = TestingNGModel(self.params)
        model.is_qat_model = True
        model.quantize_modules(self.mock_forward_input_trace)
        self.assertTrue(model.is_network_quantized)
        self.assertTrue(isinstance(model.get_neural_network(), torch.fx.GraphModule))
        out = model(self.mock_forward_input)
        self.assertEqual(out.shape, (4, 4))

    def test_double_quantize_raises(self):
        """Test if attempting to quantize modules twice raises"""
        model = TestingNGModel(self.params)
        model.is_qat_model = True
        model.quantize_modules(self.mock_forward_input_trace)
        with self.assertRaises(RuntimeError):
            model.quantize_modules(self.mock_forward_input_trace)

    def test_fp32_train_eval_works(self):
        """Test FP32 train and eval modes haven't been changed by overriding .train() method"""
        fp32_model = TestingNGModel(self.params)
        fp32_model.train()
        self.assertTrue(fp32_model.training)
        fp32_model.eval()
        self.assertFalse(fp32_model.training)


if __name__ == "__main__":
    unittest.main()
