# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import Any, Tuple

import torch
from torch import nn

from ng_model_gym.core.model.base_ng_model import BaseNGModel
from ng_model_gym.core.model.model_tracer import model_tracer
from tests.testing_utils import create_simple_params

# pylint: disable=unsubscriptable-object


class TestNN(nn.Module):
    """Test NN"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Mock forward pass"""
        return x


class TestNGModel(BaseNGModel):
    """Test NGModel"""

    def __init__(self, params):
        super().__init__(params)
        self.network = TestNN()
        self.trigger_bad_preprocessing = False

    def get_neural_network(self) -> nn.Module:
        """Mock get_neural_network"""
        return self.network

    def set_neural_network(self, neural_network: nn.Module) -> None:
        """Mock set_neural_network"""
        self.network = neural_network

    def mock_preprocess(self, x):
        """Mock preprocess"""
        return None if self.trigger_bad_preprocessing else x

    def forward(self, x):
        """Mock forward pass"""
        x = self.mock_preprocess(x)
        return self.network(x)


class TestModelTracer(unittest.TestCase):
    """Test model tracer"""

    def setUp(self):
        self.params = create_simple_params()

    def test_tracer_captures_input_tensor(self):
        """Test tracer captures single input tensor"""
        model = TestNGModel(self.params)
        t1 = torch.randn(2, 4)

        traced_data: Tuple[Any, ...] = model_tracer(model, t1)
        self.assertIsInstance(traced_data, tuple)
        self.assertEqual(len(traced_data), 1)

        traced_tensor = traced_data[0]
        self.assertIsInstance(traced_tensor, torch.Tensor)
        self.assertTrue(traced_tensor.is_cuda, "traced_tensor should be CUDA")

        torch.testing.assert_close(traced_tensor.cpu(), t1.cpu())
        self.assertEqual(traced_tensor.dtype, t1.dtype)
        self.assertEqual(tuple(traced_tensor.shape), tuple(t1.shape))

    def test_tracer_captures_dict_tensor_input(self):
        """Test tracer captures Dict[str, torch.Tensor]"""
        model = TestNGModel(self.params)

        t1 = torch.randn(2, 4)
        t2 = torch.randn(3, 4)
        input_data = {"a": t1, "b": t2}

        traced_data = model_tracer(model, input_data)
        self.assertIsInstance(traced_data, tuple)
        self.assertEqual(len(traced_data), 1)
        traced_data = traced_data[0]

        self.assertIsInstance(traced_data, dict)
        self.assertEqual(set(traced_data.keys()), {"a", "b"})

        for key, input_tensor in input_data.items():
            traced_tensor = traced_data[key]
            self.assertIsInstance(traced_tensor, torch.Tensor, f"{key} not a tensor")
            self.assertTrue(traced_tensor.is_cuda, f"{key} should be CUDA")
            self.assertEqual(
                traced_tensor.dtype, input_tensor.dtype, f"{key} dtype mismatch"
            )
            self.assertEqual(
                tuple(traced_tensor.shape),
                tuple(input_tensor.shape),
                f"{key} shape mismatch",
            )
            torch.testing.assert_close(
                traced_tensor.cpu(), input_tensor.cpu(), msg=f"{key} value mismatch"
            )

    def test_tracer_raise_missing_input_data(self):
        """Test ValueError is raised if tracer input data is None"""
        model = TestNGModel(self.params)
        model.trigger_bad_preprocessing = True

        invalid_input_data = None

        with self.assertRaises(ValueError):
            model_tracer(model, invalid_input_data)

    def test_tracer_raise_missing_bad_forward_input(self):
        """Test ValueError is raised if model forward input is somehow None"""
        model = TestNGModel(self.params)
        model.trigger_bad_preprocessing = True
        t1 = torch.randn(2, 4)

        with self.assertRaises(ValueError):
            model_tracer(model, t1)
