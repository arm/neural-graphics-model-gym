# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest
from abc import ABC, abstractmethod
from unittest.mock import patch

import torch

from ng_model_gym.core.model.model import (
    create_model,
    get_model_from_config,
    get_model_key,
)
from ng_model_gym.core.model.model_registry import register_model
from ng_model_gym.core.utils.types import TrainEvalMode
from tests.testing_utils import create_simple_params


# pylint: disable=duplicate-code
class MockNNModule:
    """Mock nn.Module class."""

    def forward(self, *args):
        """Mock forward pass"""
        return args


class MockBaseNGModel(MockNNModule, ABC):
    """Mock BaseNGModel class."""

    @abstractmethod
    def get_neural_network(self):
        """Mock get_neural_network method."""
        raise NotImplementedError

    @abstractmethod
    def set_neural_network(self, nn):
        """Mock set_neural_network method."""
        raise NotImplementedError


# pylint: enable=duplicate-code


class TestModelFactory(unittest.TestCase):
    """Test Model Factory."""

    def setUp(self):
        self.params = create_simple_params()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tearDown(self):
        pass

    def test_get_model_from_config(self):
        """Test getting the model class, based on the config file parameters."""

        for model_name in ("NSS", "nss", "Nss"):
            with self.subTest(model_name=model_name):
                self.params.model.name = model_name

                model_cls = get_model_from_config(self.params)

                self.assertTrue(isinstance(model_cls, type))
                self.assertEqual(model_cls.__name__, "NSSModel")

    @patch("ng_model_gym.core.model.model_registry.nn.Module", new=MockNNModule)
    @patch("ng_model_gym.core.model.model_registry.BaseNGModel", new=MockBaseNGModel)
    def test_get_model_no_version(self):
        """Test registering a model with no version, and getting the model based on the config."""

        # pylint: disable=duplicate-code
        @register_model("MockNSS")
        class MockNSSModel(MockBaseNGModel):  # pylint: disable=unused-variable
            """Mock NSS model"""

            def __init__(self, params):
                self.params = params
                self.nn = MockNNModule()

            def get_neural_network(self):
                return self.nn

            def set_neural_network(self, nn):
                self.nn = nn

            def forward(self, x):
                return x

        # pylint: enable=duplicate-code

        # Override model name and version from params
        self.params.model.name = "MockNSS"
        self.params.model.version = None

        model_cls = get_model_from_config(self.params)

        self.assertEqual(model_cls.__name__, "MockNSSModel")

    def test_error_on_unregistered_model_name(self):
        """Test loading a model class with a name that isn't registered."""

        # Override model name from params
        self.params.model.name = "unregistered_model"

        model_key = get_model_key(self.params.model.name, self.params.model.version)

        with self.assertRaisesRegex(
            KeyError,
            rf"Model {model_key} is not registered",
        ):
            get_model_from_config(self.params)

    def test_error_on_unregistered_model_version(self):
        """Test loading a model class with a version that isn't registered."""

        # Override model name from params
        self.params.model.version = "0"

        model_key = get_model_key(self.params.model.name, self.params.model.version)

        with self.assertRaisesRegex(
            KeyError,
            rf"Model {model_key} is not registered",
        ):
            get_model_from_config(self.params)

    def test_create_model(self):
        """Test initialising the model from the registered class."""

        self.params.model_train_eval_mode = TrainEvalMode.FP32
        self.params.dataset.recurrent_samples = None

        model = create_model(self.params, self.device)

        self.assertIsInstance(model, torch.nn.Module)
        self.assertFalse(model.is_qat_model)

    def test_create_non_feedback_model(self):
        """Test creating a model without recurrent samples doesn't return a feedback model."""

        self.params.model_train_eval_mode = TrainEvalMode.FP32
        self.params.dataset.recurrent_samples = None

        model = create_model(self.params, self.device)

        self.assertNotEqual(type(model).__name__, "FeedbackModel")

    def test_create_feedback_model(self):
        """Test creating a feedback model."""

        self.params.model_train_eval_mode = TrainEvalMode.FP32
        self.params.dataset.recurrent_samples = 4

        model = create_model(self.params, self.device)

        self.assertEqual(type(model).__name__, "FeedbackModel")

    def test_create_qat_model(self):
        """Test initialising the model as a QAT model."""

        self.params.model_train_eval_mode = TrainEvalMode.QAT_INT8
        self.params.dataset.recurrent_samples = None

        model = create_model(self.params, self.device)

        self.assertTrue(model.is_qat_model)


if __name__ == "__main__":
    unittest.main()
