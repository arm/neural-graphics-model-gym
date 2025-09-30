# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest
from abc import ABC, abstractmethod
from unittest.mock import patch

from ng_model_gym.core.model import model_registry
from ng_model_gym.core.model.model_registry import _validate_model, register_model
from ng_model_gym.core.utils.registry import Registry


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


class TestModelRegistry(unittest.TestCase):
    """Test Model Registry helper and validator."""

    def setUp(self):
        """Add patches for mock classes and test registry."""

        self.patches = [
            patch(
                "ng_model_gym.core.model.model_registry.MODEL_REGISTRY",
                new=Registry("TestModel", validator=_validate_model),
            ),
            patch("ng_model_gym.core.model.model_registry.nn.Module", new=MockNNModule),
            patch(
                "ng_model_gym.core.model.model_registry.BaseNGModel",
                new=MockBaseNGModel,
            ),
        ]

        for p in self.patches:
            p.start()

    def tearDown(self):
        """Stop patches."""

        for p in self.patches:
            p.stop()

    def test_model_registry_helper_func(self):
        """Test adding a valid model using the register_model() helper function."""

        model_name = "NSS"
        model_version = "1"

        @register_model(model_name, model_version)
        class NSSModel(MockBaseNGModel):  # pylint: disable=unused-variable
            """NSS model"""

            def __init__(self, params):
                self.params = params
                self.nn = MockNNModule()

            def get_neural_network(self):
                return self.nn

            def set_neural_network(self, nn):
                self.nn = nn

            def forward(self, x):
                return x

        self.assertEqual(
            model_registry.MODEL_REGISTRY.list_registered(),
            [f"{model_name}-v{model_version}".lower()],
        )

    def test_model_inheritance(self):
        """Test model not inheriting from correct base class causes TypeError."""

        model_name = "NSS"
        model_version = "2"

        with self.assertRaisesRegex(TypeError, "must inherit from BaseNGModel"):

            @register_model(name=model_name, version=model_version)
            class NSSModel:  # pylint: disable=unused-variable
                """NSS model"""

                def __init__(self, params):
                    self.params = params

        self.assertNotEqual(
            model_registry.MODEL_REGISTRY.list_registered(),
            [f"{model_name}-v{model_version}".lower()],
        )

    def test_model_get_neural_network_not_implemented(self):
        """Test not implementing get_neural_network() causes TypeError."""

        model_name = "NSS"
        model_version = "3"

        with self.assertRaisesRegex(
            TypeError,
            r"Make sure all abstract methods, e.g. get_neural_network\(self\), are implemented",
        ):

            @register_model(model_name, model_version)
            class NSSModel(MockBaseNGModel):  # pylint: disable=unused-variable
                """NSS model"""

                def __init__(self, params):
                    self.params = params
                    self.nn = MockNNModule()

                def forward(self, x):
                    return x

                def set_neural_network(self, nn):
                    self.nn = nn

        self.assertNotEqual(
            model_registry.MODEL_REGISTRY.list_registered(),
            [f"{model_name}-v{model_version}".lower()],
        )

    def test_model_forward_not_implemented(self):
        """Test not overriding forward() causes TypeError."""

        model_name = "NSS"
        model_version = "4"

        with self.assertRaisesRegex(TypeError, r"must override forward\(\)"):

            @register_model(model_name, model_version)
            class NSSModel(MockBaseNGModel):  # pylint: disable=unused-variable
                """NSS model"""

                def __init__(self, params):
                    self.params = params
                    self.nn = MockNNModule()

                def get_neural_network(self):
                    return self.nn

                def set_neural_network(self, nn):
                    self.nn = nn

        self.assertNotEqual(
            model_registry.MODEL_REGISTRY.list_registered(),
            [f"{model_name}-v{model_version}".lower()],
        )

    def test_model_not_a_class(self):
        """Test registering an object that isn't a class raises a TypeError."""

        model_name = "NSS"
        model_version = "5"

        with self.assertRaisesRegex(TypeError, "must be a class"):

            @register_model(model_name, model_version)
            def NSSModel():
                pass

        self.assertNotEqual(
            model_registry.MODEL_REGISTRY.list_registered(),
            [f"{model_name}-v{model_version}".lower()],
        )

    def test_model_set_neural_network_not_implemented(self):
        """Test not implementing set_neural_network() causes TypeError."""

        model_name = "NSS"
        model_version = "6"

        with self.assertRaisesRegex(
            TypeError,
            r"Make sure all abstract methods, e.g. get_neural_network\(self\), are implemented",
        ):

            @register_model(model_name, model_version)
            class NSSModel(MockBaseNGModel):  # pylint: disable=unused-variable
                """NSS model"""

                def __init__(self, params):
                    self.params = params
                    self.nn = MockNNModule()

                def forward(self, x):
                    return x

                def get_neural_network(self):
                    return self.nn

        self.assertNotEqual(
            model_registry.MODEL_REGISTRY.list_registered(),
            [f"{model_name}-v{model_version}".lower()],
        )


if __name__ == "__main__":
    unittest.main()
