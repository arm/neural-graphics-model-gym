# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

from ng_model_gym.core.utils.registry import Registry


class MockBaseModel:
    """Mock base model class."""

    def __init__(self):
        super().__init__()

    def get_neural_network(self):
        """Mock get_neural_network method."""

    def set_neural_network(self, nn):
        """Mock set_neural_network method"""


class MockBaseDataset:
    """Mock base dataset class."""

    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        return None

    def __add__(self, other):
        pass


class TestRegistry(unittest.TestCase):
    """Test Registry class."""

    def setUp(self):
        # Create registry without validator
        self.model_registry: Registry[MockBaseModel] = Registry("Model")

    def _register_NSS_model(self, name):
        """Register the NSS_Model in the model registry."""

        @self.model_registry.register(name)
        class NSSModel(MockBaseModel):
            """NSS model"""

            def __init__(self, params):  # pylint: disable=super-init-not-called
                self.params = params
                self.nn = None

            def get_neural_network(self):
                return self.nn

            def set_neural_network(self, nn):
                self.nn = nn

        return NSSModel

    def test_registering_model(self):
        """Test registering a model."""

        model_class = self._register_NSS_model("NSS_v1")

        self.assertIs(self.model_registry.get("NSS_v1"), model_class)

    def test_registering_duplicate_model(self):
        """Test re-registering NSS_V1 model results in an error."""

        self._register_NSS_model("NSS_v1")

        with self.assertRaisesRegex(
            KeyError, r"^'Model NSS_v1 is already registered\.'$"
        ):
            self._register_NSS_model("NSS_v1")

    def test_get_unregistered_model(self):
        """Test getting a model with an unregistered key results in an error."""

        with self.assertRaisesRegex(KeyError, r"^'Model NSS_v1 is not registered\.'$"):
            self.model_registry.get("NSS_v1")

    def test_listing_all_registered_models(self):
        """Test list of keys returned for all registered models."""

        self._register_NSS_model("NSS_v1")
        self._register_NSS_model("NSS_v2")
        self._register_NSS_model("NSS_v3")

        self.assertEqual(
            self.model_registry.list_registered(), ["NSS_v1", "NSS_v2", "NSS_v3"]
        )

    def test_registries_are_isolated(self):
        """Check dataset registry and model registry are isolated from each other."""

        self._register_NSS_model("NSS_v1")

        self.dataset_registry: Registry[MockBaseDataset] = Registry("Dataset")

        @self.dataset_registry.register("NSS_dataset")
        class NSSDataset(MockBaseDataset):  # pylint: disable=unused-variable
            """NSS Dataset"""

            def __init__(self):  # pylint: disable=super-init-not-called
                pass

            def __len__(self):
                return 0

            def __getitem__(self, index):
                return None

        with self.assertRaises(KeyError):
            self.model_registry.get("NSS_dataset")

        with self.assertRaises(KeyError):
            self.dataset_registry.get("NSS_v1")


if __name__ == "__main__":
    unittest.main()
