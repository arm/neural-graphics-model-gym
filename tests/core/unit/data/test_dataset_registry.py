# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import patch

from ng_model_gym.core.data import (
    _validate_dataset,
    dataset_registry,
    get_dataset_key,
    register_dataset,
)
from ng_model_gym.core.utils.registry import Registry


class MockDataset:
    """Mock Dataset class."""

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TestDatasetRegistry(unittest.TestCase):
    """Test Dataset Registry helper and validator."""

    def setUp(self):
        """Add patches for mock classes and test registry."""

        self.patches = [
            patch(
                "ng_model_gym.core.data.dataset_registry.DATASET_REGISTRY",
                new=Registry("TestDataset", validator=_validate_dataset),
            ),
            patch("ng_model_gym.core.data.dataset_registry.Dataset", new=MockDataset),
        ]

        for p in self.patches:
            p.start()

    def tearDown(self):
        """Stop patches."""

        for p in self.patches:
            p.stop()

    def test_get_dataset_key(self):
        """Test dataset key returned matches expected format."""

        self.assertEqual(get_dataset_key(name="nss"), "nss")
        self.assertEqual(get_dataset_key(name="nss", version="1"), "nss-v1")
        self.assertEqual(get_dataset_key(name="NSS", version="1"), "nss-v1")

    def test_dataset_registry_helper_func(self):
        """Test adding a valid dataset using the register_dataset() helper function."""

        dataset_name = "NSS"
        dataset_version = "1"

        @register_dataset(dataset_name, dataset_version)
        class NSSDataset(MockDataset):  # pylint: disable=unused-variable
            """NSS Dataset"""

            def __init__(self):
                pass

            def __len__(self):
                return 0

            def __getitem__(self, index):
                return None

        self.assertEqual(
            dataset_registry.DATASET_REGISTRY.list_registered(),
            [get_dataset_key(dataset_name, dataset_version)],
        )

    def test_dataset_inheritance(self):
        """Test dataset not inheriting from correct base class causes TypeError."""

        dataset_name = "NSS"
        dataset_version = "2"

        with self.assertRaisesRegex(
            TypeError, r"must inherit from torch\.utils\.data\.Dataset"
        ):

            @register_dataset(name=dataset_name, version=dataset_version)
            class NSSDataset:  # pylint: disable=unused-variable
                """NSS dataset"""

                def __init__(self, params):
                    self.params = params

        self.assertNotEqual(
            dataset_registry.DATASET_REGISTRY.list_registered(),
            [get_dataset_key(dataset_name, dataset_version)],
        )

    def test_dataset_len_not_implemented(self):
        """Test not overriding __len__() causes TypeError."""

        dataset_name = "NSS"
        dataset_version = "3"

        with self.assertRaisesRegex(TypeError, r"must override __len__\(self\)"):

            @register_dataset(dataset_name, dataset_version)
            class NSSDataset(
                MockDataset
            ):  # pylint: disable=unused-variable, abstract-method
                """NSS Dataset"""

                def __init__(self):
                    pass

                def __getitem__(self, index):
                    return None

        self.assertNotEqual(
            dataset_registry.DATASET_REGISTRY.list_registered(),
            [get_dataset_key(dataset_name, dataset_version)],
        )

    def test_dataset_getitem_not_implemented(self):
        """Test not overriding __getitem__() causes TypeError."""

        dataset_name = "NSS"
        dataset_version = "4"

        with self.assertRaisesRegex(
            TypeError, r"must override __getitem__\(self, index\)"
        ):

            @register_dataset(dataset_name, dataset_version)
            class NSSDataset(
                MockDataset
            ):  # pylint: disable=unused-variable, abstract-method
                """NSS Dataset"""

                def __init__(self):
                    pass

                def __len__(self):
                    return 0

        self.assertNotEqual(
            dataset_registry.DATASET_REGISTRY.list_registered(),
            [get_dataset_key(dataset_name, dataset_version)],
        )

    def test_dataset_not_a_class(self):
        """Test registering an object that isn't a class raises a TypeError."""

        dataset_name = "NSS"
        dataset_version = "5"

        with self.assertRaisesRegex(TypeError, "must be a class"):

            @register_dataset(dataset_name, dataset_version)
            def NSSDataset():
                pass

        self.assertNotEqual(
            dataset_registry.DATASET_REGISTRY.list_registered(),
            [get_dataset_key(dataset_name, dataset_version)],
        )
