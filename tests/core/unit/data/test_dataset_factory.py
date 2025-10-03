# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest
from pathlib import Path
from unittest.mock import patch

from torch.utils.data import Dataset

from ng_model_gym.core.data.dataloader import get_dataset, get_dataset_from_config
from ng_model_gym.core.data.dataset_registry import get_dataset_key, register_dataset
from ng_model_gym.core.data.utils import DataLoaderMode
from tests.testing_utils import create_simple_params


class MockDataset:
    """Mock Dataset class."""

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class TestDatasetFactory(unittest.TestCase):
    """Test Dataset Factory."""

    def setUp(self):
        train_data_dir = Path("./tests/usecases/nss/datasets/train")

        self.params = create_simple_params()
        self.params.dataset.path.train = train_data_dir

    def tearDown(self):
        pass

    def test_get_dataset_from_config(self):
        """Test getting the dataset class, based on the config file parameters."""

        for dataset_name in ("NSS", "nss", "Nss"):
            with self.subTest(dataset_name=dataset_name):
                self.params.dataset.name = dataset_name

                dataset_cls = get_dataset_from_config(self.params)

                self.assertTrue(isinstance(dataset_cls, type))
                self.assertEqual(dataset_cls.__name__, "NSSDataset")

    @patch("ng_model_gym.core.data.dataset_registry.Dataset", new=MockDataset)
    def test_get_dataset_from_config_no_version(self):
        """Test registering a dataset with no version,
        and getting the dataset based on the config."""

        @register_dataset("MockNSSDataset")
        class MockNSSDataset(MockDataset):  # pylint: disable=unused-variable
            """Mock NSS Dataset"""

            def __init__(self):
                pass

            def __len__(self):
                return 0

            def __getitem__(self, index):
                return None

        # Override dataset name and version from params
        self.params.dataset.name = "MockNSSDataset"
        self.params.dataset.version = None

        dataset_cls = get_dataset_from_config(self.params)

        self.assertEqual(dataset_cls.__name__, "MockNSSDataset")

    def test_error_on_unregistered_dataset_name(self):
        """Test loading a dataset class with a name that isn't registered."""

        # Override dataset name from params
        self.params.dataset.name = "unregistered_registered"

        dataset_key = get_dataset_key(
            self.params.dataset.name, self.params.dataset.version
        )

        with self.assertRaisesRegex(
            KeyError,
            rf"Dataset {dataset_key} is not registered",
        ):
            get_dataset_from_config(self.params)

    def test_error_on_unregistered_dataset_version(self):
        """Test loading a dataset class with a version that isn't registered."""

        # Override dataset version from params
        self.params.dataset.version = "0"

        dataset_key = get_dataset_key(
            self.params.dataset.name, self.params.dataset.version
        )

        with self.assertRaisesRegex(
            KeyError,
            rf"Dataset {dataset_key} is not registered",
        ):
            get_dataset_from_config(self.params)

    def test_get_dataset(self):
        """Test initialising the dataset from the registered class."""

        loader_mode = DataLoaderMode.TRAIN

        dataset = get_dataset(self.params, loader_mode)

        self.assertIsInstance(dataset, Dataset)
        self.assertEqual(dataset.loader_mode, loader_mode)


if __name__ == "__main__":
    unittest.main()
