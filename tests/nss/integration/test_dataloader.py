# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import random
import unittest

import torch
from safetensors import safe_open

from ng_model_gym.core.dataloader import NSSDataset
from ng_model_gym.usecases.nss.dataloader.utils import DataLoaderMode, DatasetType
from tests.unit.utils.utils import create_simple_params


class TestNSSDataset(unittest.TestCase):
    """Test NSSDataset class"""

    def setUp(self):
        """Set up test"""
        self.params = create_simple_params(dataset="./tests/nss/datasets/train")
        self.params.dataset.recurrent_samples = 4

    def test_existing_safetensors_file(self):
        """Test loading existing Safetensors file"""
        dataset = NSSDataset(self.params, DataLoaderMode.TRAIN)
        # Check if two Safetensors files have been found
        self.assertEqual(len(dataset.sequences), 1)
        x, y = dataset[0]

        # Check outputs exist
        self.assertIsInstance(x, dict)
        self.assertEqual(len(x), 13)
        self.assertIsInstance(y, torch.Tensor)

    def test_len_matches_frame_indexes(self):
        """Test dataset length matches total windows"""
        dataset = NSSDataset(self.params, DataLoaderMode.TRAIN)
        # Calculate number of sliding windows
        total_windows = 0
        for safetensor_file in dataset.sequences:
            with safe_open(safetensor_file, framework="pt") as f:
                length = int(f.metadata()["Length"])
            total_windows += length - (self.params.dataset.recurrent_samples - 1)
        self.assertEqual(len(dataset), total_windows)

    def test_dataloader_data_transformation(self):
        """Check data from raw to training transformation is correct"""

        params = create_simple_params(dataset="./tests/nss/datasets/train")
        params.dataset.recurrent_samples = 2
        params.train.batch_size = 2
        params.dataset.gt_augmentation = False

        dataset = NSSDataset(
            params, loader_mode=DataLoaderMode.TRAIN, extension=DatasetType.SAFETENSOR
        )

        def seed_worker(_):
            torch.manual_seed(params.train.seed)
            worker_seed = torch.initial_seed() % 2**32
            random.seed(worker_seed)

        expected_tensor = torch.Generator()
        expected_tensor.manual_seed(params.train.seed)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=params.train.batch_size,
            num_workers=0,
            pin_memory=True,
            prefetch_factor=None,
            drop_last=True,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=expected_tensor,
        )

        data = next(iter(dataloader))[0]

        golden_data = torch.load(
            "tests/nss/unit/data/nss_v1_golden_values/dataloader_output_fp32.pt",
            map_location=torch.device("cpu"),
            weights_only=True,
        )

        data.pop("seq", None)
        data.pop("colour", None)
        golden_data.pop("seq", None)

        for key in data:
            tensor = data[key]
            expected_tensor = golden_data[key]

            torch.testing.assert_close(
                tensor,
                expected_tensor,
                atol=1e-8,
                rtol=1e-5,
                msg=f"Mismatch in tensor '{key}'",
            )

            self.assertTrue(
                torch.equal(tensor, expected_tensor), f"Tensors {key} not equal"
            )


if __name__ == "__main__":
    unittest.main()
