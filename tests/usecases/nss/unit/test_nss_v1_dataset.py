# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ng_model_gym.core.data import DataLoaderMode, DatasetType
from ng_model_gym.usecases.nss.data.dataset import NSSDataset
from tests.testing_utils import create_simple_params


class TestNSSV1DatasetGolden(unittest.TestCase):
    """Test NSS v1 dataset output against known golden values."""

    def test_dataset_first_batch_golden_values(self):
        """Test first deterministic NSS v1 dataset batch."""

        expected_data = torch.load(
            "tests/usecases/nss/unit/data/nss_v1_golden_values/"
            + "nss_v1_high_dataset_output_golden.pt",
            map_location=torch.device("cpu"),
            weights_only=True,
        )

        params = create_simple_params(
            usecase="nss_v1", dataset_path=Path("tests/usecases/nss/datasets/train")
        )
        params.model.quality = "high"
        params.model.recurrent_samples = 2
        params.train.batch_size = 2
        params.dataset.gt_augmentation = False
        params.dataset.exposure = 2.0
        params.dataset.tonemapper = "reinhard"
        params.dataset.num_workers = 0

        dataset = NSSDataset(
            params,
            loader_mode=DataLoaderMode.TRAIN,
            extension=DatasetType.SAFETENSOR,
        )
        dataloader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=params.train.batch_size,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        x, y = next(iter(dataloader))
        actual_data = {**x, "ground_truth": y}

        expected_seq = expected_data.pop("seq")
        actual_seq = actual_data.pop("seq")
        self.assertEqual(actual_seq.shape, expected_seq.shape)
        self.assertEqual(actual_seq.dtype, expected_seq.dtype)

        for metadata_key in ("batch_size", "recurrent_samples", "quality"):
            expected_data.pop(metadata_key, None)

        for key, expected_tensor in expected_data.items():
            with self.subTest(key=key):
                torch.testing.assert_close(
                    actual_data[key],
                    expected_tensor,
                    rtol=1e-5,
                    atol=1e-8,
                    msg=f"Mismatch in dataset tensor '{key}'",
                )
