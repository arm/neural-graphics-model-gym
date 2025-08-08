# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import MagicMock, patch

import torch

from ng_model_gym.nss.dataloader.health_check import health_check_dataset


class TestHealthCheckDataset(unittest.TestCase):
    """Tests for health check dataset."""

    class SampleDataset(torch.utils.data.Dataset):
        """An example Dataset for testing purposes."""

        def __init__(self):
            self.x = {
                "motion": torch.rand((5, 3, 100, 10)),
                "colour_linear": torch.rand((5, 3, 100, 100)),
                "jitter": torch.rand((5, 3, 100, 100)),
            }
            self.y = torch.rand((5, 3, 100, 100))

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return self.x, self.y

    def setUp(self):
        """Create a sample dataset."""
        self.dataset = torch.utils.data.DataLoader(
            TestHealthCheckDataset.SampleDataset(), batch_size=1
        )

    @patch("ng_model_gym.nss.dataloader.health_check.logger")
    def test_health_check_in_test_mode(self, mock_logger):
        """Test that we don't check in test mode,"""
        health_check_dataset(self.dataset, "test")
        mock_logger.info.assert_called_with(
            "DATASET: Health check is only supported for the train dataset"
        )

    @patch("ng_model_gym.nss.dataloader.health_check.psnr")
    @patch("ng_model_gym.nss.dataloader.health_check.DenseWarp")
    @patch("ng_model_gym.nss.dataloader.health_check.DownSampling2D")
    @patch("ng_model_gym.nss.dataloader.health_check.tqdm")
    def test_health_check_in_train_mode_failure(
        self, mock_tqdm, mock_down_sampling2d, mock_dense_warp, mock_psnr
    ):
        """Test that we don't throw failure in the train mode when psnr is constant."""
        mock_psnr.return_value = torch.tensor(10.0)
        mock_dense_warp.return_value = MagicMock()
        mock_dense_warp.return_value.__getitem__.side_effect = lambda x: x[1]

        mock_down_sampling2d.return_value = MagicMock()
        mock_down_sampling2d.return_value.side_effect = lambda x: x

        mock_tqdm_instance = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_tqdm_instance

        health_check_dataset(self.dataset, "train")


if __name__ == "__main__":
    unittest.main()
