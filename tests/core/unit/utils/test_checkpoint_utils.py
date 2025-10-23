# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
import tempfile
import unittest
from pathlib import Path

import torch

from ng_model_gym.core.utils.checkpoint_utils import (
    latest_checkpoint_path,
    latest_training_run_dir,
    replace_prefix_in_state_dict,
)


class RestorePretrainedModelFromCheckpoints(unittest.TestCase):
    """Unit test for the checkpoints provided for the pre-trained NSS model."""

    def setUp(self):
        """Disable logging"""
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        """Restore logging"""
        logging.disable(logging.NOTSET)

    def test_throw_when_no_checkpoint_dir_set(self):
        """Throw when no checkpoint dir path is set"""
        self.assertRaises(NotADirectoryError, latest_training_run_dir, Path("test"))

    def test_throw_when_no_checkpoint_dirs(self):
        """Throw when no checkpoint subdirectories exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertRaises(LookupError, latest_training_run_dir, Path(temp_dir))

    def test_find_latest_checkpoint_dir(self):
        """Test getting the latest checkpoint dir"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dirs_to_create = [
                "27-12-12_24-00-03",  # Incorrect timestamp should be ignored
                "24-01-01_09-00-03",
                "random_dir",
                "25-01-01_10-00-03",
            ]

            for timestamped_dir in dirs_to_create:
                Path(temp_dir, timestamped_dir).mkdir()

            self.assertEqual(
                latest_training_run_dir(Path(temp_dir)),
                Path(temp_dir, "25-01-01_10-00-03"),
            )

    def test_throw_when_no_checkpoints(self):
        """Test throw when no checkpoints to restore from"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_folder = Path(temp_dir, "25-01-01_10-00-03")
            ckpt_folder.mkdir()
            self.assertRaises(LookupError, latest_checkpoint_path, ckpt_folder)

    def test_find_latest_checkpoint_file(self):
        """Test getting the latest checkpoint file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            files_to_create = ["ckpt-0.pt", "ckpt-test100.pt", "ckpt-1.pt"]

            training_dir = Path(temp_dir, "25-01-01_10-00-03")
            training_dir.mkdir()

            for checkpoint_path in files_to_create:
                Path(training_dir, checkpoint_path).touch()

            self.assertEqual(
                latest_checkpoint_path(Path(temp_dir)),
                Path(temp_dir, "25-01-01_10-00-03", "ckpt-1.pt"),
            )


class StateDictPrefixReplacement(unittest.TestCase):
    """Test replacing the prefix in a model's state dict"""

    class MockNN(torch.nn.Module):
        """Model with old namespace `nss_model`"""

        def __init__(self):
            super().__init__()
            self.nss_model = (torch.nn.Conv2d(1, 2, kernel_size=1),)

        def forward(self, x):
            """Mock forward pass"""
            return x

    def test_replace_prefix(self):
        """Test prefix replacement works correctly"""
        old_model = self.MockNN()
        state_dict = old_model.state_dict()

        out_state_dict = replace_prefix_in_state_dict(
            state_dict, "nss_model", "ng_model"
        )
        self.assertIsInstance(out_state_dict, dict)

        self.assertTrue(all(key.startswith("ng_model") for key in state_dict.keys()))


if __name__ == "__main__":
    unittest.main()
