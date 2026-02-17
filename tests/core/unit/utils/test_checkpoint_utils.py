# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
import tempfile
import time
import unittest
from pathlib import Path

from ng_model_gym.core.utils.checkpoint_utils import latest_checkpoint_in_dir


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
        self.assertRaises(NotADirectoryError, latest_checkpoint_in_dir, Path("test"))

    def test_throw_when_no_checkpoints(self):
        """Test throw when no checkpoints to restore from"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_folder = Path(temp_dir, "25-01-01_10-00-03")
            ckpt_folder.mkdir()
            self.assertRaises(FileNotFoundError, latest_checkpoint_in_dir, ckpt_folder)

    def test_find_latest_checkpoint_file(self):
        """Test getting the latest checkpoint file in directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_dir = Path(temp_dir, "my-ckpt-dir")
            ckpt_dir.mkdir()

            old_checkpoint = Path(ckpt_dir, "ckpt-0.pt")
            old_checkpoint.touch()
            time.sleep(0.01)

            latest_checkpoint = Path(ckpt_dir, "ckpt-1.pt")
            latest_checkpoint.touch()

            self.assertEqual(
                latest_checkpoint_in_dir(Path(temp_dir)), latest_checkpoint
            )

    def test_resume_from_checkpoint_file_path(self):
        """Test direct checkpoint file paths"""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir, "checkpoint.pt")
            checkpoint_path.touch()

            self.assertEqual(latest_checkpoint_in_dir(checkpoint_path), checkpoint_path)

    def test_ignore_non_pt_checkpoint_file(self):
        """Files that are not checkpoints should raise"""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir, "checkpoint.bin")
            checkpoint_path.touch()

            self.assertRaises(ValueError, latest_checkpoint_in_dir, checkpoint_path)

    def test_resume_from_checkpoint_directory_with_files(self):
        """Test checkpoint directories containing .pt files are supported"""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_files = ["ckpt-0.pt", "model.pt", "ckpt-3.pt"]
            for ckpt in checkpoint_files:
                Path(temp_dir, ckpt).touch()
                time.sleep(0.01)

            self.assertEqual(
                latest_checkpoint_in_dir(Path(temp_dir)),
                Path(temp_dir, "ckpt-3.pt"),
            )
