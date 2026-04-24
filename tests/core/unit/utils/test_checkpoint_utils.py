# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
import tempfile
import time
import unittest
from collections import OrderedDict
from pathlib import Path

import torch

from ng_model_gym.core.model.checkpoint_loader import (
    latest_checkpoint_in_dir,
    remap_feedback_model_state_dict,
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

    def test_remap_feedback_model_state_dict_strips_legacy_prefix(self):
        """Legacy FeedbackModel checkpoints should drop the `nss_model.` namespace."""
        state_dict = OrderedDict(
            {
                "nss_model.autoencoder.weight": torch.ones(1),
                "nss_model.autoencoder.bias": torch.zeros(1),
            }
        )

        remapped = remap_feedback_model_state_dict(state_dict)

        self.assertIn("autoencoder.weight", remapped)
        self.assertIn("autoencoder.bias", remapped)
        self.assertNotIn("nss_model.autoencoder.weight", remapped)

    def test_remap_feedback_model_state_dict_compacts_legacy_qat_tensor_constants(self):
        """Old PT2E QAT checkpoints should drop BatchNorm tracking counters."""
        state_dict = OrderedDict(
            {
                "nss_model.autoencoder._tensor_constant0": torch.tensor(0),
                "nss_model.autoencoder._tensor_constant1": torch.ones(32),
                "nss_model.autoencoder._tensor_constant2": torch.zeros(32),
                "nss_model.autoencoder._tensor_constant3": torch.tensor(0),
                "nss_model.autoencoder._tensor_constant4": torch.ones(64),
                "nss_model.autoencoder._tensor_constant5": torch.zeros(64),
                "nss_model.autoencoder.activation_post_process_0.scale": torch.ones(1),
            }
        )

        remapped = remap_feedback_model_state_dict(state_dict)

        self.assertEqual(
            list(remapped.keys()),
            [
                "autoencoder._tensor_constant0",
                "autoencoder._tensor_constant1",
                "autoencoder._tensor_constant2",
                "autoencoder._tensor_constant3",
                "autoencoder.activation_post_process_0.scale",
            ],
        )
        self.assertTrue(
            torch.equal(remapped["autoencoder._tensor_constant0"], torch.ones(32))
        )
        self.assertTrue(
            torch.equal(remapped["autoencoder._tensor_constant3"], torch.zeros(64))
        )

    def test_remap_feedback_model_state_dict_compacts_legacy_qat_constants_without_prefix(
        self,
    ):
        """Legacy PT2E tensor constants should be compacted even without `nss_model.`."""
        state_dict = OrderedDict(
            {
                "autoencoder._tensor_constant0": torch.tensor(0),
                "autoencoder._tensor_constant1": torch.ones(16),
                "autoencoder._tensor_constant2": torch.zeros(16),
            }
        )

        remapped = remap_feedback_model_state_dict(state_dict)

        self.assertEqual(
            list(remapped.keys()),
            ["autoencoder._tensor_constant0", "autoencoder._tensor_constant1"],
        )
        self.assertTrue(
            torch.equal(remapped["autoencoder._tensor_constant1"], torch.zeros(16))
        )

    def test_remap_feedback_model_state_dict_keeps_current_qat_tensor_constants(self):
        """Current PT2E QAT checkpoints should remain unchanged."""
        state_dict = OrderedDict(
            {
                "autoencoder._tensor_constant0": torch.ones(32),
                "autoencoder._tensor_constant1": torch.zeros(32),
                "autoencoder._tensor_constant2": torch.ones(64),
            }
        )

        remapped = remap_feedback_model_state_dict(state_dict)

        self.assertEqual(list(remapped.keys()), list(state_dict.keys()))
