# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from ng_model_gym.core.model.base_ng_model import BaseNGModel
from ng_model_gym.core.model.checkpoint_loader import (
    latest_checkpoint_in_dir,
    load_checkpoint,
)
from ng_model_gym.core.utils.enum_definitions import TrainEvalMode
from tests.testing_utils import create_simple_params


class _HookedCheckpointModel(BaseNGModel):
    """Small model that records weights-only checkpoint preparation."""

    def __init__(self, params):
        super().__init__(params)
        self.network = nn.Linear(1, 1)
        self.prepare_seen_state_dict = None

    def get_neural_network(self) -> nn.Module:
        return self.network

    def set_neural_network(self, neural_network: nn.Module) -> None:
        self.network = neural_network

    def forward(self, x):
        """Run a forward pass through the tiny test network."""
        return {"output": self.network(x)}

    def prepare_checkpoint_state_dict_for_weights_load(self, state_dict):
        """Record weights-only hook input and return loadable weights."""
        self.prepare_seen_state_dict = state_dict
        prepared = self.state_dict()
        prepared["network.weight"] = torch.full((1, 1), 2.0)
        prepared["network.bias"] = torch.full((1,), 3.0)
        return prepared


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

    def test_load_checkpoint_calls_weights_only_prepare_hook(self):
        """Weights-only checkpoint loading should call the optional model hook."""
        params = create_simple_params(usecase="nss-v1")
        params.model_train_eval_mode = TrainEvalMode.FP32
        model = _HookedCheckpointModel(params)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir, "checkpoint.pt")
            original_state = {
                "network.weight": torch.full((1, 1), -1.0),
                "network.bias": torch.full((1,), -2.0),
            }
            torch.save({"model_state_dict": original_state}, checkpoint_path)

            with patch(
                "ng_model_gym.core.model.checkpoint_loader.create_model",
                return_value=model,
            ):
                loaded_model = load_checkpoint(
                    checkpoint_path, params, torch.device("cpu")
                )

        self.assertIs(loaded_model, model)
        self.assertTrue(
            torch.equal(
                model.prepare_seen_state_dict["network.weight"],
                original_state["network.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                model.prepare_seen_state_dict["network.bias"],
                original_state["network.bias"],
            )
        )
        self.assertTrue(
            torch.equal(model.network.weight.detach(), torch.full((1, 1), 2.0))
        )
        self.assertTrue(torch.equal(model.network.bias.detach(), torch.full((1,), 3.0)))
