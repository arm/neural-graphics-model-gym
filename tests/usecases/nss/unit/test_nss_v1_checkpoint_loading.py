# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from ng_model_gym.core.model.checkpoint_loader import load_checkpoint
from ng_model_gym.core.model.model_factory import create_model
from ng_model_gym.core.utils.enum_definitions import TrainEvalMode
from tests.testing_utils import create_simple_params


class TestNSSV1CheckpointLoading(unittest.TestCase):
    """Validate NSS v1 checkpoint loading with KPN pruning."""

    def setUp(self) -> None:
        """Create checkpoints for quality-mode checkpoint loading tests."""

        self.device = torch.device("cpu")
        self.temp_path = Path(tempfile.mkdtemp())
        self.high_ckpt, self.mid_low_ckpt = self._create_fake_checkpoints()

    def tearDown(self) -> None:
        """Clean up checkpoint fixtures."""

        shutil.rmtree(self.temp_path)

    def _create_fake_checkpoints(self) -> tuple[Path, Path]:
        """Create fake high and mid/low checkpoints from model state dicts."""

        mid_params = create_simple_params(usecase="nss-v1")
        mid_params.model.quality = "mid"
        mid_params.model_train_eval_mode = TrainEvalMode.FP32

        mid_model = create_model(mid_params, self.device)
        mid_state = mid_model.state_dict()

        high_params = create_simple_params(usecase="nss-v1")
        high_params.model.quality = "high"
        high_params.model_train_eval_mode = TrainEvalMode.FP32

        high_model = create_model(high_params, self.device)
        high_state = high_model.state_dict()

        high_ckpt = self.temp_path / "nss_v1_high_fp32_test.pt"
        mid_low_ckpt = self.temp_path / "nss_v1_mid_low_fp32_test.pt"

        torch.save({"model_state_dict": high_state}, high_ckpt)
        torch.save({"model_state_dict": mid_state}, mid_low_ckpt)

        return high_ckpt, mid_low_ckpt

    def test_high_ckpt_prunes_to_mid_low(self) -> None:
        """Validate high checkpoint loads as 6x6 for high and prunes to 4x4 for mid."""

        high_params = create_simple_params(usecase="nss-v1")
        high_params.model.quality = "high"
        high_params.model_train_eval_mode = TrainEvalMode.FP32

        # Load high quality checkpoint into high quality model
        high_model = load_checkpoint(self.high_ckpt, high_params, self.device)
        high_state = high_model.state_dict()

        self.assertEqual(
            high_state["autoencoder.kpn_params.conv2d.weight"].shape[0], 36
        )  # High uses 6x6 KPN

        self.assertEqual(high_state["autoencoder.kpn_params.conv2d.bias"].shape[0], 36)

        mid_params = create_simple_params(usecase="nss-v1")
        mid_params.model.quality = "mid"
        mid_params.model_train_eval_mode = TrainEvalMode.FP32

        # Load high quality checkpoint into mid quality model
        mid_model = load_checkpoint(self.high_ckpt, mid_params, self.device)
        mid_state = mid_model.state_dict()

        self.assertEqual(
            mid_state["autoencoder.kpn_params.conv2d.weight"].shape[0], 16
        )  # Mid/low uses 4x4 KPN

        self.assertEqual(mid_state["autoencoder.kpn_params.conv2d.bias"].shape[0], 16)

    def test_mid_low_ckpt_loads_into_mid_model(self) -> None:
        """Validate that a mid/low quality checkpoint loads into a mid quality model."""

        params = create_simple_params(usecase="nss-v1")
        params.model.quality = "mid"
        params.model_train_eval_mode = TrainEvalMode.FP32

        model_from_mid_low = load_checkpoint(self.mid_low_ckpt, params, self.device)
        state_from_mid_low = model_from_mid_low.state_dict()

        self.assertIn("autoencoder.kpn_params.conv2d.weight", state_from_mid_low)

    def test_mid_low_ckpt_rejected_by_high_model(self) -> None:
        """Validate that loading a mid/low quality checkpoint into a high quality model fails."""

        params = create_simple_params(usecase="nss-v1")
        params.model.quality = "high"
        params.model_train_eval_mode = TrainEvalMode.FP32

        with self.assertRaisesRegex(ValueError, "Cannot expand KPN weights"):
            load_checkpoint(self.mid_low_ckpt, params, self.device)


if __name__ == "__main__":
    unittest.main()
