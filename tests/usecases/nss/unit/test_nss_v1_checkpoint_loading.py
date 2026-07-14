# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from ng_model_gym.core.model.checkpoint_loader import load_checkpoint
from ng_model_gym.core.model.model_factory import create_model
from ng_model_gym.core.trainer import Trainer
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
        kpn_weight_key = "autoencoder.kpn_params.conv2d.weight"
        kpn_bias_key = "autoencoder.kpn_params.conv2d.bias"
        high_state[kpn_weight_key] = torch.arange(
            high_state[kpn_weight_key].numel(),
            dtype=high_state[kpn_weight_key].dtype,
        ).reshape_as(high_state[kpn_weight_key])
        high_state[kpn_bias_key] = torch.arange(
            high_state[kpn_bias_key].shape[0],
            dtype=high_state[kpn_bias_key].dtype,
        )

        high_ckpt = self.temp_path / "nss_v1_high_fp32_test.pt"
        mid_low_ckpt = self.temp_path / "nss_v1_mid_low_fp32_test.pt"

        torch.save({"model_state_dict": high_state}, high_ckpt)
        torch.save({"model_state_dict": mid_state}, mid_low_ckpt)

        return high_ckpt, mid_low_ckpt

    def _expected_6x6_to_4x4_indices(self) -> torch.Tensor:
        return torch.tensor(
            (7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28),
            dtype=torch.long,
        )

    def _create_params(self, quality: str, mode: TrainEvalMode = TrainEvalMode.FP32):
        params = create_simple_params(usecase="nss-v1")
        params.model.quality = quality
        params.model_train_eval_mode = mode
        return params

    def _create_model(self, quality: str):
        return create_model(self._create_params(quality), self.device)

    def _make_qat_like_states(self):
        """Create synthetic PT2E-like KPN tensors without tracing QAT."""
        target_model = self._create_model("mid")
        target_state = target_model.state_dict()
        target_state["autoencoder._param_constant18"] = torch.zeros(16, 32, 3, 3)
        target_state["autoencoder._param_constant19"] = torch.zeros(16)
        target_state["autoencoder.activation_post_process_30.scale"] = torch.zeros(16)
        target_state["autoencoder.activation_post_process_30.zero_point"] = torch.zeros(
            16, dtype=torch.int32
        )
        target_state[
            "autoencoder.activation_post_process_30.activation_post_process.min_val"
        ] = torch.zeros(16)
        target_state[
            "autoencoder.activation_post_process_30.activation_post_process.max_val"
        ] = torch.zeros(16)

        checkpoint_state = target_state.copy()
        checkpoint_state["autoencoder._param_constant18"] = torch.arange(
            36 * 32 * 3 * 3, dtype=torch.float32
        ).reshape(36, 32, 3, 3)
        checkpoint_state["autoencoder._param_constant19"] = torch.arange(
            36, dtype=torch.float32
        )
        checkpoint_state["autoencoder.activation_post_process_30.scale"] = torch.arange(
            36, dtype=torch.float32
        )
        checkpoint_state[
            "autoencoder.activation_post_process_30.zero_point"
        ] = torch.arange(36, dtype=torch.int32)
        checkpoint_state[
            "autoencoder.activation_post_process_30.activation_post_process.min_val"
        ] = torch.arange(36, dtype=torch.float32)
        max_val = torch.arange(36, dtype=torch.float32) + 100.0
        checkpoint_state[
            "autoencoder.activation_post_process_30.activation_post_process.max_val"
        ] = max_val

        return target_model, target_state, checkpoint_state

    def test_high_ckpt_prunes_to_mid_low(self) -> None:
        """Validate high checkpoint loads as 6x6 and prunes to centered 4x4."""

        kpn_weight_key = "autoencoder.kpn_params.conv2d.weight"
        kpn_bias_key = "autoencoder.kpn_params.conv2d.bias"
        checkpoint_state = torch.load(self.high_ckpt, weights_only=True)[
            "model_state_dict"
        ]

        high_params = self._create_params("high")
        high_model = load_checkpoint(self.high_ckpt, high_params, self.device)
        high_state = high_model.state_dict()

        self.assertEqual(high_state[kpn_weight_key].shape[0], 36)
        self.assertEqual(high_state[kpn_bias_key].shape[0], 36)

        expected_indices = self._expected_6x6_to_4x4_indices()
        for quality in ("mid", "low"):
            with self.subTest(quality=quality):
                params = self._create_params(quality)
                with self.assertLogs(
                    "ng_model_gym.usecases.nss.model.model_v1",
                    level="INFO",
                ) as log_ctx:
                    model = load_checkpoint(self.high_ckpt, params, self.device)
                state = model.state_dict()

                self.assertEqual(state[kpn_weight_key].shape[0], 16)
                self.assertEqual(state[kpn_bias_key].shape[0], 16)
                torch.testing.assert_close(
                    state[kpn_weight_key],
                    checkpoint_state[kpn_weight_key].index_select(0, expected_indices),
                )
                torch.testing.assert_close(
                    state[kpn_bias_key],
                    checkpoint_state[kpn_bias_key].index_select(0, expected_indices),
                )
                log_text = "\n".join(log_ctx.output)
                self.assertIn("high-quality", log_text)
                self.assertIn("mid/low-quality", log_text)
                self.assertIn("(6, 6)", log_text)
                self.assertIn("(4, 4)", log_text)

    def test_mid_low_ckpt_loads_into_mid_low_models(self) -> None:
        """Validate a mid/low checkpoint loads into both 4x4 quality modes."""

        kpn_weight_key = "autoencoder.kpn_params.conv2d.weight"
        kpn_bias_key = "autoencoder.kpn_params.conv2d.bias"
        checkpoint_state = torch.load(self.mid_low_ckpt, weights_only=True)[
            "model_state_dict"
        ]

        for quality in ("mid", "low"):
            with self.subTest(quality=quality):
                params = self._create_params(quality)
                with patch(
                    "ng_model_gym.usecases.nss.model.model_v1.logger.info"
                ) as info_log:
                    model_from_mid_low = load_checkpoint(
                        self.mid_low_ckpt,
                        params,
                        self.device,
                    )
                state_from_mid_low = model_from_mid_low.state_dict()

                self.assertEqual(
                    state_from_mid_low[kpn_weight_key].shape[0],
                    16,
                )
                torch.testing.assert_close(
                    state_from_mid_low[kpn_weight_key],
                    checkpoint_state[kpn_weight_key],
                )
                torch.testing.assert_close(
                    state_from_mid_low[kpn_bias_key],
                    checkpoint_state[kpn_bias_key],
                )
                self.assertFalse(
                    any(
                        call.args
                        and call.args[0].startswith(
                            "Pruned NSS v1 KPN checkpoint tensors"
                        )
                        for call in info_log.call_args_list
                    )
                )

    def test_qat_finetune_high_fp32_prunes_to_mid_low_models(self) -> None:
        """QAT finetune should prune high FP32 weights before QAT preparation."""

        kpn_weight_key = "autoencoder.kpn_params.conv2d.weight"
        kpn_bias_key = "autoencoder.kpn_params.conv2d.bias"
        checkpoint_state = torch.load(self.high_ckpt, weights_only=True)[
            "model_state_dict"
        ]
        expected_indices = self._expected_6x6_to_4x4_indices()

        for quality in ("mid", "low"):
            with self.subTest(quality=quality):
                params = self._create_params(quality, TrainEvalMode.QAT_INT8)
                params.train.qat.checkpoints.dir = self.temp_path / f"{quality}_qat"
                model = create_model(params, self.device)

                trainer = Mock(spec=Trainer)
                trainer.model = model
                trainer.params = params
                trainer.params.train.resume = None
                trainer.params.train.finetune = self.high_ckpt
                trainer.training_mode_params = params.train.qat

                Trainer._restore_model_weights(trainer)
                state = model.state_dict()

                torch.testing.assert_close(
                    state[kpn_weight_key],
                    checkpoint_state[kpn_weight_key].index_select(0, expected_indices),
                )
                torch.testing.assert_close(
                    state[kpn_bias_key],
                    checkpoint_state[kpn_bias_key].index_select(0, expected_indices),
                )

    def test_mid_low_ckpt_rejected_by_high_model(self) -> None:
        """Validate that loading a mid/low quality checkpoint into a high quality model fails."""

        params = create_simple_params(usecase="nss-v1")
        params.model.quality = "high"
        params.model_train_eval_mode = TrainEvalMode.FP32

        with self.assertRaisesRegex(ValueError, "Cannot expand KPN weights"):
            load_checkpoint(self.mid_low_ckpt, params, self.device)

    def test_global_load_state_dict_no_longer_prunes_cross_quality(self) -> None:
        """Raw load_state_dict should not do weights-only KPN conversion."""
        mid_model = self._create_model("mid")
        high_model = self._create_model("high")

        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            mid_model.load_state_dict(high_model.state_dict())

    def test_resume_high_checkpoint_rejected_by_mid_model(self) -> None:
        """Resume should reject a high checkpoint before restoring training state."""
        checkpoint_path = self.temp_path / "nss_v1_high_resume_test.pt"
        high_state = torch.load(self.high_ckpt, weights_only=True)["model_state_dict"]
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": high_state,
                "optimizer_state_dict": {},
                "lr_scheduler_state_dict": {},
            },
            checkpoint_path,
        )

        trainer = Mock(spec=Trainer)
        trainer.model = self._create_model("mid")
        trainer.optimizer = Mock()
        trainer.lr_schedule = Mock()
        trainer.params = self._create_params("mid")
        trainer.params.train.resume = checkpoint_path
        trainer.training_mode_params = trainer.params.train.fp32
        trainer._quantize_modules = Mock()

        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            Trainer._restore_model_weights(trainer)

        trainer.optimizer.load_state_dict.assert_not_called()
        trainer.lr_schedule.load_state_dict.assert_not_called()

    def test_non_finetune_checkpoint_hook_prunes_qat_shaped_tensors_for_eval_export(
        self,
    ) -> None:
        """Non-finetune checkpoint loading may prepare QAT-shaped tensors."""
        target_model, target_state, checkpoint_state = self._make_qat_like_states()
        target_model.state_dict = Mock(return_value=target_state)

        # qat --finetune rejects QAT/int8 checkpoints in Trainer before this hook.
        prepared = target_model.prepare_checkpoint_state_dict_for_weights_load(
            checkpoint_state
        )

        expected_indices = self._expected_6x6_to_4x4_indices()
        pruned_keys = (
            "autoencoder._param_constant18",
            "autoencoder._param_constant19",
            "autoencoder.activation_post_process_30.scale",
            "autoencoder.activation_post_process_30.zero_point",
            "autoencoder.activation_post_process_30.activation_post_process.min_val",
            "autoencoder.activation_post_process_30.activation_post_process.max_val",
        )
        for key in pruned_keys:
            expected = checkpoint_state[key].index_select(0, expected_indices)
            if torch.is_floating_point(checkpoint_state[key]):
                torch.testing.assert_close(prepared[key], expected)
            else:
                self.assertTrue(torch.equal(prepared[key], expected))

    def test_weights_only_hook_rejects_unsupported_kpn_transition(self) -> None:
        """Only documented 6x6 -> 4x4 pruning should be accepted."""
        mid_model = self._create_model("mid")
        state = mid_model.state_dict()
        state["autoencoder.kpn_params.conv2d.weight"] = torch.zeros(25, 32, 3, 3)
        state["autoencoder.kpn_params.conv2d.bias"] = torch.zeros(25)

        with self.assertRaisesRegex(
            ValueError, "Unsupported NSS v1 KPN prune transition"
        ):
            mid_model.prepare_checkpoint_state_dict_for_weights_load(state)


if __name__ == "__main__":
    unittest.main()
