# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest import mock

import torch

from ng_model_gym.core.loss import LossV0_1
from ng_model_gym.core.loss.losses import LossV1, LPIPSSpatialLossV1


class MockLPIPS(torch.nn.Module):
    """LPIPS test double that exposes whether input normalization was requested."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, image_a, image_b, normalize=False):
        """Return a deterministic LPIPS value based on the normalize argument."""
        self.calls.append(
            {
                "image_a": image_a,
                "image_b": image_b,
                "normalize": normalize,
            }
        )
        value = 0.75 if normalize else 10.0
        return image_a.new_full(
            (image_a.shape[0], 1, image_a.shape[-2], image_a.shape[-1]), value
        )


class TestLossV0_1(unittest.TestCase):
    """Tests for LossV0_1 class"""

    def test_loss_function(self):
        """Test loss function against golden value"""
        device = torch.device("cuda")
        # Load loss files from test
        loss_input = torch.load(
            "tests/usecases/nss/unit/data/nss_v0_1_golden_values/loss_golden.pt",
            map_location=device,
            weights_only=True,
        )

        y_true = loss_input["y_true"]
        # Historic golden files stored both y_pred and inputs_dataset inside a single dict.
        # Support the legacy combined format first, otherwise rely on the new layout.
        if "y_pred_and_inps" in loss_input:
            y_pred_and_inps = loss_input["y_pred_and_inps"]
            y_pred = {
                "output": y_pred_and_inps["output"],
                "out_filtered": y_pred_and_inps["out_filtered"],
                "motion": y_pred_and_inps["motion"],
            }
        else:
            y_pred = dict(loss_input["y_pred"])
            self.assertIn("motion", y_pred, "y_pred must contain 'motion' key")

        recurrent_samples = y_true.shape[1]
        self.assertEqual(recurrent_samples, 4)
        criterion = LossV0_1(recurrent_samples, device)

        loss = criterion(y_true, y_pred)
        self.assertAlmostEqual(loss.item(), loss_input["loss"], places=3)


class TestLPIPSSpatialLossV1(unittest.TestCase):
    """Tests for LPIPSSpatialLossV1 class"""

    @mock.patch("ng_model_gym.core.loss.losses.lpips.LPIPS")
    def test_uses_normalized_lpips_when_blending_with_l1(self, mock_lpips):
        """Test v1 blends L1 with normalized LPIPS."""
        lpips_loss = MockLPIPS()
        mock_lpips.return_value = lpips_loss
        y_true = torch.tensor(
            [
                [
                    [[0.0, 0.2], [0.4, 0.6]],
                    [[0.1, 0.3], [0.5, 0.7]],
                    [[0.2, 0.4], [0.6, 0.8]],
                ]
            ]
        )
        y_pred_out = torch.tensor(
            [
                [
                    [[0.5, 0.2], [0.9, 0.1]],
                    [[0.0, 0.6], [0.5, 1.0]],
                    [[0.6, 0.3], [0.2, 0.8]],
                ]
            ]
        )
        criterion = LPIPSSpatialLossV1({"alpha": 0.25}, torch.device("cpu"))

        loss = criterion(y_true, {"output": y_pred_out})

        l1_loss = torch.nn.functional.l1_loss(y_true, y_pred_out, reduction="none")
        normalized_lpips_loss = torch.full((1, 1, 2, 2), 0.75)
        expected_loss = (l1_loss * 0.75 + normalized_lpips_loss * 0.25).mean()
        self.assertTrue(torch.allclose(loss, expected_loss))
        self.assertEqual(len(lpips_loss.calls), 1)
        self.assertIs(lpips_loss.calls[0]["image_a"], y_true)
        self.assertIs(lpips_loss.calls[0]["image_b"], y_pred_out)
        self.assertTrue(lpips_loss.calls[0]["normalize"])

    @mock.patch("ng_model_gym.core.loss.losses.lpips.LPIPS")
    def test_alpha_one_returns_normalized_lpips_term(self, mock_lpips):
        """Test alpha fully selects the v1 normalized LPIPS term."""
        lpips_loss = MockLPIPS()
        mock_lpips.return_value = lpips_loss
        y_true = torch.zeros((2, 3, 4, 4))
        y_pred_out = torch.ones((2, 3, 4, 4))
        criterion = LPIPSSpatialLossV1({"alpha": 1.0}, torch.device("cpu"))

        loss = criterion(y_true, {"output": y_pred_out})

        self.assertEqual(loss.item(), 0.75)
        self.assertTrue(lpips_loss.calls[0]["normalize"])

    def test_requires_alpha_loss_arg(self):
        """Test alpha is required to avoid ambiguous blending behaviour."""
        with self.assertRaisesRegex(
            ValueError, "LPIPSSpatialLossV1 requires 'alpha' in loss_args"
        ):
            LPIPSSpatialLossV1({}, torch.device("cpu"))


class TestLossV1(unittest.TestCase):
    """Tests for the NSS v1 recurrent loss."""

    def _make_sequence(self, timesteps=3):
        """Create a small CPU-safe recurrent loss input."""

        batch, channels, height, width = 1, 3, 2, 2
        y_true = torch.zeros((batch, timesteps, channels, height, width))
        y_pred = {
            "output": torch.zeros_like(y_true),
            "out_filtered": torch.zeros_like(y_true),
            "motion": torch.zeros((batch, timesteps, 2, height, width)),
        }
        return y_true, y_pred

    @mock.patch("ng_model_gym.core.loss.losses.lpips.LPIPS")
    def test_first_frame_filtered_supervision_is_additive(self, mock_lpips):
        """Test v1 adds filtered and first-frame filtered-head supervision."""

        lpips_loss = MockLPIPS()
        mock_lpips.return_value = lpips_loss
        y_true, y_pred = self._make_sequence(timesteps=3)
        y_pred["out_filtered"][:, 0, ...] = 2.0
        loss_args = {
            "alpha_reg_weight": 0.0,
            "change_pred_weight": 0.0,
            "filtered_weight": 0.25,
            "first_frame_weight": 0.5,
            "lpips_weight": 0.2,
            "max_weight": 1.0,
            "min_weight": 1.0,
            "temporal_reg_weight": 0.0,
            "theta_reg_weight": 0.0,
        }
        criterion = LossV1(3, torch.device("cpu"), loss_args)

        loss = criterion(y_true, y_pred)

        spatial_lpips = torch.tensor(0.2 * 0.75)
        filtered_l1 = torch.tensor(2.0 / 3.0)
        first_frame_loss = torch.tensor(2.0 + 0.2 * 0.75)
        expected = spatial_lpips + 0.25 * filtered_l1 + 0.5 * first_frame_loss
        torch.testing.assert_close(loss, expected)
        self.assertEqual(len(lpips_loss.calls), 3)
        self.assertTrue(all(call["normalize"] for call in lpips_loss.calls))
        torch.testing.assert_close(
            lpips_loss.calls[-1]["image_a"],
            y_pred["out_filtered"][:, 0, ...],
        )
        torch.testing.assert_close(
            lpips_loss.calls[-1]["image_b"],
            y_true[:, 0, ...],
        )

    @mock.patch("ng_model_gym.core.loss.losses.lpips.LPIPS")
    def test_spatial_lpips_calls_are_normalized_on_cpu_without_autocast(
        self, mock_lpips
    ):
        """Test spatial LPIPS uses normalized inputs and avoids CPU autocast."""

        lpips_loss = MockLPIPS()
        mock_lpips.return_value = lpips_loss
        y_true, y_pred = self._make_sequence(timesteps=4)
        for timestep in range(4):
            y_pred["output"][:, timestep, ...] = float(timestep)
        loss_args = {
            "alpha_reg_weight": 0.0,
            "change_pred_weight": 0.0,
            "first_frame_weight": 0.0,
            "temporal_reg_weight": 0.0,
            "theta_reg_weight": 0.0,
        }
        criterion = LossV1(4, torch.device("cpu"), loss_args)

        with mock.patch("ng_model_gym.core.loss.losses.torch.autocast") as autocast:
            criterion(y_true, y_pred)

        autocast.assert_not_called()
        for timestep in range(1, 4):
            matching_calls = [
                call
                for call in lpips_loss.calls
                if torch.equal(call["image_a"], y_pred["output"][:, timestep, ...])
            ]
            self.assertEqual(len(matching_calls), 1)
            self.assertTrue(matching_calls[0]["normalize"])

    @mock.patch("ng_model_gym.core.loss.losses.lpips.LPIPS")
    def test_missing_optional_regularizer_tensors_do_not_fail(self, mock_lpips):
        """Test nonzero optional weights are safe when optional tensors are absent."""

        mock_lpips.return_value = MockLPIPS()
        y_true, y_pred = self._make_sequence(timesteps=3)
        criterion = LossV1(3, torch.device("cpu"), {})

        loss = criterion(y_true, y_pred)

        self.assertTrue(torch.isfinite(loss))

    @mock.patch("ng_model_gym.core.loss.losses.lpips.LPIPS")
    def test_timestep_count_must_exceed_one(self, mock_lpips):
        """Test v1 raises for sequences without temporal pairs."""

        mock_lpips.return_value = MockLPIPS()
        y_true, y_pred = self._make_sequence(timesteps=1)
        criterion = LossV1(1, torch.device("cpu"), {})

        with self.assertRaisesRegex(ValueError, "length greater than 1"):
            criterion(y_true, y_pred)

    @mock.patch("ng_model_gym.core.loss.losses.lpips.LPIPS")
    def test_loss_args_override_training_defaults(self, mock_lpips):
        """Test key v1 defaults are overridden from loss_args."""

        mock_lpips.return_value = MockLPIPS()
        loss_args = {
            "alpha_reg_weight": 0.33,
            "lpips_net": "vgg",
            "min_weight": 0.25,
            "temporal_reg_channels": 4,
            "temporal_reg_weight": 0.7,
        }

        criterion = LossV1(3, torch.device("cpu"), loss_args)

        self.assertEqual(criterion.alpha_reg_weight, 0.33)
        self.assertEqual(criterion.lpips_net, "vgg")
        self.assertEqual(criterion.min_weight, 0.25)
        self.assertEqual(criterion.temporal_reg_channels, 4)
        self.assertEqual(criterion.temporal_reg_weight, 0.7)
        mock_lpips.assert_called_once_with(net="vgg")

    @mock.patch("ng_model_gym.core.loss.losses.lpips.LPIPS")
    def test_loss_args_omitted_uses_training_parity_defaults(self, mock_lpips):
        """Test omitted v1 loss args use the NSS training defaults."""

        mock_lpips.return_value = MockLPIPS()

        criterion = LossV1(3, torch.device("cpu"), {})

        self.assertEqual(criterion.alpha_reg_weight, 0.0001)
        self.assertEqual(criterion.min_weight, 0.1)
        self.assertEqual(criterion.temporal_reg_channels, 1)
        self.assertEqual(criterion.temporal_reg_weight, 0.7)

    @mock.patch("ng_model_gym.core.loss.losses.lpips.LPIPS")
    def test_change_prediction_motion_spatial_mismatch_raises(self, mock_lpips):
        """Test change prediction does not resize mismatched motion fields."""

        mock_lpips.return_value = MockLPIPS()
        y_true = torch.zeros((1, 3, 3, 2, 4))
        y_pred = {
            "output": torch.zeros_like(y_true),
            "out_filtered": torch.zeros_like(y_true),
            "motion": torch.zeros((1, 3, 2, 3, 3)),
        }
        loss_args = {
            "alpha_reg_weight": 0.0,
            "change_pred_weight": 1.0,
            "first_frame_weight": 0.0,
            "temporal_reg_weight": 0.0,
            "theta_reg_weight": 0.0,
        }
        criterion = LossV1(3, torch.device("cpu"), loss_args)

        with self.assertRaisesRegex(ValueError, "motion spatial dimensions"):
            criterion(y_true, y_pred)

    @mock.patch("ng_model_gym.core.loss.losses.lpips.LPIPS")
    def test_temporal_weight_mask_disocclusion_batch_mismatch_raises(self, mock_lpips):
        """Test temporal masks reject disocclusion masks with the wrong batch."""

        mock_lpips.return_value = MockLPIPS()
        y_true, y_pred = self._make_sequence(timesteps=3)
        y_pred["temporal_params"] = torch.zeros((1, 3, 2, 2, 2))
        y_pred["disocclusion_mask"] = torch.zeros((2, 3, 1, 2, 2))
        loss_args = {
            "alpha_reg_weight": 1.0,
            "change_pred_weight": 0.0,
            "first_frame_weight": 0.0,
            "temporal_reg_weight": 0.0,
            "theta_reg_weight": 0.0,
        }
        criterion = LossV1(3, torch.device("cpu"), loss_args)

        with self.assertRaisesRegex(ValueError, "disocclusion_mask batch"):
            criterion(y_true, y_pred)

    @mock.patch("ng_model_gym.core.loss.losses.lpips.LPIPS")
    def test_temporal_weight_mask_disocclusion_timestep_mismatch_raises(
        self, mock_lpips
    ):
        """Test temporal masks reject disocclusion masks with extra timesteps."""

        mock_lpips.return_value = MockLPIPS()
        y_true, y_pred = self._make_sequence(timesteps=3)
        y_pred["temporal_params"] = torch.zeros((1, 3, 2, 2, 2))
        y_pred["disocclusion_mask"] = torch.zeros((1, 4, 1, 2, 2))
        loss_args = {
            "alpha_reg_weight": 1.0,
            "change_pred_weight": 0.0,
            "first_frame_weight": 0.0,
            "temporal_reg_weight": 0.0,
            "theta_reg_weight": 0.0,
        }
        criterion = LossV1(3, torch.device("cpu"), loss_args)

        with self.assertRaisesRegex(ValueError, "disocclusion_mask timesteps"):
            criterion(y_true, y_pred)

    @mock.patch("ng_model_gym.core.loss.losses.lpips.LPIPS")
    def test_temporal_weight_mask_reset_event_batch_mismatch_raises(self, mock_lpips):
        """Test temporal masks reject reset events with the wrong batch."""

        mock_lpips.return_value = MockLPIPS()
        y_true, y_pred = self._make_sequence(timesteps=3)
        y_pred["temporal_params"] = torch.zeros((1, 3, 2, 2, 2))
        y_pred["reset_event"] = torch.ones((2, 3, 1, 1, 1))
        loss_args = {
            "alpha_reg_weight": 1.0,
            "change_pred_weight": 0.0,
            "first_frame_weight": 0.0,
            "temporal_reg_weight": 0.0,
            "theta_reg_weight": 0.0,
        }
        criterion = LossV1(3, torch.device("cpu"), loss_args)

        with self.assertRaisesRegex(ValueError, "reset_event batch"):
            criterion(y_true, y_pred)

    @mock.patch("ng_model_gym.core.loss.losses.lpips.LPIPS")
    def test_temporal_weight_mask_reset_event_timestep_mismatch_raises(
        self, mock_lpips
    ):
        """Test temporal masks reject reset events with extra timesteps."""

        mock_lpips.return_value = MockLPIPS()
        y_true, y_pred = self._make_sequence(timesteps=3)
        y_pred["temporal_params"] = torch.zeros((1, 3, 2, 2, 2))
        y_pred["reset_event"] = torch.ones((1, 4, 1, 1, 1))
        loss_args = {
            "alpha_reg_weight": 1.0,
            "change_pred_weight": 0.0,
            "first_frame_weight": 0.0,
            "temporal_reg_weight": 0.0,
            "theta_reg_weight": 0.0,
        }
        criterion = LossV1(3, torch.device("cpu"), loss_args)

        with self.assertRaisesRegex(ValueError, "reset_event timesteps"):
            criterion(y_true, y_pred)

    @mock.patch("ng_model_gym.core.loss.losses.lpips.LPIPS")
    def test_temporal_weight_mask_reset_event_rank_mismatch_raises(self, mock_lpips):
        """Test temporal masks reject reset events without a time dimension."""

        mock_lpips.return_value = MockLPIPS()
        y_true, y_pred = self._make_sequence(timesteps=3)
        y_pred["temporal_params"] = torch.zeros((1, 3, 2, 2, 2))
        y_pred["reset_event"] = torch.ones((1,))
        loss_args = {
            "alpha_reg_weight": 1.0,
            "change_pred_weight": 0.0,
            "first_frame_weight": 0.0,
            "temporal_reg_weight": 0.0,
            "theta_reg_weight": 0.0,
        }
        criterion = LossV1(3, torch.device("cpu"), loss_args)

        with self.assertRaisesRegex(ValueError, "reset_event rank"):
            criterion(y_true, y_pred)

        y_pred["reset_event"] = torch.tensor(1.0)
        with self.assertRaisesRegex(ValueError, "reset_event rank"):
            criterion(y_true, y_pred)
