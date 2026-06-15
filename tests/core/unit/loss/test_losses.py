# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest import mock

import torch

from ng_model_gym.core.loss import LossV0_1
from ng_model_gym.core.loss.losses import LPIPSSpatialLossV1


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
