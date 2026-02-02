# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from ng_model_gym.core.loss import LossV1


class TestLossV1(unittest.TestCase):
    """Tests for LossV1 class"""

    def test_loss_function(self):
        """Test loss function against golden value"""
        device = torch.device("cuda")
        # Load loss files from test
        loss_input = torch.load(
            "tests/usecases/nss/unit/data/nss_v1_golden_values/loss_golden.pt",
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
        criterion = LossV1(recurrent_samples, device)

        loss = criterion(y_true, y_pred)
        self.assertAlmostEqual(loss.item(), loss_input["loss"], places=3)
