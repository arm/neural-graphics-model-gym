# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from ng_model_gym.core.loss.losses import LossV1


class TestLossV1(unittest.TestCase):
    """Tests for LossV1 class"""

    def test_loss_function(self):
        """Test loss function against golden value"""
        device = torch.device("cuda")
        # Load loss files from test
        loss_input = torch.load(
            "tests/nss/unit/data/nss_v1_golden_values/loss_golden.pt",
            map_location=device,
            weights_only=True,
        )

        y_true = loss_input["y_true"]
        y_pred_and_inps = loss_input["y_pred_and_inps"]
        recurrent_samples = y_true.shape[1]
        self.assertEqual(recurrent_samples, 4)
        criterion = LossV1(recurrent_samples, device)

        loss, _ = criterion(y_true, y_pred_and_inps)
        self.assertAlmostEqual(loss.item(), loss_input["loss"], places=3)


if __name__ == "__main__":
    unittest.main()
