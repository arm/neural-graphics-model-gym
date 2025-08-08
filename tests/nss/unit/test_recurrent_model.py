# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from ng_model_gym.nss.model.model_blocks import AutoEncoderV1
from ng_model_gym.nss.model.model_v1 import create_feedback_model_with_nss
from ng_model_gym.utils.types import TrainEvalMode
from tests.unit.utils.utils import create_simple_params


class TestFeedbackModel(unittest.TestCase):
    """Tests for FeedbackModel class."""

    def setUp(self):
        """Setup feedback model."""
        params = create_simple_params(dataset="")
        params.model_train_eval_mode = TrainEvalMode.FP32
        params.dataset.gt_augmentation = True
        params.train.batch_size = 2
        params.dataset.recurrent_samples = 4
        self.device = torch.device("cuda")
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        self.model = create_feedback_model_with_nss(params, self.device)
        self.batch = params.train.batch_size
        self.recurrence = params.dataset.recurrent_samples
        self.data = {
            "colour_linear": torch.randn(self.batch, self.recurrence, 3, 128, 128),
            "depth": torch.randn(self.batch, self.recurrence, 1, 128, 128),
            "depth_params": torch.randn(self.batch, self.recurrence, 4, 128, 128),
            "ground_truth_linear": torch.randn(
                self.batch, self.recurrence, 3, 256, 256
            ),
            "jitter": torch.randn(self.batch, self.recurrence, 2, 1, 1),
            "motion": torch.randn(self.batch, self.recurrence, 2, 256, 256),
            "render_size": torch.randn(self.batch, self.recurrence, 2, 1, 1),
            "zNear": torch.randn(self.batch, self.recurrence, 1, 1, 1),
            "zFar": torch.randn(self.batch, self.recurrence, 1, 1, 1),
            "seq": torch.randn(self.batch, self.recurrence, 1, 1, 1),
            "exposure": torch.randn(self.batch, self.recurrence, 1, 1, 1),
            "colour": torch.randn(self.batch, self.recurrence, 3, 128, 128),
        }

        self.data = {key: tensor.to(self.device) for key, tensor in self.data.items()}

    def test_shape_feedback_model_forward_pass(self):
        """Test feedback model training shape"""
        with torch.no_grad():
            self.model.train()
            model_out = self.model(self.data)

        self.assertEqual(
            model_out["output_linear"].shape, (self.batch, self.recurrence, 3, 256, 256)
        )
        self.assertEqual(
            model_out["output"].shape, (self.batch, self.recurrence, 3, 256, 256)
        )
        self.assertEqual(
            model_out["feedback"].shape, (self.batch, self.recurrence, 4, 128, 128)
        )
        self.assertEqual(
            model_out["derivative"].shape, (self.batch, self.recurrence, 2, 128, 128)
        )
        self.assertEqual(
            model_out["depth_dilated"].shape, (self.batch, self.recurrence, 1, 128, 128)
        )

    def test_golden_feedback_model_forward(self):
        """Test feedback model training"""

        feedback_input_golden = torch.load(
            "tests/nss/unit/data/nss_v1_golden_values/feedback_input_golden.pt",
            map_location=self.device,
            weights_only=True,
        )

        feedback_output_golden = torch.load(
            "tests/nss/unit/data/nss_v1_golden_values/feedback_output_golden.pt",
            map_location=self.device,
            weights_only=True,
        )["outputs"]

        self.model.train()

        autoencoder_with_golden_state = AutoEncoderV1()
        autoencoder_with_golden_state.load_state_dict(
            feedback_input_golden["autoencoder_state"]
        )
        autoencoder_with_golden_state.to(self.device)
        self.model.nss_model.autoencoder = autoencoder_with_golden_state

        model_out = self.model(feedback_input_golden["feedback_input"])

        tolerance = 1e-3
        expected_output_linear = feedback_output_golden["output_linear"]
        torch.testing.assert_close(
            model_out["output_linear"],
            expected_output_linear,
            rtol=tolerance,
            atol=tolerance,
        )

        expected_output = feedback_output_golden["output"]
        self.assertTrue(
            torch.allclose(
                model_out["output"], expected_output, rtol=tolerance, atol=tolerance
            )
        )

        expected_feedback = feedback_output_golden["feedback"]
        torch.testing.assert_close(
            model_out["feedback"], expected_feedback, rtol=tolerance, atol=tolerance
        )

        expected_derivative = feedback_output_golden["derivative"]
        torch.testing.assert_close(
            model_out["derivative"],
            expected_derivative,
            rtol=tolerance,
            atol=tolerance,
        )

        expected_depth_dilated = feedback_output_golden["depth_dilated"]
        torch.testing.assert_close(
            model_out["depth_dilated"],
            expected_depth_dilated,
            rtol=tolerance,
            atol=tolerance,
        )


if __name__ == "__main__":
    unittest.main()
