# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch
from torch import nn

from ng_model_gym.usecases.nss.model.model_blocks import AutoEncoderV1

FEEDBACK_CH = 4


class TestAutoEncoder(unittest.TestCase):
    """Tests AutoEncoder class."""

    def setUp(self):
        """Set up the test class."""
        self.autoencoder_model = AutoEncoderV1()

        # Set all weights and biases to fixed values
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

        self.autoencoder_model.apply(init_weights)

        # Create a fixed input tensor with known values
        self.input_tensor = torch.ones((1, 12, 64, 64))

    def test_output_shapes(self):
        """Test if the AutoEncoder produces the correct output shape"""
        # Run the forward pass
        (k0, k1, k2, k3), temporal_params, feedback_out = self.autoencoder_model(
            self.input_tensor
        )

        # Check output shapes
        self.assertEqual(k0.shape, (1, 4, 64, 64))
        self.assertEqual(k1.shape, (1, 4, 64, 64))
        self.assertEqual(k2.shape, (1, 4, 64, 64))
        self.assertEqual(k3.shape, (1, 4, 64, 64))
        self.assertEqual(temporal_params.shape, (1, 4, 64, 64))
        self.assertEqual(feedback_out.shape, (1, FEEDBACK_CH, 64, 64))

    def test_forward_pass_consistency(self):
        """Test if the AutoEncoder produces stable values"""
        # Run the forward pass twice
        (
            (k0_a, k1_a, k2_a, k3_a),
            temporal_params1,
            feedback_out1,
        ) = self.autoencoder_model(self.input_tensor)
        (
            (k0_b, k1_b, k2_b, k3_b),
            temporal_params2,
            feedback_out2,
        ) = self.autoencoder_model(self.input_tensor)

        # Check Kernel is the same between runs
        self.assertTrue(torch.allclose(k0_a, k0_b, atol=1e-6))
        self.assertTrue(torch.allclose(k1_a, k1_b, atol=1e-6))
        self.assertTrue(torch.allclose(k2_a, k2_b, atol=1e-6))
        self.assertTrue(torch.allclose(k3_a, k3_b, atol=1e-6))

        # Check that the feedback output is identical
        self.assertTrue(
            torch.allclose(feedback_out1, feedback_out2, atol=1e-6),
            "Feedback changed between runs",
        )

        # Check that the temporal output is identical
        self.assertTrue(
            torch.allclose(temporal_params1, temporal_params2, atol=1e-6),
            "Temporal changed between runs",
        )

    def test_output_values(self):
        """Test output values matches expected values"""
        (k0, k1, k2, k3), temporal_params, feedback_out = self.autoencoder_model(
            self.input_tensor
        )

        expected_value = torch.sigmoid(torch.tensor(0.1)).item()

        self.assertAlmostEqual(k0.flatten()[0].item(), expected_value, places=6)
        self.assertAlmostEqual(k1.flatten()[0].item(), expected_value, places=6)
        self.assertAlmostEqual(k2.flatten()[0].item(), expected_value, places=6)
        self.assertAlmostEqual(k3.flatten()[0].item(), expected_value, places=6)
        self.assertAlmostEqual(
            temporal_params.flatten()[0].item(), expected_value, places=6
        )
        self.assertAlmostEqual(
            feedback_out.flatten()[0].item(), expected_value, places=6
        )

    def test_model_training(self):
        """Test we can do backward pass"""
        (k0, k1, k2, k3), temporal_params, feedback_out = self.autoencoder_model(
            self.input_tensor
        )
        loss = (
            k0.mean()
            + k1.mean()
            + k2.mean()
            + k3.mean()
            + temporal_params.mean()
            + feedback_out.mean()
        )
        loss.backward()

        # Check that gradients are here
        for param in self.autoencoder_model.parameters():
            self.assertIsNotNone(param.grad)

    def test_zero_input(self):
        """Sanity check for output"""
        zero_input = torch.randn((1, 12, 64, 64))
        (k0, k1, k2, k3), temporal_params, feedback_out = self.autoencoder_model(
            zero_input
        )

        # Should not have NaN or Inf in any output
        self.assertFalse(torch.isnan(k0).any())
        self.assertFalse(torch.isinf(k0).any())
        self.assertFalse(torch.isnan(k1).any())
        self.assertFalse(torch.isinf(k1).any())
        self.assertFalse(torch.isnan(k2).any())
        self.assertFalse(torch.isinf(k2).any())
        self.assertFalse(torch.isnan(k3).any())
        self.assertFalse(torch.isinf(k3).any())
        self.assertFalse(torch.isnan(temporal_params).any())
        self.assertFalse(torch.isinf(temporal_params).any())
        self.assertFalse(torch.isnan(feedback_out).any())
        self.assertFalse(torch.isinf(feedback_out).any())


class TestAutoEncoderGolden(unittest.TestCase):
    """Test autoencoder implementation against known inputs and outputs"""

    def test_forward_pass(self):
        """Test autoencoder forward pass"""
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        device = torch.device("cuda")

        autoencoder_input = torch.load(
            "tests/nss/unit/data/nss_v1_golden_values/autoencoder_input_golden.pt",
            map_location=device,
            weights_only=True,
        )

        autoencoder_output_tensors = torch.load(
            "tests/nss/unit/data/nss_v1_golden_values/autoencoder_output_golden.pt",
            map_location=device,
            weights_only=True,
        )

        autoencoder = AutoEncoderV1()
        autoencoder.load_state_dict(autoencoder_output_tensors["autoencoder_model"])
        autoencoder.to(device)
        autoencoder_input_tensors = autoencoder_input["input_tensor"]
        (k0, k1, k2, k3), temporal_params, feedback_out = autoencoder(
            autoencoder_input_tensors
        )

        expected_k0 = autoencoder_output_tensors["kernels"][0]
        self.assertTrue(torch.allclose(k0, expected_k0, rtol=1e-3, atol=1e-3))

        expected_k1 = autoencoder_output_tensors["kernels"][1]
        self.assertTrue(torch.allclose(k1, expected_k1, rtol=1e-3, atol=1e-3))

        expected_k2_tp = autoencoder_output_tensors["kernels"][2]
        self.assertTrue(torch.allclose(k2, expected_k2_tp, rtol=1e-3, atol=1e-3))

        expected_k3 = autoencoder_output_tensors["kernels"][3]
        self.assertTrue(torch.allclose(k3, expected_k3, rtol=1e-3, atol=1e-3))

        expected_temporal = autoencoder_output_tensors["temporal_params"]
        self.assertTrue(
            torch.allclose(temporal_params, expected_temporal, rtol=1e-3, atol=1e-3)
        )

        expected_feedback_out = autoencoder_output_tensors["feedback"]
        self.assertTrue(
            torch.allclose(feedback_out, expected_feedback_out, rtol=1e-3, atol=1e-3)
        )


if __name__ == "__main__":
    unittest.main()
