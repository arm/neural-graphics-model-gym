# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from ng_model_gym.nss.model.pre_processing import (
    pre_process_v1_bwd,
    pre_process_v1_sa_bwd,
    PreProcessV1,
    PreProcessV1_ShaderAccurate,
)


class TestPreProcess(unittest.TestCase):
    """Tests for PreProcess class in PyTorch."""

    def setUp(self):
        """Set up pre-processing inputs."""

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        self.assertTrue(
            torch.cuda.is_available(),
            "PreProcess tests currently require GPU build of torch",
        )

        self.device = torch.device("cuda")

        self.preprocess = PreProcessV1()
        self.shader_acc_preprocess = PreProcessV1_ShaderAccurate()

        self.batch_size = batch_size = 4

        colour_linear = torch.rand(batch_size, 3, 64, 64)
        history = torch.rand(batch_size, 3, 128, 128)
        motion = torch.rand(batch_size, 2, 128, 128)
        depth = torch.rand(batch_size, 1, 64, 64)
        depth_tm1 = torch.rand(batch_size, 1, 64, 64)
        true_depth_tm1 = torch.rand(batch_size, 1, 64, 64)
        jitter = torch.rand(batch_size, 2, 1, 1)
        jitter_tm1 = torch.rand(batch_size, 2, 1, 1)
        feedback_tm1 = torch.rand(batch_size, 1, 64, 64)
        derivative_tm1 = torch.rand(batch_size, 2, 64, 64)
        depth_params = torch.rand(batch_size, 4, 64, 64)
        exposure = torch.rand(batch_size, 1, 1, 1)
        render_size = torch.tensor(
            (batch_size, 3, 64, 64), dtype=torch.float32
        )  # Expects fp32 render_size
        dm_scale = torch.tensor([0.5])

        self.inputs = {
            "colour_linear": colour_linear,
            "history": history,
            "motion": motion,
            "motion_lr": motion,
            "depth": depth,
            "depth_tm1": depth_tm1,
            "true_depth_tm1": true_depth_tm1,
            "jitter": jitter,
            "jitter_tm1": jitter_tm1,
            "feedback_tm1": feedback_tm1,
            "derivative_tm1": derivative_tm1,
            "depth_params": depth_params,
            "exposure": exposure,
            "render_size": render_size,
            "dm_scale": dm_scale,
        }

        # Used in backprop
        self.out_luma = torch.randn(batch_size, 1, 1, 1, device=self.device)
        self.out_tensor = torch.randn(batch_size, 3, 64, 64, device=self.device)

        # Move Input Tensors to GPU
        self.inputs = {
            k: (v.cuda() if isinstance(v, torch.Tensor) else v)
            for k, v in self.inputs.items()
        }

    def test_fwd_pass_returns_values(self):
        """Test forward pass can run and return some values."""
        input_tensor, derivative, depth_dilated = self.preprocess.apply(
            self.inputs["colour_linear"],
            self.inputs["history"],
            self.inputs["motion"],
            self.inputs["depth"],
            self.inputs["depth_tm1"],
            self.inputs["jitter"],
            self.inputs["jitter_tm1"],
            self.inputs["feedback_tm1"],
            self.inputs["derivative_tm1"],
            self.inputs["depth_params"],
            self.inputs["exposure"],
            self.inputs["render_size"],
            self.inputs["dm_scale"],
        )

        self.assertIsNotNone(input_tensor)
        self.assertIsNotNone(derivative)
        self.assertIsNotNone(depth_dilated)

    def test_shader_acc_fwd_pass_returns_values(self):
        """Test shader accurate forward pass can run and return some values."""
        input_tensor, derivative, depth_dilated = self.shader_acc_preprocess.apply(
            self.inputs["colour_linear"],
            self.inputs["history"],
            self.inputs["motion_lr"],
            self.inputs["depth"],
            self.inputs["true_depth_tm1"],
            self.inputs["depth_tm1"],
            self.inputs["jitter"],
            self.inputs["jitter_tm1"],
            self.inputs["feedback_tm1"],
            self.inputs["derivative_tm1"],
            self.inputs["depth_params"],
            self.inputs["exposure"],
            self.inputs["render_size"],
            self.inputs["dm_scale"],
        )

        self.assertIsNotNone(input_tensor)
        self.assertIsNotNone(derivative)
        self.assertIsNotNone(depth_dilated)

    def test_bwd_pass_returns_values(self):
        """Test backward pass can run and return some values."""
        grad_history, grad_feedback_tm1, grad_dm_scale = pre_process_v1_bwd(
            self.inputs["colour_linear"],
            self.inputs["history"],
            self.inputs["motion"],
            self.inputs["depth"],
            self.inputs["depth_tm1"],
            self.inputs["jitter"],
            self.inputs["jitter_tm1"],
            self.inputs["feedback_tm1"],
            self.inputs["derivative_tm1"],
            self.inputs["depth_params"],
            self.inputs["exposure"],
            self.inputs["render_size"],
            self.inputs["dm_scale"],
            output_tensor=[
                self.out_tensor,
                self.out_tensor,
            ],  # Putting same value for output and its derivative
            out_luma_derivative=self.out_luma,
            out_depth_t=torch.rand_like(self.inputs["depth"]),
        )

        self.assertIsNotNone(grad_history)
        self.assertIsNotNone(grad_feedback_tm1)
        self.assertIsNotNone(grad_dm_scale)

    def test_shader_acc_bwd_pass_returns_values(self):
        """Test shader accurate backward pass can run and return some values."""
        grad_history, grad_feedback_tm1, grad_dm_scale = pre_process_v1_sa_bwd(
            self.inputs["colour_linear"],
            self.inputs["history"],
            self.inputs["motion_lr"],
            self.inputs["depth"],
            self.inputs["true_depth_tm1"],
            self.inputs["depth_tm1"],
            self.inputs["jitter"],
            self.inputs["jitter_tm1"],
            self.inputs["feedback_tm1"],
            self.inputs["derivative_tm1"],
            self.inputs["depth_params"],
            self.inputs["exposure"],
            self.inputs["render_size"],
            self.inputs["dm_scale"],
            output_tensor=[
                self.out_tensor,
                self.out_tensor,
            ],  # Putting same value for output and its derivative
            out_luma_derivative=self.out_luma,
            out_depth_t=torch.rand_like(self.inputs["depth"]),
        )

        self.assertIsNotNone(grad_history)
        self.assertIsNotNone(grad_feedback_tm1)
        self.assertIsNotNone(grad_dm_scale)


class TestPreprocessGolden(unittest.TestCase):
    """Test preprocess implementation against known inputs and outputs"""

    def test_preprocess(self):
        """Test preprocess implementation"""
        device = torch.device("cuda")

        preprocess_input = torch.load(
            "tests/nss/unit/data/nss_v1_golden_values/preprocess_inputs_golden.pt",
            map_location=device,
            weights_only=True,
        )

        preprocess_output = torch.load(
            "tests/nss/unit/data/nss_v1_golden_values/preprocess_output_golden.pt",
            map_location=device,
            weights_only=True,
        )

        input_tensor, derivative, depth_dilated = PreProcessV1().apply(
            preprocess_input["colour_linear"],
            preprocess_input["history"],
            preprocess_input["motion"],
            preprocess_input["depth"],
            preprocess_input["depth_tm1"],
            preprocess_input["jitter"],
            preprocess_input["jitter_tm1"],
            preprocess_input["feedback_tm1"],
            preprocess_input["derivative_tm1"],
            preprocess_input["depth_params"],
            preprocess_input["exposure"],
            preprocess_input["render_size"],
            preprocess_input["dm_scale"],
        )

        expected_input_tensor = preprocess_output["input_tensor"]
        self.assertTrue(
            torch.allclose(input_tensor, expected_input_tensor, rtol=1e-4, atol=1e-4)
        )

        expected_derivative = preprocess_output["derivative"]
        self.assertTrue(
            torch.allclose(derivative, expected_derivative, rtol=1e-4, atol=1e-4)
        )

        expected_depth_dilated = preprocess_output["depth_dilated"]

        self.assertTrue(
            torch.allclose(depth_dilated, expected_depth_dilated, rtol=1e-4, atol=1e-4)
        )


class TestShaderAccPreprocessGolden(unittest.TestCase):
    """Test shader accurate preprocess implementation against known inputs and outputs"""

    def test_shader_acc_preprocess(self):
        """Test shader accurate preprocess implementation"""
        device = torch.device("cuda")

        preprocess_input = torch.load(
            "tests/nss/unit/data/nss_v1_golden_values/shader_acc_preprocess_inputs_golden.pt",
            map_location=device,
            weights_only=True,
        )

        preprocess_output = torch.load(
            "tests/nss/unit/data/nss_v1_golden_values/shader_acc_preprocess_output_golden.pt",
            map_location=device,
            weights_only=True,
        )

        input_tensor, derivative, depth_dilated = PreProcessV1_ShaderAccurate().apply(
            preprocess_input["colour_linear"],
            preprocess_input["history"],
            preprocess_input["motion_lr"],
            preprocess_input["depth"],
            preprocess_input["true_depth_tm1"],
            preprocess_input["depth_tm1"],
            preprocess_input["jitter"],
            preprocess_input["jitter_tm1"],
            preprocess_input["feedback_tm1"],
            preprocess_input["derivative_tm1"],
            preprocess_input["depth_params"],
            preprocess_input["exposure"],
            preprocess_input["render_size"],
            preprocess_input["dm_scale"],
        )

        expected_input_tensor = preprocess_output["input_tensor"]

        self.assertTrue(
            torch.allclose(input_tensor, expected_input_tensor, rtol=1e-4, atol=1e-4)
        )

        expected_derivative = preprocess_output["derivative"]
        self.assertTrue(
            torch.allclose(derivative, expected_derivative, rtol=1e-4, atol=1e-4)
        )

        expected_depth_dilated = preprocess_output["depth_dilated"]

        self.assertTrue(
            torch.allclose(depth_dilated, expected_depth_dilated, rtol=1e-4, atol=1e-4)
        )


if __name__ == "__main__":
    unittest.main()
