# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from ng_model_gym.nss.model.post_processing import (
    PostProcessV1,
    PostProcessV1_ShaderAccurate,
)


class TestPostProcess(unittest.TestCase):
    """Tests for PostProcess class in PyTorch."""

    def setUp(self):
        """Set up post-processing config."""
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        self.device = torch.device("cuda")
        self.batch_size = 4
        self.scale = 2.0

        self.colour = torch.rand(self.batch_size, 3, 64, 64, device=self.device)
        self.history = torch.rand(self.batch_size, 3, 128, 128, device=self.device)
        self.kpn_params = torch.rand(self.batch_size, 9, 128, 128, device=self.device)
        self.temporal_params = torch.rand(
            self.batch_size, 2, 128, 128, device=self.device
        )
        self.motion = torch.rand(self.batch_size, 2, 128, 128, device=self.device)
        self.motion_lr = self.motion
        self.exposure = torch.rand(self.batch_size, 1, 1, 1, device=self.device)
        self.jitter = torch.rand(self.batch_size, 2, 1, 1, device=self.device)

        self.offset_lut = torch.zeros(self.batch_size, 4, 2, 2, device=self.device)
        self.scale_tensor = torch.tensor(
            self.scale, dtype=torch.float32, device=self.device
        )
        self.idx_modulo = torch.tensor(2, dtype=torch.int32, device=self.device)
        self.depth_dilated = torch.rand(
            self.batch_size, 1, 128, 128, device=self.device
        )

    def test_output_shape(self):
        """Test that the output shape matches the history shape."""

        output, output_filtered = PostProcessV1.apply(
            self.colour,
            self.history,
            self.kpn_params,
            self.temporal_params,
            self.motion,
            self.exposure,
            self.jitter,
            self.offset_lut,
            self.scale_tensor,
            self.idx_modulo,
        )

        self.assertEqual(output.shape, self.history.shape)
        self.assertEqual(output_filtered.shape, self.history.shape)

    def test_shader_acc_output_shape(self):
        """Test that the output shape matches the history shape."""

        output, output_filtered = PostProcessV1_ShaderAccurate.apply(
            self.colour,
            self.history,
            self.kpn_params,
            self.temporal_params,
            self.motion_lr,
            self.depth_dilated,
            self.exposure,
            self.jitter,
            self.offset_lut,
            self.scale_tensor,
            self.idx_modulo,
        )

        self.assertEqual(output.shape, self.history.shape)
        self.assertEqual(output_filtered.shape, self.history.shape)

    def test_backward_pass(self):
        """Test that the shapes of the gradients from the backward pass
        match the input tensor shapes."""

        self.history.requires_grad_()
        self.kpn_params.requires_grad_()
        self.temporal_params.requires_grad_()

        output, _ = PostProcessV1.apply(
            self.colour,
            self.history,
            self.kpn_params,
            self.temporal_params,
            self.motion,
            self.exposure,
            self.jitter,
            self.offset_lut,
            self.scale_tensor,
            self.idx_modulo,
        )

        output_gradient = torch.ones_like(output)
        output.backward(gradient=output_gradient)

        self.assertEqual(self.history.grad.shape, self.history.shape)
        self.assertEqual(self.kpn_params.grad.shape, self.kpn_params.shape)
        self.assertEqual(self.temporal_params.grad.shape, self.temporal_params.shape)

    def test_shader_acc_backward_pass(self):
        """Test that the shapes of the gradients from the shader accurate backward pass
        match the input tensor shapes."""

        self.history.requires_grad_()
        self.kpn_params.requires_grad_()
        self.temporal_params.requires_grad_()

        output, _ = PostProcessV1_ShaderAccurate.apply(
            self.colour,
            self.history,
            self.kpn_params,
            self.temporal_params,
            self.motion_lr,
            self.depth_dilated,
            self.exposure,
            self.jitter,
            self.offset_lut,
            self.scale_tensor,
            self.idx_modulo,
        )

        output_gradient = torch.ones_like(output)
        output.backward(gradient=output_gradient)

        self.assertEqual(self.history.grad.shape, self.history.shape)
        self.assertEqual(self.kpn_params.grad.shape, self.kpn_params.shape)
        self.assertEqual(self.temporal_params.grad.shape, self.temporal_params.shape)


class TestPostprocessGolden(unittest.TestCase):
    """Test postprocess implementation against known inputs and outputs"""

    def test_postprocess(self):
        """Test postprocess implementation"""
        device = torch.device("cuda")

        postprocess_input = torch.load(
            "tests/nss/unit/data/nss_v1_golden_values/postprocess_inputs_golden.pt",
            map_location=device,
            weights_only=True,
        )

        postprocess_output = torch.load(
            "tests/nss/unit/data/nss_v1_golden_values/postprocess_output_golden.pt",
            map_location=device,
            weights_only=True,
        )

        output_linear, out_filtered = PostProcessV1.apply(
            postprocess_input["colour_linear"],
            postprocess_input["history"],
            postprocess_input["kpn_params"],
            postprocess_input["temporal_params"],
            postprocess_input["motion"],
            postprocess_input["exposure"],
            postprocess_input["jitter"],
            postprocess_input["offset_lut"],
            postprocess_input["scale"],
            postprocess_input["idx_mod"],
        )
        RTOL = 1e-3
        ATOL = 1e-3
        expected_output_linear = postprocess_output["output_linear"]
        torch.testing.assert_close(
            output_linear, expected_output_linear, rtol=RTOL, atol=ATOL
        )

        expected_out_filtered = postprocess_output["out_filtered"]
        torch.testing.assert_close(
            out_filtered, expected_out_filtered, rtol=RTOL, atol=ATOL
        )


class TestShaderAccPostprocessGolden(unittest.TestCase):
    """Test shader accurate postprocess implementation against known inputs and outputs"""

    def test_shader_acc_postprocess(self):
        """Test shader accurate postprocess implementation"""
        device = torch.device("cuda")

        postprocess_input = torch.load(
            "tests/nss/unit/data/nss_v1_golden_values/shader_acc_postprocess_inputs_golden.pt",
            map_location=device,
            weights_only=True,
        )

        postprocess_output = torch.load(
            "tests/nss/unit/data/nss_v1_golden_values/shader_acc_postprocess_output_golden.pt",
            map_location=device,
            weights_only=True,
        )

        output_linear, out_filtered = PostProcessV1_ShaderAccurate.apply(
            postprocess_input["colour_linear"],
            postprocess_input["history"],
            postprocess_input["kpn_params"],
            postprocess_input["temporal_params"],
            postprocess_input["motion_lr"],
            postprocess_input["depth_dilated"],
            postprocess_input["exposure"],
            postprocess_input["jitter"],
            postprocess_input["offset_lut"],
            postprocess_input["scale"],
            postprocess_input["idx_mod"],
        )

        RTOL = 1e-3
        ATOL = 1e-3

        expected_output_linear = postprocess_output["output_linear"]
        torch.testing.assert_close(
            output_linear, expected_output_linear, rtol=RTOL, atol=ATOL
        )

        expected_out_filtered = postprocess_output["out_filtered"]
        torch.testing.assert_close(
            out_filtered, expected_out_filtered, rtol=RTOL, atol=ATOL
        )


if __name__ == "__main__":
    unittest.main()
