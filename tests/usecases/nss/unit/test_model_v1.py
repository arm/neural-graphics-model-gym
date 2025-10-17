# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0


import unittest

import torch

from ng_model_gym.core.data.utils import ToneMapperMode
from ng_model_gym.core.model.model_factory import create_model
from ng_model_gym.core.utils.types import TrainEvalMode
from ng_model_gym.usecases.nss.model.model_blocks import AutoEncoderV1
from tests.base_gpu_test import BaseGPUMemoryTest
from tests.testing_utils import create_simple_params


class TestNSSModelV1(BaseGPUMemoryTest):
    """Tests for NSSModel class"""

    def test_forward_pass_golden_values(self):
        """Test forward pass against known golden values"""
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        device = torch.device("cuda")

        forward_inputs = torch.load(
            "tests/usecases/nss/unit/data/nss_v1_golden_values/forward_pass_inputs_golden.pt",
            map_location=device,
            weights_only=True,
        )

        forward_outputs = torch.load(
            "tests/usecases/nss/unit/data/nss_v1_golden_values/forward_pass_output_golden.pt",
            map_location=device,
            weights_only=True,
        )["output"]

        autoencoder = AutoEncoderV1()
        autoencoder.load_state_dict(forward_inputs["autoencoder_state"])
        autoencoder.to(device)

        params = create_simple_params()
        params.dataset.recurrent_samples = None
        params.dataset.tonemapper = ToneMapperMode.REINHARD
        params.model_train_eval_mode = TrainEvalMode.FP32

        nss_model = create_model(params, device)
        nss_model.autoencoder = autoencoder

        model_outputs = nss_model(forward_inputs["inputs"])

        expected_output = forward_outputs["output"]

        RTOL = 1e-3
        ATOL = 1e-3

        torch.testing.assert_close(
            model_outputs["output"], expected_output, rtol=RTOL, atol=ATOL
        )

        expected_output_linear = forward_outputs["output_linear"]
        torch.testing.assert_close(
            model_outputs["output_linear"],
            expected_output_linear,
            rtol=RTOL,
            atol=ATOL,
        )

        expected_out_filtered = forward_outputs["out_filtered"]
        torch.testing.assert_close(
            model_outputs["out_filtered"],
            expected_out_filtered,
            rtol=RTOL,
            atol=ATOL,
        )

        expected_feedback = forward_outputs["feedback"]
        torch.testing.assert_close(
            model_outputs["feedback"], expected_feedback, rtol=RTOL, atol=ATOL
        )

        expected_derivative = forward_outputs["derivative"]
        torch.testing.assert_close(
            model_outputs["derivative"], expected_derivative, rtol=RTOL, atol=ATOL
        )

        expected_depth_dilated = forward_outputs["depth_dilated"]
        torch.testing.assert_close(
            model_outputs["depth_dilated"],
            expected_depth_dilated,
            rtol=RTOL,
            atol=ATOL,
        )

        expected_ground_truth = forward_outputs["ground_truth"]
        torch.testing.assert_close(
            model_outputs["ground_truth"],
            expected_ground_truth,
            rtol=RTOL,
            atol=ATOL,
        )

        expected_input_color = forward_outputs["input_color"]
        torch.testing.assert_close(
            model_outputs["input_color"], expected_input_color, rtol=RTOL, atol=ATOL
        )


if __name__ == "__main__":
    unittest.main()
