# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest
from pathlib import Path

import torch

from ng_model_gym.core.model import create_model
from ng_model_gym.core.utils.enum_definitions import TrainEvalMode
from tests.base_gpu_test import BaseGPUMemoryTest
from tests.testing_utils import create_simple_params

_GOLDEN_ROOT = Path(__file__).resolve().parent / "data" / "nfru_v1_golden_values"

# pylint: disable=duplicate-code


@unittest.skip("NFRU CI/assets disabled for now")
class TestPostProcess(BaseGPUMemoryTest):
    """Tests for NFRU usecase postprocessing."""

    def setUp(self):
        """Set up post-processing config."""
        super().setUp()

        self.assertTrue(
            torch.cuda.is_available(),
            "Postprocess tests currently require GPU build of torch",
        )

        self.device = torch.device("cuda")

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        self.params = create_simple_params(usecase="nfru")
        self.params.model_train_eval_mode = TrainEvalMode.FP32

        self.model = create_model(self.params, self.device)

        self.network = self.model.network
        self.network.shader_accurate = False

        self.batch_size = batch_size = 4
        self.height = height = 64
        self.width = width = 64

        rgb_m1 = torch.rand(batch_size, 3, height, width)
        rgb_p1 = torch.rand(batch_size, 3, height, width)
        mv_t_f30_m1 = torch.rand(batch_size, 2, height // 2, width // 2)
        flow_t_f30_xx = torch.rand(batch_size, 2, height // 4, width // 4)
        learnt_params = torch.rand(batch_size, 4, height // 4, width // 4)

        timestep = 0.5

        self.inputs = {
            "rgb_m1": rgb_m1,
            "rgb_p1": rgb_p1,
            "mv_t_f30_m1": mv_t_f30_m1,
            "flow_t_f30_xx": flow_t_f30_xx,
            "learnt_params": learnt_params,
            "timestep": timestep,
        }

        # Move input tensors to GPU
        self.inputs = {
            k: (v.cuda() if isinstance(v, torch.Tensor) else v)
            for k, v in self.inputs.items()
        }

    def test_postprocess_output_shape(self):
        """Test postprocess returns value and check output shape."""

        postprocess_input = self.inputs

        with torch.no_grad():
            output = self.network._post_process(
                flow_t_f30_xx=postprocess_input["flow_t_f30_xx"],
                mv_t_f30_m1=postprocess_input["mv_t_f30_m1"],
                rgb_m1=postprocess_input["rgb_m1"],
                rgb_p1=postprocess_input["rgb_p1"],
                learnt_params=postprocess_input["learnt_params"],
                timestep=float(postprocess_input["timestep"]),
            )

        self.assertIsNotNone(output)

        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.device.type, self.device.type)

        self.assertFalse(torch.isnan(output).any())

        self.assertEqual(
            output.shape,
            (
                self.batch_size,
                3,
                self.height,
                self.width,
            ),
        )


@unittest.skip("NFRU CI/assets disabled for now")
class TestPostprocessGolden(BaseGPUMemoryTest):
    """Test postprocess implementation against known inputs and outputs"""

    def setUp(self):
        super().setUp()

        self.assertTrue(
            torch.cuda.is_available(),
            "Postprocess tests currently require GPU build of torch",
        )

        self.device = torch.device("cuda")

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        self.params = create_simple_params(usecase="nfru")
        self.params.model_train_eval_mode = TrainEvalMode.FP32

        self.model = create_model(self.params, self.device)

        self.network = self.model.network
        self.network.shader_accurate = False

    def test_postprocess(self):
        """Test postprocess implementation"""

        postprocess_input = torch.load(
            _GOLDEN_ROOT / "postprocess_inputs_golden.pt",
            map_location=self.device,
            weights_only=True,
        )

        with torch.no_grad():
            output = self.network._post_process(
                flow_t_f30_xx=postprocess_input["flow_t_f30_xx"],
                mv_t_f30_m1=postprocess_input["mv_t_f30_m1"],
                rgb_m1=postprocess_input["rgb_m1"],
                rgb_p1=postprocess_input["rgb_p1"],
                learnt_params=postprocess_input["learnt_params"],
                timestep=postprocess_input["timestep"],
            )

        postprocess_output = torch.load(
            _GOLDEN_ROOT / "postprocess_output_golden.pt",
            map_location=self.device,
            weights_only=True,
        )

        RTOL = 1e-3
        ATOL = 1e-3

        expected_output = postprocess_output["output"]

        torch.testing.assert_close(output, expected_output, rtol=RTOL, atol=ATOL)
