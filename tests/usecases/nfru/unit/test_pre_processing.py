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
class TestPreProcess(BaseGPUMemoryTest):
    """Tests for NFRU usecase preprocessing."""

    def setUp(self):
        """Set up pre-processing inputs."""
        super().setUp()

        self.assertTrue(
            torch.cuda.is_available(),
            "PreProcess tests currently require GPU build of torch",
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
        depth_m1 = torch.rand(batch_size, 1, height // 2, width // 2)
        depth_p1 = torch.rand(batch_size, 1, height // 2, width // 2)
        mv_t_f30_m1 = torch.rand(batch_size, 2, height // 2, width // 2)
        flow_t_f30_xx = torch.rand(batch_size, 2, height // 4, width // 4)
        depth_p1_warp_t = torch.rand(batch_size, 1, height // 2, width // 2)
        depth_p1_warp_p1 = torch.rand(batch_size, 1, height // 2, width // 2)
        motion_mat_m1p1 = torch.rand(batch_size, 4, 4)
        motion_mat_p1m1 = torch.rand(batch_size, 4, 4)
        depth_params = torch.rand(batch_size, 4, 1, 1)

        timestep = 0.5

        self.inputs = {
            "rgb_m1": rgb_m1,
            "rgb_p1": rgb_p1,
            "depth_m1": depth_m1,
            "depth_p1": depth_p1,
            "mv_t_f30_m1": mv_t_f30_m1,
            "flow_t_f30_xx": flow_t_f30_xx,
            "depth_p1_warp_t": depth_p1_warp_t,
            "depth_p1_warp_p1": depth_p1_warp_p1,
            "motion_mat_m1p1": motion_mat_m1p1,
            "motion_mat_p1m1": motion_mat_p1m1,
            "depth_params": depth_params,
            "timestep": timestep,
        }

        # Move input tensors to GPU
        self.inputs = {
            k: (v.cuda() if isinstance(v, torch.Tensor) else v)
            for k, v in self.inputs.items()
        }

    def test_preprocess_returns_values(self):
        """Test NFRU preprocess returns valid network input tensor."""
        preprocess_input = self.inputs

        with torch.no_grad():
            network_in = self.network._pre_process(
                flow_t_f30_xx=preprocess_input["flow_t_f30_xx"],
                mv_t_f30_m1=preprocess_input["mv_t_f30_m1"],
                rgb_m1=preprocess_input["rgb_m1"],
                rgb_p1=preprocess_input["rgb_p1"],
                depth_m1=preprocess_input["depth_m1"],
                depth_p1=preprocess_input["depth_p1"],
                depth_p1_warp_t=preprocess_input["depth_p1_warp_t"],
                depth_p1_warp_p1=preprocess_input["depth_p1_warp_p1"],
                motion_mat_m1p1=preprocess_input["motion_mat_m1p1"],
                motion_mat_p1m1=preprocess_input["motion_mat_p1m1"],
                depth_params=preprocess_input["depth_params"],
                timestep=float(preprocess_input["timestep"]),
            )

        self.assertIsNotNone(network_in)

        self.assertEqual(network_in.dtype, torch.float32)
        self.assertEqual(network_in.device.type, self.device.type)
        self.assertEqual(
            network_in.shape,
            (
                self.batch_size,
                self.network.in_ch,
                self.height // 4,
                self.width // 4,
            ),
        )

    def test_preprocess_shader_acc_returns_values(self):
        """Test NFRU shader accurate preprocess returns valid network input tensor."""
        self.network.shader_accurate = True

        preprocess_input = self.inputs

        with torch.no_grad():
            network_in = self.network._pre_process(
                flow_t_f30_xx=preprocess_input["flow_t_f30_xx"],
                mv_t_f30_m1=preprocess_input["mv_t_f30_m1"],
                rgb_m1=preprocess_input["rgb_m1"],
                rgb_p1=preprocess_input["rgb_p1"],
                depth_m1=preprocess_input["depth_m1"],
                depth_p1=preprocess_input["depth_p1"],
                depth_p1_warp_t=preprocess_input["depth_p1_warp_t"],
                depth_p1_warp_p1=preprocess_input["depth_p1_warp_p1"],
                motion_mat_m1p1=preprocess_input["motion_mat_m1p1"],
                motion_mat_p1m1=preprocess_input["motion_mat_p1m1"],
                depth_params=preprocess_input["depth_params"],
                timestep=float(preprocess_input["timestep"]),
            )

        self.assertIsNotNone(network_in)

        self.assertEqual(network_in.dtype, torch.float32)
        self.assertEqual(network_in.device.type, self.device.type)
        self.assertEqual(
            network_in.shape,
            (
                self.batch_size,
                self.network.in_ch,
                self.height // 4,
                self.width // 4,
            ),
        )


@unittest.skip("NFRU CI/assets disabled for now")
class TestNFRUPreprocessGolden(BaseGPUMemoryTest):
    """Test NFRU preprocess implementation against known inputs and outputs."""

    def setUp(self):
        super().setUp()

        self.assertTrue(
            torch.cuda.is_available(),
            "PreProcess tests currently require GPU build of torch",
        )

        self.device = torch.device("cuda")

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        self.params = create_simple_params(usecase="nfru")
        self.params.model_train_eval_mode = TrainEvalMode.FP32

        self.model = create_model(self.params, self.device)

        self.network = self.model.network
        self.network.shader_accurate = False

    def test_preprocess(self):
        """Test preprocess implementation against known inputs and outputs."""

        preprocess_input = torch.load(
            _GOLDEN_ROOT / "preprocess_inputs_golden.pt",
            map_location=self.device,
            weights_only=True,
        )

        with torch.no_grad():
            network_in = self.network._pre_process(
                flow_t_f30_xx=preprocess_input["flow_t_f30_xx"],
                mv_t_f30_m1=preprocess_input["mv_t_f30_m1"],
                rgb_m1=preprocess_input["rgb_m1"],
                rgb_p1=preprocess_input["rgb_p1"],
                depth_m1=preprocess_input["depth_m1"],
                depth_p1=preprocess_input["depth_p1"],
                depth_p1_warp_t=preprocess_input["depth_p1_warp_t"],
                depth_p1_warp_p1=preprocess_input["depth_p1_warp_p1"],
                motion_mat_m1p1=preprocess_input["motion_mat_m1p1"],
                motion_mat_p1m1=preprocess_input["motion_mat_p1m1"],
                depth_params=preprocess_input["depth_params"],
                timestep=float(preprocess_input["timestep"]),
                random_seed=int(preprocess_input["random_seed"].item()),
            )

        preprocess_output = torch.load(
            _GOLDEN_ROOT / "preprocess_output_golden.pt",
            map_location=self.device,
            weights_only=True,
        )

        RTOL = 1e-3
        ATOL = 1e-3

        expected_network_in = preprocess_output["network_in"]

        torch.testing.assert_close(
            network_in,
            expected_network_in,
            rtol=RTOL,
            atol=ATOL,
        )


@unittest.skip("NFRU CI/assets disabled for now")
class TestShaderAccPreprocessGolden(BaseGPUMemoryTest):
    """Test shader accurate NFRU preprocess implementation against known inputs and outputs."""

    def setUp(self):
        super().setUp()

        self.assertTrue(
            torch.cuda.is_available(),
            "PreProcess tests currently require GPU build of torch",
        )

        self.device = torch.device("cuda")

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        self.params = create_simple_params(usecase="nfru")
        self.params.model_train_eval_mode = TrainEvalMode.FP32

        self.model = create_model(self.params, self.device)

        self.network = self.model.network
        self.network.shader_accurate = True

    def test_shader_acc_preprocess(self):
        """Test shader accurate preprocess implementation."""

        preprocess_input = torch.load(
            _GOLDEN_ROOT / "preprocess_inputs_golden_shader_acc.pt",
            map_location=self.device,
            weights_only=True,
        )

        with torch.no_grad():
            network_in = self.network._pre_process(
                flow_t_f30_xx=preprocess_input["flow_t_f30_xx"],
                mv_t_f30_m1=preprocess_input["mv_t_f30_m1"],
                rgb_m1=preprocess_input["rgb_m1"],
                rgb_p1=preprocess_input["rgb_p1"],
                depth_m1=preprocess_input["depth_m1"],
                depth_p1=preprocess_input["depth_p1"],
                depth_p1_warp_t=preprocess_input["depth_p1_warp_t"],
                depth_p1_warp_p1=preprocess_input["depth_p1_warp_p1"],
                motion_mat_m1p1=preprocess_input["motion_mat_m1p1"],
                motion_mat_p1m1=preprocess_input["motion_mat_p1m1"],
                depth_params=preprocess_input["depth_params"],
                timestep=float(preprocess_input["timestep"]),
                random_seed=int(preprocess_input["random_seed"].item()),
            )

        preprocess_output = torch.load(
            _GOLDEN_ROOT / "preprocess_output_golden_shader_acc.pt",
            map_location=self.device,
            weights_only=True,
        )

        RTOL = 1e-3
        ATOL = 1e-3

        expected_network_in = preprocess_output["network_in"]

        torch.testing.assert_close(
            network_in,
            expected_network_in,
            rtol=RTOL,
            atol=ATOL,
        )
