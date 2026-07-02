# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code

import unittest

import torch

from ng_model_gym.core.model import create_model
from ng_model_gym.core.utils.enum_definitions import TrainEvalMode
from tests.base_gpu_test import BaseGPUMemoryTest
from tests.testing_utils import create_simple_params

_RTOL = 1e-2
_ATOL = 1e-2


class TestNSSV1ModelCoreGolden(BaseGPUMemoryTest):
    """Test NSS v1 core forward pass against known golden values."""

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required for NSS v1 Slang-backed core-forward goldens.",
    )
    def test_core_forward_pass_golden_values(self):
        """Test NSS v1 core forward pass."""

        device = torch.device("cuda")

        forward_golden_inputs = torch.load(
            "tests/usecases/nss/unit/data/nss_v1_golden_values/"
            + "nss_v1_high_forward_pass_inputs_golden.pt",
            map_location=device,
            weights_only=True,
        )

        forward_golden_outputs = torch.load(
            "tests/usecases/nss/unit/data/nss_v1_golden_values/"
            + "nss_v1_high_forward_pass_output_golden.pt",
            map_location=device,
            weights_only=True,
        )

        params = create_simple_params(usecase="nss_v1")
        params.model_train_eval_mode = TrainEvalMode.FP32
        params.model.quality = "high"
        params.model.recurrent_samples = 2
        params.train.batch_size = 2

        nss_model = create_model(params, device)
        nss_model.get_neural_network().load_state_dict(
            forward_golden_inputs["autoencoder_state"]
        )
        nss_model.eval()

        model_inputs = {
            key: value
            for key, value in forward_golden_inputs.items()
            if isinstance(value, torch.Tensor)
        }
        with torch.no_grad():
            model_outputs = nss_model.core_forward(model_inputs)

        torch.testing.assert_close(
            model_outputs["output"],
            forward_golden_outputs["output"],
            rtol=_RTOL,
            atol=_ATOL,
        )
        torch.testing.assert_close(
            model_outputs["output_linear"],
            forward_golden_outputs["output_linear"],
            rtol=_RTOL,
            atol=_ATOL,
        )
        torch.testing.assert_close(
            model_outputs["out_filtered"],
            forward_golden_outputs["out_filtered"],
            rtol=_RTOL,
            atol=_ATOL,
        )
        torch.testing.assert_close(
            model_outputs["temporal_params"],
            forward_golden_outputs["temporal_params"],
            rtol=_RTOL,
            atol=_ATOL,
        )
        torch.testing.assert_close(
            model_outputs["disocclusion_mask"],
            forward_golden_outputs["disocclusion_mask"],
            rtol=_RTOL,
            atol=_ATOL,
        )
        torch.testing.assert_close(
            model_outputs["derivative"],
            forward_golden_outputs["derivative"],
            rtol=_RTOL,
            atol=_ATOL,
        )
        torch.testing.assert_close(
            model_outputs["ground_truth"],
            forward_golden_outputs["ground_truth"],
            rtol=_RTOL,
            atol=_ATOL,
        )
        torch.testing.assert_close(
            model_outputs["input_color"],
            forward_golden_outputs["input_color"],
            rtol=_RTOL,
            atol=_ATOL,
        )

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required for NSS v1 Slang-backed core-forward backward smoke tests.",
    )
    def test_core_forward_backward_smoke(self):
        """A synthetic core-forward loss should backpropagate finite gradients."""

        device = torch.device("cuda")

        for quality in ("high", "mid"):
            with self.subTest(quality=quality):
                params = create_simple_params(usecase="nss_v1")
                params.model_train_eval_mode = TrainEvalMode.FP32
                params.model.quality = quality
                params.model.recurrent_samples = 2
                params.train.batch_size = 2

                nss_model = create_model(params, device)
                nss_model.train()
                nss_model.zero_grad(set_to_none=True)

                batch_size = params.train.batch_size
                lr_height = 128
                lr_width = 128
                hr_height = 256
                hr_width = 256

                render_size = torch.zeros(batch_size, 2, 1, 1, device=device)
                render_size[:, 0, :, :] = lr_height
                render_size[:, 1, :, :] = lr_width

                model_inputs = {
                    "colour_linear": torch.rand(
                        batch_size, 3, lr_height, lr_width, device=device
                    ),
                    "depth": torch.rand(
                        batch_size, 1, lr_height, lr_width, device=device
                    ),
                    "depth_params": torch.rand(
                        batch_size, 4, lr_height, lr_width, device=device
                    ),
                    "ground_truth_linear": torch.rand(
                        batch_size, 3, hr_height, hr_width, device=device
                    ),
                    "jitter": torch.zeros(batch_size, 2, 1, 1, device=device),
                    "motion": torch.zeros(
                        batch_size, 2, hr_height, hr_width, device=device
                    ),
                    "motion_lr": torch.zeros(
                        batch_size, 2, lr_height, lr_width, device=device
                    ),
                    "render_size": render_size,
                    "seq": torch.ones(batch_size, 1, 1, 1, device=device),
                    "exposure": torch.ones(batch_size, 1, 1, 1, device=device),
                }
                model_inputs = nss_model.set_buffers(model_inputs)

                model_outputs = nss_model.core_forward(model_inputs)
                loss = (
                    model_outputs["output_linear"].square().mean()
                    + model_outputs["out_filtered"].square().mean()
                    + model_outputs["temporal_params"].square().mean()
                    + model_outputs["derivative"].square().mean()
                )
                loss.backward()

                grad_checks = (
                    nss_model.autoencoder.conv2d_0.conv2d.weight.grad,
                    nss_model.autoencoder.kpn_params.conv2d.weight.grad,
                    nss_model.autoencoder.temporal_params_out_conv.conv2d.weight.grad,
                )
                for grad in grad_checks:
                    self.assertIsNotNone(grad)
                    self.assertTrue(torch.isfinite(grad).all().item())
                    self.assertGreater(grad.abs().sum().item(), 0.0)
