# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

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
