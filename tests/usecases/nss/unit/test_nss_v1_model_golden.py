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

_RTOL = 0.1
_ATOL = 0.01


class TestNSSV1ModelGolden(BaseGPUMemoryTest):
    """Test NSS v1 recurrent model forward pass against known golden values."""

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required for NSS v1 Slang-backed recurrent goldens.",
    )
    def test_recurrent_forward_golden_values(self):
        """Test NSS v1 recurrent forward pass for all quality modes."""

        for quality in ("high", "mid", "low"):
            with self.subTest(quality=quality):
                device = torch.device("cuda")

                nss_golden_input = torch.load(
                    "tests/usecases/nss/unit/data/nss_v1_golden_values/"
                    + f"nss_v1_{quality}_nss_input_golden.pt",
                    map_location=device,
                    weights_only=True,
                )

                nss_golden_output = torch.load(
                    "tests/usecases/nss/unit/data/nss_v1_golden_values/"
                    + f"nss_v1_{quality}_nss_output_golden.pt",
                    map_location=device,
                    weights_only=True,
                )

                params = create_simple_params(usecase="nss_v1")
                params.model_train_eval_mode = TrainEvalMode.FP32
                params.model.quality = quality
                params.model.recurrent_samples = 2
                params.train.batch_size = 2

                nss_model = create_model(params, device)
                nss_model.get_neural_network().load_state_dict(
                    nss_golden_input["autoencoder_state"]
                )
                nss_model.train()

                model_input = {
                    key: value
                    for key, value in nss_golden_input.items()
                    if isinstance(value, torch.Tensor)
                }
                with torch.no_grad():
                    model_output = nss_model(model_input)

                torch.testing.assert_close(
                    model_output["output"],
                    nss_golden_output["output"],
                    rtol=_RTOL,
                    atol=_ATOL,
                )
                torch.testing.assert_close(
                    model_output["output_linear"],
                    nss_golden_output["output_linear"],
                    rtol=_RTOL,
                    atol=_ATOL,
                )
                torch.testing.assert_close(
                    model_output["out_filtered"],
                    nss_golden_output["out_filtered"],
                    rtol=_RTOL,
                    atol=_ATOL,
                )
                torch.testing.assert_close(
                    model_output["temporal_params"],
                    nss_golden_output["temporal_params"],
                    rtol=_RTOL,
                    atol=_ATOL,
                )
                torch.testing.assert_close(
                    model_output["disocclusion_mask"],
                    nss_golden_output["disocclusion_mask"],
                    rtol=_RTOL,
                    atol=_ATOL,
                )
                torch.testing.assert_close(
                    model_output["derivative"],
                    nss_golden_output["derivative"],
                    rtol=_RTOL,
                    atol=_ATOL,
                )
                torch.testing.assert_close(
                    model_output["ground_truth"],
                    nss_golden_output["ground_truth"],
                    rtol=_RTOL,
                    atol=_ATOL,
                )
                torch.testing.assert_close(
                    model_output["input_color"],
                    nss_golden_output["input_color"],
                    rtol=_RTOL,
                    atol=_ATOL,
                )
                torch.testing.assert_close(
                    model_output["reset_event"],
                    nss_golden_output["reset_event"],
                    rtol=_RTOL,
                    atol=_ATOL,
                )
                torch.testing.assert_close(
                    model_output["motion"],
                    nss_golden_output["motion"],
                    rtol=_RTOL,
                    atol=_ATOL,
                )
