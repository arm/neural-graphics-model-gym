# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from ng_model_gym.usecases.nss.model.model_v1 import NSSV1Model
from tests.base_gpu_test import BaseGPUMemoryTest
from tests.testing_utils import create_simple_params

# pylint: disable=duplicate-code

_SHADER_RTOL = 1e-5
_SHADER_ATOL = 5e-6


class TestNSSV1PostprocessGolden(BaseGPUMemoryTest):
    """Test NSS v1 postprocess implementation against known inputs and outputs."""

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required for NSS v1 Slang-backed postprocess goldens.",
    )
    def test_postprocess(self):
        """Test postprocess implementation for all NSS v1 quality modes."""

        device = torch.device("cuda")

        for quality in ("high", "mid", "low"):
            with self.subTest(quality=quality):
                postprocess_golden_input = torch.load(
                    "tests/usecases/nss/unit/data/nss_v1_golden_values/"
                    + f"nss_v1_{quality}_postprocess_inputs_golden.pt",
                    map_location=device,
                    weights_only=True,
                )

                postprocess_golden_output = torch.load(
                    "tests/usecases/nss/unit/data/nss_v1_golden_values/"
                    + f"nss_v1_{quality}_postprocess_output_golden.pt",
                    map_location=device,
                    weights_only=True,
                )

                nss_model = self._create_model(quality, device)
                # Two model input keys that NSSV1Model.postprocess() also reads
                postprocess_golden_input["ground_truth_linear"] = torch.zeros_like(
                    postprocess_golden_input["history"]
                )
                postprocess_golden_input["reset_event"] = postprocess_golden_input[
                    "reset"
                ]

                derivative = torch.empty_like(
                    postprocess_golden_input["temporal_params"]
                )
                disocclusion_mask = torch.empty_like(
                    postprocess_golden_input["nearest_depth_offset"]
                )
                outputs = nss_model.postprocess(
                    postprocess_golden_input["kpn_params"],
                    postprocess_golden_input,
                    postprocess_golden_input["temporal_params"],
                    postprocess_golden_input["nearest_depth_offset"],
                    derivative,
                    disocclusion_mask,
                )

                torch.testing.assert_close(
                    outputs["output_linear"],
                    postprocess_golden_output["output_linear"],
                    rtol=_SHADER_RTOL,
                    atol=_SHADER_ATOL,
                )
                torch.testing.assert_close(
                    outputs["out_filtered"],
                    postprocess_golden_output["out_filtered"],
                    rtol=_SHADER_RTOL,
                    atol=_SHADER_ATOL,
                )

    @staticmethod
    def _create_model(quality, device):
        params = create_simple_params(usecase="nss_v1")
        params.model.quality = quality
        params.model.recurrent_samples = 2
        params.train.batch_size = 2
        model = NSSV1Model(params).to(device)
        model.eval()
        return model
