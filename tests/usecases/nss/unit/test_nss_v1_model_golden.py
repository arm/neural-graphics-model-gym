# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code
import unittest

import torch

from ng_model_gym.core.model import create_model
from tests.base_gpu_test import BaseGPUMemoryTest
from tests.usecases.nss.unit.nss_v1_test_utils import (
    assert_tensors_close,
    create_nss_v1_test_params,
    load_nss_v1_golden,
    NSS_V1_RECURRENT_OUTPUT_KEYS,
    tensor_values,
)

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

                nss_golden_input = load_nss_v1_golden(
                    f"nss_v1_{quality}_nss_input_golden.pt",
                    device,
                )
                nss_golden_output = load_nss_v1_golden(
                    f"nss_v1_{quality}_nss_output_golden.pt",
                    device,
                )

                params = create_nss_v1_test_params(quality)

                nss_model = create_model(params, device)
                nss_model.get_neural_network().load_state_dict(
                    nss_golden_input["autoencoder_state"]
                )
                nss_model.train()

                model_input = tensor_values(nss_golden_input)
                with torch.no_grad():
                    model_output = nss_model(model_input)

                assert_tensors_close(
                    model_output,
                    nss_golden_output,
                    NSS_V1_RECURRENT_OUTPUT_KEYS,
                    rtol=_RTOL,
                    atol=_ATOL,
                )
