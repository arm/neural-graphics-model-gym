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


class TestNSSV1PreprocessGolden(BaseGPUMemoryTest):
    """Test NSS v1 preprocess implementation against known inputs and outputs."""

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required for NSS v1 Slang-backed preprocess goldens.",
    )
    def test_preprocess(self):
        """Test preprocess implementation for NSS v1 quality modes."""

        device = torch.device("cuda")

        # Note: mid and low quality mode run the same code in pre-processing
        for quality in ("high", "mid"):
            with self.subTest(quality=quality):
                preprocess_input = torch.load(
                    "tests/usecases/nss/unit/data/nss_v1_golden_values/"
                    + f"nss_v1_{quality}_preprocess_inputs_golden.pt",
                    map_location=device,
                    weights_only=True,
                )

                preprocess_output = torch.load(
                    "tests/usecases/nss/unit/data/nss_v1_golden_values/"
                    + f"nss_v1_{quality}_preprocess_output_golden.pt",
                    map_location=device,
                    weights_only=True,
                )

                nss_model = self._create_model(quality, device)
                (
                    input_tensor,
                    derivative,
                    disocclusion_mask,
                    nearest_depth_offset,
                ) = nss_model.preprocess(preprocess_input)

                torch.testing.assert_close(
                    input_tensor,
                    preprocess_output["input_tensor"],
                    rtol=_SHADER_RTOL,
                    atol=_SHADER_ATOL,
                )
                torch.testing.assert_close(
                    derivative,
                    preprocess_output["derivative"],
                    rtol=_SHADER_RTOL,
                    atol=_SHADER_ATOL,
                )
                torch.testing.assert_close(
                    disocclusion_mask,
                    preprocess_output["disocclusion_mask"],
                    rtol=_SHADER_RTOL,
                    atol=_SHADER_ATOL,
                )
                torch.testing.assert_close(
                    nearest_depth_offset,
                    preprocess_output["nearest_depth_offset"],
                    rtol=_SHADER_RTOL,
                    atol=_SHADER_ATOL,
                )

    def test_offset_lut(self):
        """Test offset LUT generation for NSS v1 quality modes."""

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for quality in ("high", "mid"):
            with self.subTest(quality=quality):
                offset_lut_golden_input = torch.load(
                    "tests/usecases/nss/unit/data/nss_v1_golden_values/"
                    + f"nss_v1_{quality}_offset_lut_input_golden.pt",
                    map_location=device,
                    weights_only=True,
                )

                offset_lut_golden_output = torch.load(
                    "tests/usecases/nss/unit/data/nss_v1_golden_values/"
                    + f"nss_v1_{quality}_offset_lut_output_golden.pt",
                    map_location=device,
                    weights_only=True,
                )

                nss_model = self._create_model(quality, device)
                (
                    offset_lut,
                    idx_modulo,
                ) = nss_model._generate_offset_lut(  # pylint: disable=protected-access
                    offset_lut_golden_input["jitter"],
                    tuple(
                        int(value)
                        for value in offset_lut_golden_input["input_shape"].tolist()
                    ),
                    tuple(
                        int(value)
                        for value in offset_lut_golden_input["output_shape"].tolist()
                    ),
                )

                torch.testing.assert_close(
                    offset_lut,
                    offset_lut_golden_output["offset_lut"],
                    rtol=_SHADER_RTOL,
                    atol=_SHADER_ATOL,
                )
                torch.testing.assert_close(
                    idx_modulo,
                    offset_lut_golden_output["idx_modulo"],
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
