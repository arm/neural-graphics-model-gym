# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from ng_model_gym.core.data.data_utils import tonemap_forward
from ng_model_gym.core.model.shaders.slang_utils import SlangOutput
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
                slang = nss_model._get_slang()  # pylint: disable=protected-access
                (
                    _input_shape,
                    _process_shape,
                    hr_shape,
                    _pad_shape,
                    _depth_shape,
                ) = nss_model._calculate_dispatch_dims(  # pylint: disable=protected-access
                    postprocess_golden_input
                )

                output_linear, out_filtered_linear = slang.post_process(
                    in_colour=postprocess_golden_input["colour_linear"],
                    in_history=postprocess_golden_input["history"],
                    in_kpn_params=postprocess_golden_input["kpn_params"],
                    in_temporal_params=postprocess_golden_input["temporal_params"],
                    in_motion=postprocess_golden_input["motion_lr"],
                    in_nearest_depth_off=postprocess_golden_input[
                        "nearest_depth_offset"
                    ],
                    in_exposure=postprocess_golden_input["exposure"],
                    in_jitter=postprocess_golden_input["jitter"],
                    in_offset_lut=postprocess_golden_input["offset_lut"],
                    in_idx_modulo=postprocess_golden_input["idx_modulo"],
                    in_reset=postprocess_golden_input["reset"],
                    out_constructors={
                        "out_colour": SlangOutput(shape=hr_shape, device=str(device)),
                        "out_colour_filtered": SlangOutput(
                            shape=hr_shape,
                            device=str(device),
                        ),
                    },
                    dispatch_size=[hr_shape[0], hr_shape[2], hr_shape[3]],
                )
                out_filtered = tonemap_forward(
                    out_filtered_linear * postprocess_golden_input["exposure"],
                    mode=nss_model.tonemapper,
                )

                torch.testing.assert_close(
                    output_linear,
                    postprocess_golden_output["output_linear"],
                    rtol=_SHADER_RTOL,
                    atol=_SHADER_ATOL,
                )
                torch.testing.assert_close(
                    out_filtered_linear,
                    postprocess_golden_output["out_filtered_linear"],
                    rtol=_SHADER_RTOL,
                    atol=_SHADER_ATOL,
                )
                torch.testing.assert_close(
                    out_filtered,
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
