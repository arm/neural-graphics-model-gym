# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from ng_model_gym.core.model.shaders.slang_utils import SlangOutput
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
                slang = nss_model._get_slang()  # pylint: disable=protected-access
                (
                    input_shape,
                    process_shape,
                    _hr_shape,
                    pad_shape,
                    depth_shape,
                ) = nss_model._calculate_dispatch_dims(  # pylint: disable=protected-access
                    preprocess_input
                )
                derivative_shape = (
                    pad_shape if nss_model.preprocess_half_res_input else input_shape
                )
                nearest_offset_shape = (
                    pad_shape if nss_model.preprocess_half_res_input else process_shape
                )

                depth_tm1 = slang.depth_scatter(
                    in_motion=preprocess_input["motion_lr"],
                    in_depth=preprocess_input["depth"],
                    in_render_size=preprocess_input["render_size"],
                    out_constructors={
                        "out_tensor": SlangOutput(
                            init="full",
                            shape=depth_shape,
                            value=torch.iinfo(torch.int32).max,
                            dtype=torch.int32,
                            device=str(device),
                        ),
                    },
                    dispatch_size=[depth_shape[0], depth_shape[2], depth_shape[3]],
                )

                (
                    input_tensor,
                    derivative,
                    disocclusion_mask,
                    nearest_depth_offset,
                ) = slang.pre_process(
                    in_colour=preprocess_input["colour_linear"],
                    in_history=preprocess_input["history"],
                    in_motion=preprocess_input["motion_lr"],
                    in_depth=preprocess_input["depth"],
                    in_depth_tm1=depth_tm1,
                    in_jitter=preprocess_input["jitter"],
                    in_jitter_tm1=preprocess_input["jitter_tm1"],
                    in_feedback_tm1=preprocess_input["temporal_params_tm1"],
                    in_derivative_tm1=preprocess_input["derivative_tm1"],
                    in_depth_params=preprocess_input["depth_params"],
                    in_exposure=preprocess_input["exposure"],
                    in_render_size=preprocess_input["render_size"],
                    out_constructors={
                        "out_tensor": SlangOutput(
                            shape=pad_shape,
                            channel_dim=12,
                            device=str(device),
                        ),
                        "out_luma_derivative": SlangOutput(
                            shape=derivative_shape,
                            channel_dim=4,
                            device=str(device),
                        ),
                        "out_disocclusion_mask": SlangOutput(
                            shape=process_shape,
                            channel_dim=2,
                            device=str(device),
                        ),
                        "out_nearest_depth_off": SlangOutput(
                            shape=nearest_offset_shape,
                            channel_dim=nss_model._nearest_depth_offset_channels(),
                            device=str(device),
                        ),
                    },
                    dispatch_size=[pad_shape[0], pad_shape[2], pad_shape[3]],
                )

                torch.testing.assert_close(
                    depth_tm1,
                    preprocess_output["depth_tm1"],
                    rtol=_SHADER_RTOL,
                    atol=_SHADER_ATOL,
                )
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
