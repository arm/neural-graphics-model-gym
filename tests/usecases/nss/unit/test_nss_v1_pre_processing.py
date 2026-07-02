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


class TestNSSV1PreprocessSmoke(BaseGPUMemoryTest):
    """Standalone smoke tests for NSS v1 preprocess."""

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required for NSS v1 Slang-backed preprocess smoke tests.",
    )
    def test_preprocess_smoke_shapes_and_finiteness(self):
        """Synthetic preprocess inputs should produce finite tensors."""

        device = torch.device("cuda")

        for quality in ("high", "mid"):
            with self.subTest(quality=quality):
                nss_model, preprocess_input = self._create_synthetic_inputs(
                    quality, device
                )
                (
                    input_shape,
                    process_shape,
                    _hr_shape,
                    pad_shape,
                    _depth_shape,
                ) = nss_model._calculate_dispatch_dims(  # pylint: disable=protected-access
                    preprocess_input
                )
                derivative_shape = (
                    pad_shape if nss_model.preprocess_half_res_input else input_shape
                )
                nearest_offset_shape = (
                    pad_shape if nss_model.preprocess_half_res_input else process_shape
                )

                (
                    input_tensor,
                    derivative,
                    disocclusion_mask,
                    nearest_depth_offset,
                ) = nss_model.preprocess(preprocess_input)

                self.assertEqual(
                    input_tensor.shape, (pad_shape[0], 12, pad_shape[2], pad_shape[3])
                )
                self.assertEqual(
                    derivative.shape,
                    (derivative_shape[0], 4, derivative_shape[2], derivative_shape[3]),
                )
                self.assertEqual(
                    disocclusion_mask.shape,
                    (process_shape[0], 2, process_shape[2], process_shape[3]),
                )
                self.assertEqual(
                    nearest_depth_offset.shape,
                    (
                        nearest_offset_shape[0],
                        nss_model._nearest_depth_offset_channels(),
                        nearest_offset_shape[2],
                        nearest_offset_shape[3],
                    ),
                )
                self.assertTrue(torch.isfinite(input_tensor).all().item())
                self.assertTrue(torch.isfinite(derivative).all().item())
                self.assertTrue(torch.isfinite(disocclusion_mask).all().item())
                self.assertTrue(torch.isfinite(nearest_depth_offset).all().item())

    @staticmethod
    def _create_model(quality, device):
        params = create_simple_params(usecase="nss_v1")
        params.model.quality = quality
        params.model.recurrent_samples = 2
        params.train.batch_size = 2
        model = NSSV1Model(params).to(device)
        model.eval()
        return model

    @classmethod
    def _create_synthetic_inputs(cls, quality, device):
        nss_model = cls._create_model(quality, device)

        batch_size = 2
        lr_height = 128
        lr_width = 128
        hr_height = 256
        hr_width = 256

        render_size = torch.zeros(batch_size, 2, 1, 1, device=device)
        render_size[:, 0, :, :] = lr_height
        render_size[:, 1, :, :] = lr_width

        preprocess_input = {
            "colour_linear": torch.rand(
                batch_size, 3, lr_height, lr_width, device=device
            ),
            "history": torch.rand(batch_size, 3, hr_height, hr_width, device=device),
            "motion_lr": torch.zeros(batch_size, 2, lr_height, lr_width, device=device),
            "depth": torch.rand(batch_size, 1, lr_height, lr_width, device=device),
            "jitter": torch.zeros(batch_size, 2, 1, 1, device=device),
            "jitter_tm1": torch.zeros(batch_size, 2, 1, 1, device=device),
            "depth_params": torch.rand(batch_size, 4, 1, 1, device=device),
            "exposure": torch.ones(batch_size, 1, 1, 1, device=device),
            "render_size": render_size,
        }

        (
            _input_shape,
            _process_shape,
            _hr_shape,
            pad_shape,
            _depth_shape,
        ) = nss_model._calculate_dispatch_dims(  # pylint: disable=protected-access
            {
                **preprocess_input,
                "ground_truth_linear": torch.rand(
                    batch_size, 3, hr_height, hr_width, device=device
                ),
                "motion": torch.zeros(
                    batch_size, 2, hr_height, hr_width, device=device
                ),
                "seq": torch.ones(batch_size, 1, 1, 1, device=device),
                "reset_event": torch.ones(batch_size, 1, 1, 1, device=device),
            }
        )
        derivative_shape = (
            pad_shape
            if nss_model.preprocess_half_res_input
            else (batch_size, 3, lr_height, lr_width)
        )

        preprocess_input["temporal_params_tm1"] = torch.rand(
            batch_size, 4, pad_shape[2], pad_shape[3], device=device
        )
        preprocess_input["derivative_tm1"] = torch.rand(
            batch_size,
            4,
            derivative_shape[2],
            derivative_shape[3],
            device=device,
        )

        return nss_model, preprocess_input


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
