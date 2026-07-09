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


class TestNSSV1PostprocessSmoke(BaseGPUMemoryTest):
    """Standalone smoke tests for NSS v1 postprocess."""

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required for NSS v1 Slang-backed postprocess smoke tests.",
    )
    def test_postprocess_smoke_shapes_and_finiteness(self):
        """Synthetic postprocess inputs should produce finite HR outputs."""

        device = torch.device("cuda")

        for quality in ("high", "mid", "low"):
            with self.subTest(quality=quality):
                (
                    model,
                    postprocess_input,
                    kpn_params,
                    temporal_params,
                    nearest_depth_offset,
                    derivative,
                    disocclusion_mask,
                    hr_shape,
                ) = self._create_synthetic_inputs(quality, device)

                outputs = model.postprocess(
                    kpn_params,
                    postprocess_input,
                    temporal_params,
                    nearest_depth_offset,
                    derivative,
                    disocclusion_mask,
                )

                output_linear = outputs["output_linear"]
                out_filtered = outputs["out_filtered"]
                self.assertEqual(output_linear.shape, hr_shape)
                self.assertEqual(out_filtered.shape, hr_shape)
                self.assertTrue(torch.isfinite(output_linear).all().item())
                self.assertTrue(torch.isfinite(out_filtered).all().item())

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required for NSS v1 Slang-backed postprocess smoke tests.",
    )
    def test_postprocess_accepts_reset_input(self):
        """Reset tensors of zeros and ones should both execute successfully."""

        device = torch.device("cuda")

        for quality in ("high", "mid", "low"):
            with self.subTest(quality=quality):
                (
                    model,
                    postprocess_input,
                    kpn_params,
                    temporal_params,
                    nearest_depth_offset,
                    derivative,
                    disocclusion_mask,
                    hr_shape,
                ) = self._create_synthetic_inputs(quality, device)

                for reset_value in (0.0, 1.0):
                    with self.subTest(quality=quality, reset_value=reset_value):
                        reset_input = {
                            **postprocess_input,
                            "reset_event": torch.full_like(
                                postprocess_input["reset_event"], reset_value
                            ),
                        }

                        outputs = model.postprocess(
                            kpn_params,
                            reset_input,
                            temporal_params,
                            nearest_depth_offset,
                            derivative,
                            disocclusion_mask,
                        )

                        output_linear = outputs["output_linear"]
                        out_filtered = outputs["out_filtered"]
                        self.assertEqual(output_linear.shape, hr_shape)
                        self.assertEqual(out_filtered.shape, hr_shape)
                        self.assertTrue(torch.isfinite(output_linear).all().item())
                        self.assertTrue(torch.isfinite(out_filtered).all().item())

    def _create_synthetic_inputs(self, quality, device):
        model = TestNSSV1PostprocessGolden._create_model(quality, device)

        batch_size = 2
        lr_height = 128
        lr_width = 128
        hr_height = 256
        hr_width = 256

        render_size = torch.zeros(batch_size, 2, 1, 1, device=device)
        render_size[:, 0, :, :] = lr_height
        render_size[:, 1, :, :] = lr_width

        dispatch_input = {
            "colour_linear": torch.rand(
                batch_size, 3, lr_height, lr_width, device=device
            ),
            "history": torch.rand(batch_size, 3, hr_height, hr_width, device=device),
            "depth": torch.rand(batch_size, 1, lr_height, lr_width, device=device),
            "depth_params": torch.rand(batch_size, 4, 1, 1, device=device),
            "ground_truth_linear": torch.rand(
                batch_size, 3, hr_height, hr_width, device=device
            ),
            "jitter": torch.zeros(batch_size, 2, 1, 1, device=device),
            "motion": torch.zeros(batch_size, 2, hr_height, hr_width, device=device),
            "motion_lr": torch.zeros(batch_size, 2, lr_height, lr_width, device=device),
            "render_size": render_size,
            "seq": torch.ones(batch_size, 1, 1, 1, device=device),
            "exposure": torch.ones(batch_size, 1, 1, 1, device=device),
            "reset_event": torch.ones(batch_size, 1, 1, 1, device=device),
        }

        (
            input_shape,
            process_shape,
            hr_shape,
            pad_shape,
            _depth_shape,
        ) = model._calculate_dispatch_dims(
            dispatch_input
        )  # pylint: disable=protected-access
        derivative_shape = pad_shape if model.preprocess_half_res_input else input_shape
        nearest_offset_shape = (
            pad_shape if model.preprocess_half_res_input else process_shape
        )

        postprocess_input = {
            "colour_linear": dispatch_input["colour_linear"],
            "history": dispatch_input["history"],
            "ground_truth_linear": dispatch_input["ground_truth_linear"],
            "motion_lr": dispatch_input["motion_lr"],
            "exposure": dispatch_input["exposure"],
            "jitter": dispatch_input["jitter"],
            "reset_event": torch.zeros(batch_size, 1, 1, 1, device=device),
        }

        kpn_params = torch.rand(
            batch_size,
            model.autoencoder.kpn_ch,
            pad_shape[2] // 4,
            pad_shape[3] // 4,
            device=device,
        )
        temporal_params = torch.rand(
            batch_size,
            model.autoencoder.temporal_ch,
            pad_shape[2],
            pad_shape[3],
            device=device,
        )
        nearest_depth_offset = torch.rand(
            batch_size,
            model._nearest_depth_offset_channels(),  # pylint: disable=protected-access
            nearest_offset_shape[2],
            nearest_offset_shape[3],
            device=device,
        )
        derivative = torch.rand(
            batch_size,
            4,
            derivative_shape[2],
            derivative_shape[3],
            device=device,
        )
        disocclusion_mask = torch.rand(
            batch_size,
            2,
            process_shape[2],
            process_shape[3],
            device=device,
        )

        return (
            model,
            postprocess_input,
            kpn_params,
            temporal_params,
            nearest_depth_offset,
            derivative,
            disocclusion_mask,
            hr_shape,
        )


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
        params.processing.shader_accurate = True
        params.model.quality = quality
        params.model.recurrent_samples = 2
        params.train.batch_size = 2
        model = NSSV1Model(params).to(device)
        model.eval()
        return model
