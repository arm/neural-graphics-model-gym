# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from ng_model_gym.usecases.nfru.utils.colour_pipeline import build_colour_pipeline
from ng_model_gym.usecases.nfru.utils.constants import _REC709_LUMA_WEIGHTS


class TestColourPipeline(unittest.TestCase):
    """Tests for NFRU colour-pipeline exposure handling."""

    def test_auto_exposure_supports_batched_nchw_tensors(self) -> None:
        """Auto exposure should compute a per-sample scalar for batched model inputs."""
        pipeline = build_colour_pipeline(
            {
                "pipeline": [],
                "exposure": "auto",
                "auto_exposure_key_value": 1.5,
                "auto_exposure_variance": {"m1": 0.5},
            }
        )
        rgb_linear = torch.tensor(
            [
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[0.5, 0.5], [0.5, 0.5]],
                    [[0.25, 0.25], [0.25, 0.25]],
                ],
                [
                    [[4.0, 4.0], [4.0, 4.0]],
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[0.5, 0.5], [0.5, 0.5]],
                ],
            ],
            dtype=torch.float32,
        )

        output = pipeline(rgb_linear, x_in={}, time_index="m1")

        weights = torch.tensor(_REC709_LUMA_WEIGHTS, dtype=rgb_linear.dtype).view(
            1, 3, 1, 1
        )
        mean_luminance = torch.sum(rgb_linear * weights, dim=1, keepdim=True).mean(
            dim=(-2, -1), keepdim=True
        )
        expected = rgb_linear * ((1.5 * 0.5) / mean_luminance)

        self.assertEqual(output.shape, rgb_linear.shape)
        torch.testing.assert_close(output, expected)

    def test_auto_exposure_clamps_zero_luminance(self) -> None:
        """Auto exposure should stay finite for black frames."""
        pipeline = build_colour_pipeline(
            {
                "pipeline": [],
                "exposure": "auto",
                "auto_exposure_key_value": 1.0,
                "auto_exposure_variance": None,
            }
        )
        rgb_linear = torch.zeros((2, 3, 4, 4), dtype=torch.float32)

        output = pipeline(rgb_linear, x_in={}, time_index="m1")

        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        torch.testing.assert_close(output, torch.zeros_like(output))


if __name__ == "__main__":
    unittest.main()
