# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch
import torch.nn.functional as F

from scripts.safetensors_generator.dataset_readers.flow_ops import (
    _window,
    upscale_and_dilate_flow,
)


class TestFlowOps(unittest.TestCase):
    """Direct unit coverage for script-side NFRU flow helpers."""

    @staticmethod
    def _sample_flow_and_depth() -> tuple[torch.Tensor, torch.Tensor]:
        flow = torch.zeros(1, 2, 3, 3, dtype=torch.float32)
        flow[:, :, 1, 1] = torch.tensor([10.0, 20.0], dtype=torch.float32).view(1, 2)
        depth = torch.ones(1, 1, 3, 3, dtype=torch.float32)
        depth[:, :, 1, 1] = 0.0
        return flow, depth

    def test_window_matches_explicit_reflect_pad_and_unfold(self):
        """Window extraction should match reference reflected 3x3 patches."""
        x = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)

        actual = _window(x, ksize=3, stride=2, mode="reflect")
        # Golden values for comparison
        expected = torch.tensor(
            [
                [
                    [
                        [[4.0, 6.0], [4.0, 6.0]],
                        [[5.0, 7.0], [5.0, 7.0]],
                        [[6.0, 6.0], [6.0, 6.0]],
                        [[0.0, 2.0], [8.0, 10.0]],
                        [[1.0, 3.0], [9.0, 11.0]],
                        [[2.0, 2.0], [10.0, 10.0]],
                        [[4.0, 6.0], [4.0, 6.0]],
                        [[5.0, 7.0], [5.0, 7.0]],
                        [[6.0, 6.0], [6.0, 6.0]],
                    ],
                    [
                        [[16.0, 18.0], [16.0, 18.0]],
                        [[17.0, 19.0], [17.0, 19.0]],
                        [[18.0, 18.0], [18.0, 18.0]],
                        [[12.0, 14.0], [20.0, 22.0]],
                        [[13.0, 15.0], [21.0, 23.0]],
                        [[14.0, 14.0], [22.0, 22.0]],
                        [[16.0, 18.0], [16.0, 18.0]],
                        [[17.0, 19.0], [17.0, 19.0]],
                        [[18.0, 18.0], [18.0, 18.0]],
                    ],
                ]
            ],
            dtype=torch.float32,
        )

        self.assertEqual(actual.shape, (1, 2, 9, 2, 2))
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_upscale_and_dilate_flow_selects_nearest_depth_motion(self):
        """Depth-guided dilation should propagate the nearest-depth motion."""
        flow, depth = self._sample_flow_and_depth()

        actual = upscale_and_dilate_flow(flow, depth, scale=1.0, kernel_size=3)

        expected = torch.zeros_like(flow)
        expected[:, :, :, :] = torch.tensor([10.0, 20.0], dtype=torch.float32).view(
            1, 2, 1, 1
        )
        torch.testing.assert_close(actual, expected)

    def test_upscale_and_dilate_flow_does_not_scale_non_flow_inputs(self):
        """Non-flow inputs should be dilated and upsampled without motion scaling."""
        flow, depth = self._sample_flow_and_depth()

        dilated = upscale_and_dilate_flow(flow, depth, scale=1.0, is_flow=False)
        actual = upscale_and_dilate_flow(flow, depth, scale=2.0, is_flow=False)
        expected = F.interpolate(dilated, size=(6, 6), mode="nearest")
        flow_scaled = upscale_and_dilate_flow(flow, depth, scale=2.0, is_flow=True)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(flow_scaled, expected * 2.0)

    def test_upscale_and_dilate_flow_bilinear_matches_manual_reference(self):
        """Bilinear interpolation should match manual dilation + interpolate logic."""
        flow = torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3)
        depth = torch.tensor(
            [[[[0.4, 0.3, 0.2], [0.5, 0.0, 0.6], [0.7, 0.8, 0.9]]]],
            dtype=torch.float32,
        )

        dilated = upscale_and_dilate_flow(flow, depth, scale=1.0, kernel_size=3)
        expected = (
            F.interpolate(dilated, size=(6, 6), mode="bilinear", align_corners=False)
            * 2.0
        )
        actual = upscale_and_dilate_flow(
            flow,
            depth,
            scale=2.0,
            kernel_size=3,
            interpolation="bilinear",
        )

        self.assertEqual(actual.shape, (1, 2, 6, 6))
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
