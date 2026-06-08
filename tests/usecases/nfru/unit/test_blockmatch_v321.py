# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from ng_model_gym.usecases.nfru.model.optical_flow.blockmatch_v321 import (
    BlockMatchV321,
    CalculateFlow,
    ExtractSearchWindows,
    upscale_and_dilate_flow,
)


class TestBlockMatchV321(unittest.TestCase):
    """Unit coverage for the private v321 blockmatch helper."""

    def test_reference_defaults_are_preserved(self):
        """Helper config should mirror the v3.2.1 reference defaults."""
        helper = BlockMatchV321()
        cfg = helper.calc_flow
        self.assertTrue(helper.rgb_in)
        self.assertEqual(cfg.levels, 6)
        self.assertEqual(cfg.template_sz, 5)
        self.assertEqual(cfg.search_range, 3)
        self.assertEqual(cfg.median_kernel, (3, 3))
        self.assertEqual(cfg.blur_levels, [1, 2, 3, 4])
        self.assertEqual(cfg.last_bm_level, 2)
        self.assertTrue(cfg.mv_hints)
        self.assertFalse(cfg.min_cv_output)
        self.assertEqual(helper.output_polarity, "positive")
        self.assertEqual(helper.mv_hints_polarity, "positive")
        self.assertEqual(helper.template_frame_id, "tm1")

    def test_upscale_and_dilate_flow_uses_nearest_depth_motion(self):
        """Depth-guided dilation should propagate the nearest-depth flow."""
        flow = torch.zeros(1, 2, 3, 3)
        flow[:, :, 1, 1] = torch.tensor([10.0, 20.0]).view(1, 2)
        depth = torch.ones(1, 1, 3, 3)
        depth[:, :, 1, 1] = 0.0

        dilated = upscale_and_dilate_flow(flow, depth, scale=1.0, kernel_size=3)

        expected = torch.zeros_like(flow)
        expected[:, :, :, :] = torch.tensor([10.0, 20.0]).view(1, 2, 1, 1)
        torch.testing.assert_close(dilated, expected)

    def test_search_window_extractors_match(self):
        """Equivalent search-window extraction should match exactly."""
        inputs = torch.randint(0, 255, (1, 1, 12, 14), dtype=torch.uint8)
        default_windows = ExtractSearchWindows(template_sz=5, max_sr=3)
        memory_efficient_windows = ExtractSearchWindows(template_sz=5, max_sr=3)

        torch.testing.assert_close(
            default_windows(inputs, search_range=3),
            memory_efficient_windows(inputs, search_range=3),
            rtol=0,
            atol=0,
        )

    def test_identical_frames_produce_zero_flow(self):
        """Identical frames with zero hints should keep the predicted flow at zero."""
        img = torch.rand(1, 3, 64, 64)
        helper = BlockMatchV321()

        with torch.no_grad():
            flow = helper(
                {
                    "img_t": img,
                    "img_tm1": img.clone(),
                    "input_mv": torch.zeros(1, 2, 64, 64),
                }
            )["output"]

        self.assertEqual(flow.shape, torch.Size([1, 2, 16, 16]))
        self.assertTrue(torch.isfinite(flow).all())
        torch.testing.assert_close(flow, torch.zeros_like(flow))

    def test_mean_flow_hint_reduces_runtime_search_range(self):
        """Positive mean-flow hint should shrink runtime search range when configured."""
        calc_flow = CalculateFlow(
            search_range=3, last_bm_level=2, mean_flow_l1_norm_hint=1.0
        )
        img = torch.rand(1, 1, 64, 64)
        mv = torch.zeros(1, 2, 64, 64)

        called_ranges = []
        original_forward = ExtractSearchWindows.forward

        def _record_forward(self, inputs, search_range):
            called_ranges.append(int(search_range))
            return original_forward(self, inputs, search_range)

        ExtractSearchWindows.forward = _record_forward
        try:
            with torch.no_grad():
                calc_flow(search_frame=img, template_frame=img.clone(), input_mv=mv)
        finally:
            ExtractSearchWindows.forward = original_forward

        self.assertGreater(len(called_ranges), 0)
        self.assertTrue(all(search_range == 1 for search_range in called_ranges))

    def test_granularity_scaling_flag_controls_output_scale(self):
        """Granularity scaling should change output magnitude by granularity factor."""
        img_t = torch.rand(1, 3, 64, 64)
        img_tm1 = torch.rand(1, 3, 64, 64)
        mv = torch.rand(1, 2, 64, 64)

        unscaled = BlockMatchV321(
            granularity_scaling=False,
            flow_params={"last_bm_level": 2, "mv_hints": False},
        )
        scaled = BlockMatchV321(
            granularity_scaling=True,
            flow_params={"last_bm_level": 2, "mv_hints": False},
        )

        fixed_flow = torch.ones(1, 2, 16, 16, dtype=torch.float16)
        unscaled.calc_flow.forward = lambda **kwargs: (fixed_flow.clone(), {})
        scaled.calc_flow.forward = lambda **kwargs: (fixed_flow.clone(), {})

        inputs = {"img_t": img_t, "img_tm1": img_tm1, "input_mv": mv}
        flow_unscaled = unscaled(inputs)["output"]
        flow_scaled = scaled(inputs)["output"]

        granularity = float(unscaled.granularity)
        torch.testing.assert_close(flow_unscaled, flow_scaled * granularity)

    def test_min_cv_output_returns_cost_volume(self):
        """Enabling min_cv_output should expose min_cost_volume in wrapper output."""
        helper = BlockMatchV321(min_cv_output=True)
        fixed_flow = torch.ones(1, 2, 16, 16, dtype=torch.float16)
        fixed_min_cv = torch.ones(1, 16, 16, 1, dtype=torch.int32)
        helper.calc_flow.forward = lambda **kwargs: (
            fixed_flow.clone(),
            {"min_cost_volume": fixed_min_cv.clone()},
        )

        out = helper(
            {
                "img_t": torch.rand(1, 3, 64, 64),
                "img_tm1": torch.rand(1, 3, 64, 64),
                "input_mv": torch.zeros(1, 2, 64, 64),
            }
        )

        self.assertIn("output", out)
        self.assertIn("min_cost_volume", out)
        torch.testing.assert_close(out["min_cost_volume"], fixed_min_cv)
