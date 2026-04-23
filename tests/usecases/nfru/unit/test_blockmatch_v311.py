# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from ng_model_gym.usecases.nfru.model.blockmatch_v311 import (
    BlockMatchV311,
    BlockMatchV311Config,
    ExtractSearchWindows,
    upscale_and_dilate_flow,
)


class TestBlockMatchV311(unittest.TestCase):
    """Unit coverage for the private v311 blockmatch helper."""

    def test_reference_defaults_are_preserved(self):
        """Helper config should mirror the v1 reference defaults."""
        cfg = BlockMatchV311Config()
        self.assertTrue(cfg.rgb_in)
        self.assertEqual(cfg.levels, 6)
        self.assertEqual(cfg.template_sz, 5)
        self.assertEqual(cfg.search_range, 3)
        self.assertEqual(cfg.median_kernel, (3, 3))
        self.assertEqual(cfg.blur_levels, (1, 2, 3, 4))
        self.assertEqual(cfg.last_bm_level, 2)
        self.assertTrue(cfg.mv_hints)
        self.assertFalse(cfg.min_cv_output)
        self.assertFalse(cfg.oob_replacement)
        self.assertEqual(cfg.output_polarity, "positive")
        self.assertEqual(cfg.mv_hints_polarity, "positive")
        self.assertEqual(cfg.template_frame_id, "tm1")

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
        """Memory-efficient window extraction should match the dense implementation."""
        inputs = torch.randint(0, 255, (1, 1, 12, 14), dtype=torch.uint8)
        default_windows = ExtractSearchWindows(
            template_sz=5, max_sr=3, memory_efficient=False
        )
        memory_efficient_windows = ExtractSearchWindows(
            template_sz=5, max_sr=3, memory_efficient=True
        )

        torch.testing.assert_close(
            default_windows(inputs, search_range=3),
            memory_efficient_windows(inputs, search_range=3),
            rtol=0,
            atol=0,
        )

    def test_identical_frames_produce_zero_flow(self):
        """Identical frames with zero hints should keep the predicted flow at zero."""
        img = torch.rand(1, 3, 64, 64)
        helper = BlockMatchV311()

        with torch.no_grad():
            flow = helper(img, img.clone(), torch.zeros(1, 2, 64, 64))

        self.assertEqual(flow.shape, torch.Size([1, 2, 16, 16]))
        self.assertTrue(torch.isfinite(flow).all())
        torch.testing.assert_close(flow, torch.zeros_like(flow))

    def test_default_helper_keeps_memory_efficient_search_windows_in_eval(self):
        """Eval should not silently switch back to the dense search-window path."""
        img = torch.rand(1, 3, 64, 64)
        helper = BlockMatchV311()
        helper.eval()

        self.assertTrue(helper.calc_flow._search_windows.memory_efficient)

        with torch.no_grad():
            helper(img, img.clone(), torch.zeros(1, 2, 64, 64))

        self.assertTrue(helper.calc_flow._search_windows.memory_efficient)
