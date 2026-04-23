# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from ng_model_gym.core.model import DownSampling2D as CoreDownSampling2D
from ng_model_gym.core.model import UpSampling2D as CoreUpSampling2D
from ng_model_gym.usecases.nfru.utils.down_sampling_2d import (
    DownSampling2D as NFRUDownSampling2D,
)
from ng_model_gym.usecases.nfru.utils.up_sampling_2d import (
    UpSampling2D as NFRUUpSampling2D,
)


class TestNFRUResamplingWrappers(unittest.TestCase):
    """Regression tests for the NFRU resampling wrappers."""

    def test_nearest_is_normalized_to_nearest_exact(self):
        """NFRU wrappers should map nearest mode to nearest-exact."""
        up = NFRUUpSampling2D(interpolation="nearest")
        down = NFRUDownSampling2D(interpolation="nearest")

        self.assertEqual(up.interpolation, "nearest-exact")
        self.assertEqual(down.interpolation, "nearest-exact")

    def test_core_layers_keep_nearest_mode(self):
        """Core resampling layers should preserve nearest mode as-is."""
        up = CoreUpSampling2D(interpolation="nearest")
        down = CoreDownSampling2D(interpolation="nearest")

        self.assertEqual(up.interpolation, "nearest")
        self.assertEqual(down.interpolation, "nearest")

    def test_wrapper_uses_scale_factor_path(self):
        """NFRU wrappers should dispatch interpolate with scale_factor."""
        up = NFRUUpSampling2D(size=2.0)
        down = NFRUDownSampling2D(size=2.0)

        self.assertIn("scale_factor", up.method.keywords)
        self.assertEqual(up.method.keywords["scale_factor"], (2.0, 2.0))
        self.assertIn("scale_factor", down.method.keywords)
        self.assertEqual(down.method.keywords["scale_factor"], [0.5, 0.5])

    def test_basic_output_shapes_match_expected(self):
        """NFRU wrappers should produce the expected shape changes."""
        up = NFRUUpSampling2D(size=2.0, interpolation="nearest")
        down = NFRUDownSampling2D(size=2.0, interpolation="nearest")

        x_up = torch.randn(1, 3, 8, 8)
        y_up = up(x_up)
        self.assertEqual(y_up.shape, (1, 3, 16, 16))

        x_down = torch.randn(1, 3, 16, 16)
        y_down = down(x_down)
        self.assertEqual(y_down.shape, (1, 3, 8, 8))


if __name__ == "__main__":
    unittest.main()
