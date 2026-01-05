# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import numpy as np
import torch

from ng_model_gym.core.utils.exr_utils import read_exr_torch


class TestExrUtils(unittest.TestCase):
    """Test for reading an EXR file"""

    def test_read_exr_torch(self):
        """Tests the read_exr_torch function"""

        exr = read_exr_torch(
            "tests/datasets/test_exr/x2/motion/0002/0000.exr",
            dtype=np.float32,
            channels="RGBA",
        )
        self.assertEqual(exr.dtype, torch.float32)
        self.assertEqual(exr.shape, (1, 4, 540, 960))
        # Check that the data isn't all zero!
        self.assertGreater(np.max(exr.numpy()), 0.0)
