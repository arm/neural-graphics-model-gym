# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import filecmp
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ng_model_gym.core.utils import dds_utils


class TestDdsUtils(unittest.TestCase):
    """Test for the dds_utils module"""

    def setUp(self):
        """Setup common test data and state"""
        self.tmp_path = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.tmp_path)

    def test_read_dds(self):
        """Tests the read_dds function"""

        dds = dds_utils.read_dds("tests/core/unit/utils/data/SampleDDS.dds")
        self.assertEqual(dds.dtype, np.float16)
        self.assertEqual(dds.shape, (3, 540, 960))
        # Check that the data isn't all zero!
        self.assertGreater(np.max(dds), 0.0)

        # Check a specific pixel to make sure that the R11G11B10 format is being decoded correctly
        self.assertEqual(dds[0, 437, 413], np.float16(82.0))  # R
        self.assertEqual(dds[1, 437, 413], np.float16(41.5))  # G
        self.assertEqual(dds[2, 437, 413], np.float16(20.5))  # B

    def test_save_dds(self):
        """Tests the save_dds function"""

        # Create a simple 10x10 test image
        r = np.fromfunction(lambda y, x: (x + y) / 18, (10, 10), dtype=np.float16)
        g = np.fromfunction(lambda y, x: (x * y) / 81, (10, 10), dtype=np.float16)
        b = np.fromfunction(lambda y, x: (x - y + 9) / 18, (10, 10), dtype=np.float16)
        dds = np.stack((r, g, b), axis=0)

        # Save as R11G11B10 to test the floating-point encoding logic
        test_dds_path = self.tmp_path / "TestDDS.dds"
        dds_utils.save_dds(dds, test_dds_path, dds_utils.DXGI_FORMAT_R11G11B10_FLOAT)

        # Check if matches the checked-in reference file
        self.assertTrue(
            filecmp.cmp(
                "tests/core/unit/utils/data/GoldenDDS.dds", test_dds_path, shallow=False
            )
        )
