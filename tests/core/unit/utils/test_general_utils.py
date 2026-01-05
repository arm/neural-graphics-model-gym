# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import io
import unittest
from contextlib import redirect_stdout

from ng_model_gym.core.utils.config_utils import print_config_options


class TestPrintConfigOptions(unittest.TestCase):
    """Test for printing config options"""

    def test_print_config_options_output(self):
        """Test config options are printed to stdout"""
        io_buffer = io.StringIO()
        with redirect_stdout(io_buffer):
            print_config_options()
        output = io_buffer.getvalue()

        self.assertTrue(output.strip().startswith("{"))
        self.assertTrue(output.strip().endswith("}"))
        self.assertIn('"dataset"', output)
        self.assertIn('"recurrent_samples"', output)
        self.assertIn('"output"', output)
        self.assertIn('"train"', output)
