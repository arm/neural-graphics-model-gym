# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
import unittest

from ng_model_gym.core.utils.memory_log_decorator import memory_log_decorator


@memory_log_decorator
def placeholder(n):
    """Placeholder function for test."""
    return n * n


class TestMemoryLogDecorator(unittest.TestCase):
    """Test memory_log_decorator function."""

    def test_logging_info_is_called(self):
        """Test that memory usage is logged."""
        res = placeholder(100)

        # Check that function is called properly
        self.assertEqual(res, 10000)

        # Check that we logged some info
        with self.assertLogs(logging.getLogger(), level="DEBUG") as al:
            placeholder(1)
        s = "The function [placeholder] used 0.00 MiB (RAM CPU)."
        output = al.output
        self.assertTrue(
            any(s in msg for msg in output),
            f"'Could not found {s}' log messages: {output}",
        )
