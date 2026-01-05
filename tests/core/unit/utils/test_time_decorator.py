# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
import time
import unittest

from ng_model_gym.core.utils.time_decorator import time_decorator


@time_decorator
def placeholder(n):
    """Placeholder function for test."""
    time.sleep(n)


class TestTimeDecorator(unittest.TestCase):
    """Test time_decorator function."""

    def test_logging_info_is_called(self):
        """Test that name of the function is logged."""
        with self.assertLogs(logging.getLogger(), level="DEBUG") as al:
            placeholder(1)
        substring = "Executed function [placeholder]:"
        self.assertTrue(
            any(substring in message for message in al.output),
            f"'{substring}' not found in log messages: {al.output}",
        )
        substring = "CPU"
        self.assertTrue(
            any(substring in message for message in al.output),
            f"'{substring}' not found in log messages: {al.output}",
        )
        substring = "GPU"
        self.assertTrue(
            any(substring in message for message in al.output),
            f"'{substring}' not found in log messages: {al.output}",
        )
