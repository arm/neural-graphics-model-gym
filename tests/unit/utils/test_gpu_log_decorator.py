# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
import unittest
from unittest.mock import MagicMock, patch

import torch

from ng_model_gym.core.utils.gpu_log_decorator import (
    gpu_log_decorator,
    GPUSTAT_AVAILABLE,
)


@gpu_log_decorator(enabled=True)
def sample_function():
    """Test function."""
    return "Log GPU usage"


class TestGPULogDecorator(unittest.TestCase):
    """Test gpu_log_decorator function."""

    @unittest.skipIf(
        not GPUSTAT_AVAILABLE or not torch.cuda.is_available(),
        "gpustat not installed, skipped this test.",
    )
    @patch("ng_model_gym.core.utils.gpu_log_decorator.GPUStatCollection")
    def test_decorator_enabled(self, mock_gpustat_cls):
        """
        Check that we query gpustat function before and after the function.
        """
        # Mock the gpustat return data
        mock_gpustat_instance = MagicMock()
        mock_gpustat_instance.jsonify.return_value = {
            "gpus": [
                {"utilization.gpu": 10, "memory.used": 100},
                {"utilization.gpu": 20, "memory.used": 200},
            ]
        }
        mock_gpustat_cls.new_query.return_value = mock_gpustat_instance

        result = sample_function()
        self.assertEqual(result, "Log GPU usage")

        # Ensure gpustat was called twice (before and after func)
        self.assertEqual(mock_gpustat_cls.new_query.call_count, 2)

        # Check that we logged some info
        with self.assertLogs(logging.getLogger(), level="DEBUG") as al:
            sample_function()
        substring = "utilization: "
        output = al.output
        self.assertTrue(
            any(substring in msg for msg in output),
            f"'Could not found {substring}' log messages: {output}",
        )


if __name__ == "__main__":
    unittest.main()
