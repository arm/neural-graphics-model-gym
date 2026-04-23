# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import gc
import unittest

import torch


def _clear_cuda():
    """Clear CUDA memory if available (across all devices)."""
    if not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        torch.cuda.set_device(idx)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


class BaseGPUMemoryTest(unittest.TestCase):
    """Test case class that ensures GPU memory is cleared before and after each test."""

    def setUp(self):
        _clear_cuda()
        self.addCleanup(self._cleanup_gpu)

    def _cleanup_gpu(self):
        gc.collect()  # Force freeing of unreachable objects
        _clear_cuda()

    def _assert_peak_vram_usage(
        self, stdout: str, expected_vram_usage: int, tolerance: float
    ) -> None:
        """Assert VRAM usage parsed from stdout does not exceed expected tolerance."""
        stdout_lines = stdout.lower().splitlines()
        peak_vram_usage = None
        for line in reversed(stdout_lines):
            if "memory used:" in line:
                peak_vram_usage = float(line.split()[-2])
                break

        if peak_vram_usage is None:
            self.fail("Could not find GPU memory usage in stdout")

        print(f"Observed peak GPU memory usage: {peak_vram_usage:.2f} MiB")

        max_vram_allowed = expected_vram_usage * (1 + tolerance)
        self.assertLessEqual(
            peak_vram_usage,
            max_vram_allowed,
            (
                "Peak VRAM usage exceeded tolerance- "
                f"Peak usage: {peak_vram_usage:.2f} MiB, "
                f"Max allowance: {max_vram_allowed:.2f} MiB based on {expected_vram_usage:.2f}"
                f" with a tolerance of {tolerance * 100:.2f}%"
            ),
        )
