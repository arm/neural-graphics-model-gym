# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
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
