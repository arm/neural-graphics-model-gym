# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
import time

# Let ensure that the wrapper function retains the original function's name, module and docstring.
from functools import wraps

import torch

logger = logging.getLogger(__name__)


def time_decorator(func):
    """Log execution time for the provided function (CPU + optional GPU)."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()

        if torch.cuda.is_available():
            start_gpu = torch.cuda.Event(enable_timing=True)
            end_gpu = torch.cuda.Event(enable_timing=True)
            start_gpu.record()
        else:
            start_gpu = None
            end_gpu = None

        result = func(*args, **kwargs)

        if torch.cuda.is_available() and end_gpu is not None:
            end_gpu.record()
            # We need to make sure that GPU computations have finished
            torch.cuda.synchronize()
            # Should be in ms
            gpu_time_ms = start_gpu.elapsed_time(end_gpu)
        else:
            gpu_time_ms = 0.0

        end_time = time.time()
        cpu_time_seconds = end_time - start_time

        total_minutes = cpu_time_seconds / 60
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)

        logger.debug(
            "\nExecuted function [%s]:\n"
            "    CPU  : %g seconds (%s hour(s) and %s minute(s))\n"
            "    GPU  : %g ms (%g seconds) [CUDA available: %s]",
            func.__name__,
            cpu_time_seconds,
            hours,
            minutes,
            gpu_time_ms,
            gpu_time_ms / 1000.0,
            torch.cuda.is_available(),
        )

        return result

    return wrapper
