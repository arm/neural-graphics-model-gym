# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
import time
from functools import wraps

import torch
from memory_profiler import memory_usage

try:
    from gpustat import GPUStatCollection

    GPUSTAT_AVAILABLE = True
except ImportError:
    GPUStatCollection = None
    GPUSTAT_AVAILABLE = False

logger = logging.getLogger(__name__)


def gpu_log_decorator(enabled=True, log_level=logging.INFO):
    """Logs GPU usage with gpustat before and after a function call."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Skip if disabled.
            if not enabled:
                return func(*args, **kwargs)

            # Raise error if gpustat is not available
            if not GPUSTAT_AVAILABLE:
                raise ImportError(
                    "gpustat should be installed with 'pip install gpustat' "
                )

            try:
                # Measurements are more accurate if we synchronize first.
                torch.cuda.synchronize()
                stats_start = GPUStatCollection.new_query().jsonify()["gpus"]
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "Failed to query GPU usage before running '%s': %s",
                    func.__name__,
                    e,
                )
                stats_start = None

            result = func(*args, **kwargs)

            try:
                torch.cuda.synchronize()
                stats_end = GPUStatCollection.new_query().jsonify()["gpus"]
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "Failed to query GPU usage after running '%s': %s", func.__name__, e
                )
                stats_end = None

            if stats_start and stats_end:
                for i, (start, end) in enumerate(zip(stats_start, stats_end)):
                    util_before = start.get("utilization.gpu", "N/A")
                    util_after = end.get("utilization.gpu", "N/A")
                    mem_before = start.get("memory.used", "N/A")
                    mem_after = end.get("memory.used", "N/A")

                    logger.log(
                        log_level,
                        f"[{func.__name__}] GPU {i}: "
                        f"utilization: {util_before}% -> {util_after}%, "
                        f"memory used: {mem_before} MiB -> {mem_after} MiB",
                    )
            else:
                logger.log(
                    log_level, f"[{func.__name__}] GPU usage data not available."
                )

            return result

        return wrapper

    return decorator


def memory_log_decorator(func):
    """Log memory usage of the function call."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        mem_start = memory_usage(-1, interval=0.1, timeout=1)
        res = func(*args, **kwargs)
        mem_end = memory_usage(-1, interval=0.1, timeout=1)
        logger.debug(
            f"The function [{func.__name__}] used {mem_end[0] - mem_start[0]:.2f} MiB (RAM CPU)."
        )

        return res

    return wrapper


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
