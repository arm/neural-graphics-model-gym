# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
from functools import wraps

import torch

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
