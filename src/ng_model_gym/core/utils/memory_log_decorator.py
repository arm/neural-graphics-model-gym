# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging

# Let ensure that the wrapper function retains the original function's name, module and docstring.
from functools import wraps

from memory_profiler import memory_usage

logger = logging.getLogger(__name__)


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
