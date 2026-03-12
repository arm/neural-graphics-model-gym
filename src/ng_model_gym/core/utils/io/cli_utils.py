# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
from contextlib import contextmanager

import click
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)


def is_invoked_cli() -> bool:
    """Checks if the current program is running from our CLI"""
    ctx = click.get_current_context(silent=True)
    is_cli_program = bool(ctx and ctx.obj.get("ng-model-gym-cli-active"))
    return is_cli_program


@contextmanager
def suspend_tqdm_bar():
    """Hide tqdm bar whilst in this context manager"""
    tqdm_bars = list(getattr(tqdm, "_instances", []))

    for tqdm_bar in tqdm_bars:
        tqdm_bar.clear()

    try:
        yield
    finally:
        for tqdm_bar in tqdm_bars:
            tqdm_bar.refresh()
