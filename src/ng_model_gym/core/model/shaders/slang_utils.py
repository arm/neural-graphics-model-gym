# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import logging
from functools import lru_cache
from importlib.resources import files

import slangtorch
from rich.console import Console

from ng_model_gym.core.utils.general_utils import is_invoked_cli, suspend_tqdm_bar

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_slang_module(shader_dir, shader_file):
    """Load a Slang module from the specified shader path and file"""
    shader_path = files(shader_dir) / shader_file

    if not is_invoked_cli():
        logger.info("Loading Slang shaders...")
        return slangtorch.loadModule(
            shader_path,
            skipNinjaCheck=True,  # Use cache
            verbose=False,
        )

    status = Console().status("[bold green]Loading Slang shaders...", spinner="dots")

    # Suspend tqdm bar and show spinner in CLI
    with suspend_tqdm_bar(), status:
        module = slangtorch.loadModule(
            shader_path,
            skipNinjaCheck=True,  # Use cache
            verbose=False,
        )

    return module
