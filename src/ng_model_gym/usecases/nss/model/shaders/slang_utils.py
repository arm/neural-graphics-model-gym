# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
from functools import lru_cache
from importlib.resources import files

from rich.console import Console

from ng_model_gym.core.utils.general_utils import is_invoked_cli

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_slang_module(slang_shader="nss_v1.slang"):
    """Load a Slang module from the specified shader file.

    We cache to save having to recompiling the shader every time.
    """
    # pylint: disable-next=import-outside-toplevel
    import slangtorch

    shader_folder_path = "ng_model_gym.usecases.nss.model.shaders"
    shader_path = files(shader_folder_path) / slang_shader

    # Check if program was invoked by CLI
    if is_invoked_cli():
        with Console().status("[bold green]Loading Slang shaders…", spinner="dots"):
            module = slangtorch.loadModule(shader_path, verbose=False)

    else:
        logger.info("Loading Slang shaders...")
        module = slangtorch.loadModule(shader_path, verbose=False)

    return module
