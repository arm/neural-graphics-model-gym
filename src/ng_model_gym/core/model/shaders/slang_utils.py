# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import slangtorch
from rich.console import Console

from ng_model_gym.core.utils.general_utils import is_invoked_cli

logger = logging.getLogger(__name__)


FAST_TEST_ENABLED = os.getenv("FAST_TEST") == "1"


def _fix_metadata(shader_path: Path) -> None:
    """Patching bug in slangtorch pointing to a .pyd Windows files instead of .so for cache"""
    cache_root = shader_path.parent / ".slangtorch_cache" / shader_path.stem
    if not cache_root.exists():
        return

    for hash_dir in cache_root.iterdir():
        if not hash_dir.is_dir():
            continue

        for build_dir in hash_dir.iterdir():
            if not build_dir.is_dir():
                continue

            metadata_path = build_dir / "metadata.json"
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                continue

            module_name = metadata.get("moduleName")
            if not module_name:
                continue

            so_path = build_dir / f"{module_name}.so"
            if so_path.exists() and metadata.get("moduleBinary") != str(so_path):
                metadata["moduleBinary"] = str(so_path)
                metadata_path.write_text(
                    json.dumps(metadata, indent=4), encoding="utf-8"
                )
                logger.debug(
                    f"Patched slangtorch metadata to {so_path}",
                )


if FAST_TEST_ENABLED:
    _ORIG_LOAD_MODULE = slangtorch.loadModule

    # Patch slang torch
    def _patched_load_module(*args, **kwargs):
        module = _ORIG_LOAD_MODULE(*args, **kwargs)
        shader_arg = kwargs.get("shader_path") if kwargs else None
        if shader_arg is None and args:
            shader_arg = args[0]
        if shader_arg is not None:
            shader_path = Path(shader_arg)
        else:
            return module
        _fix_metadata(shader_path)
        return module

    slangtorch.loadModule = _patched_load_module


@lru_cache(maxsize=1)
def load_slang_module(shader_dir, shader_file):
    """
    Load a Slang module from the specified shader path and file.
    If `FAST_TEST` env var is enabled, load from cache.
    """

    shader_path = files(shader_dir) / shader_file

    skip_ninja_check = FAST_TEST_ENABLED

    # Check if program was invoked by CLI
    if is_invoked_cli():
        with Console().status("[bold green]Loading Slang shaders…", spinner="dots"):
            module = slangtorch.loadModule(
                shader_path,
                skipNinjaCheck=skip_ninja_check,
                verbose=False,
            )

    else:
        logger.info("Loading Slang shaders...")
        module = slangtorch.loadModule(
            shader_path,
            skipNinjaCheck=skip_ninja_check,
            verbose=False,
        )

    return module
