# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from functools import lru_cache
from importlib.resources import files
from numbers import Number
from typing import Any, Callable, Dict, Optional, Union

import joblib
import slangtorch
import torch
from rich.console import Console

from ng_model_gym.core.utils.io.cli_utils import is_invoked_cli, suspend_tqdm_bar

from .slang_function_wrapper import convert_slang_wrapped_function

logger = logging.getLogger(__name__)


def _normalise_slang_defines(
    defines: Optional[Dict[str, object]],
) -> tuple[tuple[str, object], ...]:
    """Convert optional Slang defines into a cacheable tuple."""

    if not defines:
        return ()

    normalised = []
    for name, value in defines.items():
        define_value = value if isinstance(value, (str, Number)) else str(value)
        normalised.append((str(name), define_value))

    return tuple(sorted(normalised))


def _normalise_include_paths(
    include_paths: Optional[Sequence[object]],
) -> tuple[str, ...]:
    """Convert include paths into a cacheable tuple."""

    if not include_paths:
        return ()

    return tuple(str(path) for path in include_paths)


@lru_cache(maxsize=None)
def _load_slang_module_cached(
    shader_dir: str,
    shader_file: str,
    autograd: bool,
    defines: tuple[tuple[str, object], ...],
    include_paths: tuple[str, ...],
):
    """Load and cache a Slang module for a unique compile configuration."""

    shader_path = files(shader_dir) / shader_file
    load_kwargs: Dict[str, Any] = {
        "skipNinjaCheck": True,  # Use cache
        "verbose": False,
    }
    if defines:
        load_kwargs["defines"] = dict(defines)
    if include_paths:
        load_kwargs["includePaths"] = list(include_paths)

    module = slangtorch.loadModule(shader_path, **load_kwargs)
    if autograd:
        return convert_slang_wrapped_function(module)

    return module


def load_slang_module(
    shader_dir,
    shader_file,
    autograd=False,
    defines: Optional[Dict[str, object]] = None,
    include_paths: Optional[Sequence[object]] = None,
):
    """
    Load a Slang module from the specified shader path and file.

    Optional compile-time `defines` and `include_paths` are forwarded to
    `slangtorch.loadModule()`.

    If autograd=True, Autograd functions are auto-generated.
    See SlangOutput for more details.
    """
    normalised_defines = _normalise_slang_defines(defines)
    normalised_include_paths = _normalise_include_paths(include_paths)

    def _load_slang_module():
        """Wrapped implementation without logging"""
        return _load_slang_module_cached(
            shader_dir,
            shader_file,
            autograd,
            normalised_defines,
            normalised_include_paths,
        )

    if not is_invoked_cli():
        logger.info("Loading Slang shaders...")
        return _load_slang_module()

    status = Console().status("[bold green]Loading Slang shaders...", spinner="dots")

    # Suspend tqdm bar and show spinner in CLI
    with suspend_tqdm_bar(), status:
        module = _load_slang_module()

    return module


class SlangOutput:
    """
    Describes auto-generated custom Autograd functions for Slang module bindings, removing
    the need to write specific fwd/bwd functions. Requires autograd=True to be passed to
    load_slang_module(). Usage:

    ```
        from ng_model_gym.core.model.shaders.slang_utils import load_slang_module, SlangOutput

        m = loadModule('path/to/slang/file', 'my_shader.slang')

        output = m.my_slang_function(
            first_input=var1,
            second_input=var2,
            third_input=var3,
            out_constructors={
                "first_output": SlangOutput(
                    shape=(
                        first_input.shape[0],
                        16,
                        first_input.shape[2],
                        first_input.shape[3],
                    ),
                    init='zeros',
                    dtype=torch.float32,
                )
            },
        )
    ```

    SlangOutput's init arg can be a string, or a user defined function to create a custom
    constructor:

    ```
        output = m.my_slang_function(
            first_input=var1,
            second_input=var2,
            third_input=var3,
            out_constructors={
                "first_output": SlangOutput(
                    init=lambda _: my_custom_code_here
                )
            },
        )
    ```

    By default, dispatch sizes are based on the first output argument, in order.
    By default, block sizes default to 512 or 256 for forward/backward passes respectively.
    They can be manually defined. Example:

    ```
        output = m.my_slang_function(
            first_input=var1,
            second_input=var2,
            third_input=var3,
            out_constructors={
                "first_output": SlangOutput(
                    shape=(
                        first_input.shape[0],
                        16,
                        first_input.shape[2],
                        first_input.shape[3],
                    ),
                    init='zeros',
                    dtype=torch.float32,
                )
            },
            dispatch_size = [third_input.shape[0], third_input.shape[2], third_input.shape[3]],
            block_size=1024,
        )
    ```
    """

    def __init__(
        self, *args, init: Union[str, Callable] = "zeros", device="cuda", **kwargs
    ):
        self.init: Union[str, Callable] = init
        self.device = device
        self.args = args
        self.kwargs = kwargs

        self.init_mapping: Dict[str, Callable] = {
            "zeros": self._init_zeros,
            "ones": self._init_ones,
            "full": self._init_full,
        }

    def _init_zeros(
        self,
        shape: Sequence[int],
        channel_dim: Optional[int] = None,
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        if len(shape) != 4:
            raise ValueError("_init_zeros only supports rank 4 tensors currently.")

        if channel_dim is None:
            channel_dim = shape[1]

        return torch.zeros(
            (shape[0], channel_dim, shape[2], shape[3]),
            dtype=dtype,
            device=self.device,
        )

    def _init_ones(
        self,
        shape: Sequence[int],
        channel_dim: Optional[int] = None,
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        if len(shape) != 4:
            raise ValueError("_init_ones only supports rank 4 tensors currently.")

        if channel_dim is None:
            channel_dim = shape[1]

        return torch.ones(
            (shape[0], channel_dim, shape[2], shape[3]),
            dtype=dtype,
            device=self.device,
        )

    def _init_full(
        self,
        shape: Sequence[int],
        value: Union[torch.Tensor, Number],
        channel_dim: Optional[int] = None,
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        if len(shape) != 4:
            raise ValueError("_init_full only supports rank 4 tensors currently.")

        if channel_dim is None:
            channel_dim = shape[1]

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("_init_full expects a scalar tensor for `value`.")
            fill_value = value.to(self.device).item()
        else:
            fill_value = value

        return torch.full(
            (shape[0], channel_dim, shape[2], shape[3]),
            fill_value=fill_value,
            dtype=dtype,
            device=self.device,
        )

    def __hash__(self):
        return joblib.hash(self())

    def __call__(self):
        if isinstance(self.init, str):
            if self.init not in self.init_mapping:
                raise ValueError(f"`init` function '{self.init}' not recognised")
            init_fn = self.init_mapping[self.init]
        else:
            init_fn = self.init
        return init_fn(*self.args, **self.kwargs)
