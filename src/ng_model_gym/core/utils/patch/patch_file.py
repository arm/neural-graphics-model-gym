# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSES/LicenseRef-BSD-3-TorchAO.txt file in the top-level directory.
# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0 AND LicenseRef-BSD-3-TorchAO
# pylint: skip-file

import importlib
import logging
from typing import Any, Callable

import torch
from torch.fx import GraphModule

logger = logging.getLogger(__name__)


def _get_aten_graph_module_for_pattern(
    pattern: Callable,
    example_inputs: tuple[Any, ...],
    is_cuda: bool = False,
    **kwargs,
) -> GraphModule:
    """
    Convert the pattern to an FX graph with decomposed aten ops.
    """
    if is_cuda:
        example_inputs = tuple(
            [x.cuda() if isinstance(x, torch.Tensor) else x for x in example_inputs]
        )

    aten_pattern = torch.export.export(
        pattern,  # type: ignore[arg-type]
        example_inputs,
        kwargs,
        strict=True,
    ).module(
        check_guards=False
    )  # Added fix to work with ExecuTorch.

    aten_pattern.graph.eliminate_dead_code()  # type: ignore[operator, union-attr]
    aten_pattern.recompile()  # type: ignore[operator]

    # ep.module() adds copy_ nodes for the mutated inputs.
    # For patterns, it doesn't matter
    for node in aten_pattern.graph.nodes:  # type: ignore[union-attr]
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.copy_.default
            and len(node.users) == 0
        ):
            aten_pattern.graph.erase_node(node)  # type: ignore[operator, union-attr]

    aten_pattern.graph.eliminate_dead_code()  # type: ignore[operator, union-attr]
    aten_pattern.recompile()  # type: ignore[operator]

    return aten_pattern  # type: ignore[return-value]


def _apply_patch():
    target_module = "torchao.quantization.pt2e.utils"
    target_function = "_get_aten_graph_module_for_pattern"
    module = importlib.import_module(target_module)

    if hasattr(module, target_function):
        module._get_aten_graph_module_for_pattern = _get_aten_graph_module_for_pattern
        logger.debug("Patch applied.")
    else:
        logger.debug(f"Patch not applied: target function {target_function} not found.")


_apply_patch()
