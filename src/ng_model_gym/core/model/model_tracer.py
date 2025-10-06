# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Tuple

import torch
from torch import nn
from torch.utils._pytree import tree_map

from ng_model_gym.core.model.base_ng_model import BaseNGModel
from ng_model_gym.core.utils.tensor_types import TensorData


def model_tracer(ng_model: BaseNGModel, input_data: TensorData) -> Tuple[Any, ...]:
    """Trace PyTorch module to capture forward pass tensors"""

    # Move input data to GPU
    to_gpu = lambda x: x.to("cuda") if isinstance(x, torch.Tensor) else x
    # tree_map is an internal torch util to traverse containers with tensors
    input_data = tree_map(to_gpu, input_data)

    target_module_to_trace = ng_model.get_neural_network()

    captured = None

    class _StopForward(Exception):
        """Custom exception"""

    def capture_forward_input_callback(module: nn.Module, forward_input: Tuple[any]):
        """Get forward pass inputs of a module"""

        if module is target_module_to_trace:
            nonlocal captured

            captured = forward_input

            # Raise to stop forward pass
            raise _StopForward()

        raise RuntimeError("Model tracer did not capture valid module")

    hook = target_module_to_trace.register_forward_pre_hook(
        capture_forward_input_callback, with_kwargs=False
    )
    try:
        with torch.no_grad():
            try:
                # Run model forward with callback hook
                ng_model(input_data)
            except _StopForward:
                pass
    finally:
        hook.remove()

    match captured:
        case None | (None,):
            raise ValueError(
                "Model tracer capture is None. Most likely input_data to tracer is "
                "empty or something went wrong with data passed into model forward pass."
            )
        case _:
            return captured
