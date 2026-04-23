# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch
from slangtorch.util.builtin_wrappers import DiffTensorView

from ng_model_gym.core.model.shaders.slang_function_wrapper import WrappedSlangFunction


class _FakeKernel:
    def __init__(self):
        self.block_size = None
        self.grid_size = None

    def launchRaw(self, blockSize, gridSize):
        """Simulate the behavior of a Slang kernel launch by storing the block and grid sizes."""
        self.block_size = blockSize
        self.grid_size = gridSize


class _FakeWrappedFunction:
    def __init__(self):
        self.argnames = ["x", "y"]
        self.argwrappers = [DiffTensorView, torch.Tensor]
        self.fwd_kernel = _FakeKernel()
        self.bwd_kernel = _FakeKernel()
        self.last_fwd_kwargs = None
        self.last_bwd_kwargs = None

    def __call__(self, **kwargs):
        self.last_fwd_kwargs = kwargs
        return self.fwd_kernel

    def bwd(self, **kwargs):
        """Simulate the behavior of a Slang backward function by storing the input kwargs."""
        self.last_bwd_kwargs = kwargs
        return self.bwd_kernel


class _FakeGrowingTensorBuffer:
    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    def as_tensor(self):
        """Simulate the behavior of a growing tensor buffer by returning the underlying tensor."""
        return self._tensor


class TestSlangFunctionWrapper(unittest.TestCase):
    """
    Unit tests for the WrappedSlangFunction class, focusing on the make_forward and
    make_backward methods.
    """

    def test_make_forward_unwraps_as_tensor_outputs(self):
        """
        Test that make_forward correctly unwraps outputs as tensors
        and passes them to the Slang function.
        """
        slang_func = _FakeWrappedFunction()
        wrapped = WrappedSlangFunction("fake_forward", slang_func)

        x = torch.randn(2, 3, 4, 5)
        y_tensor = torch.randn(2, 1, 4, 5)

        forward = wrapped.make_forward(
            call_params={"x": x},
            out_constructors={"y": lambda: _FakeGrowingTensorBuffer(y_tensor)},
            dispatch_grid_size={},
        )

        outputs = forward(x=x)

        self.assertEqual(len(outputs), 1)
        self.assertTrue(torch.equal(outputs[0], y_tensor))
        self.assertIsInstance(slang_func.last_fwd_kwargs["y"], torch.Tensor)
        self.assertEqual(slang_func.fwd_kernel.block_size, (512, 1, 1))
        self.assertEqual(slang_func.fwd_kernel.grid_size, (1, 1, 1))

    def test_make_backward_unwraps_as_tensor_inputs_and_outputs(self):
        """
        Test that make_backward correctly unwraps inputs and outputs as tensors
        and passes them to the Slang function.
        """
        slang_func = _FakeWrappedFunction()
        wrapped = WrappedSlangFunction("fake_backward", slang_func)

        x = torch.randn(2, 3, 4, 5)
        y = torch.randn(2, 1, 4, 5)

        backward = wrapped.make_backward(
            call_params={"x": x},
            call_params_grad={"x": x},
            out_constructors={"y": object()},
            dispatch_grid_size={},
        )

        grad_outputs = backward(
            x=_FakeGrowingTensorBuffer(x),
            y=_FakeGrowingTensorBuffer(y),
        )

        self.assertEqual(len(grad_outputs), 1)
        self.assertEqual(tuple(grad_outputs[0].shape), tuple(x.shape))
        self.assertTrue(torch.all(grad_outputs[0] == 0))

        bwd_input = slang_func.last_bwd_kwargs["x"]
        self.assertIsInstance(bwd_input, DiffTensorView)
        self.assertIsInstance(bwd_input.value, torch.Tensor)
        self.assertIsInstance(bwd_input.grad, torch.Tensor)
        self.assertTrue(torch.equal(bwd_input.value, x))
        self.assertTrue(torch.all(bwd_input.grad == 0))

        self.assertIsInstance(slang_func.last_bwd_kwargs["y"], torch.Tensor)
        self.assertEqual(slang_func.bwd_kernel.block_size, (256, 1, 1))
        self.assertEqual(slang_func.bwd_kernel.grid_size, (1, 1, 1))
