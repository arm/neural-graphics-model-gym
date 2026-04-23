# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

"""
Routines to auto-generate custom Autograd functions for Slang module bindings, removing
the need to write specific fwd/bwd functions.

Do not access this module directly. Instead, use load_slang_module() and class SlangOutput
in slang_utils.
"""

import logging
from inspect import Parameter, Signature
from typing import Any, Callable, Dict, List, Union

import joblib
import numpy as np
import torch
from slangtorch.util.builtin_wrappers import DiffTensorView
from slangtorch.util.wrapper import WrappedFunction

DEFAULT_FWD_BLOCK_SIZE = 512
DEFAULT_BWD_BLOCK_SIZE = 256

OUT_CONSTRUCTOR_ARG_NAME = "out_constructors"
DISPATCH_SIZE_ARG_NAME = "dispatch_size"
BLOCK_SIZE_ARG_NAME = "block_size"
CONSTRUCTOR_ARGS = [
    OUT_CONSTRUCTOR_ARG_NAME,
    DISPATCH_SIZE_ARG_NAME,
    BLOCK_SIZE_ARG_NAME,
]


class WrappedSlangFunction:
    """
    Lazily builds a PyTorch-callable wrapper around a Slang `WrappedFunction`.

    The wrapper separates runtime tensor inputs from control parameters
    (`out_constructors`, dispatch size, block size), then generates:
    - a forward launcher that allocates outputs and dispatches the Slang kernel
    - a backward launcher that reconstructs `DiffTensorView` gradients and
      dispatches the Slang backward kernel

    The generated call path is exposed through a dynamic
    `torch.autograd.Function` class, with a one-time smoke test to validate
    forward/backward interoperability before normal execution.
    """

    def __init__(self, func_name: str, slang_func: WrappedFunction):
        self.slang_func: WrappedFunction = slang_func
        self.func_name: str = func_name
        self.initialised: bool = False
        self.arg_in_pos_mapping = {}
        self.autograd_tested: bool = False
        self._arg_wrappers_by_name: Dict[str, Any] = {}

    def _is_diff_tensor_wrapper(self, name_or_wrapper: Union[str, Any]) -> bool:
        if isinstance(name_or_wrapper, str):
            wrapper_entry = self._arg_wrappers_by_name.get(name_or_wrapper)
            if wrapper_entry is None:
                try:
                    idx = self.slang_func.argnames.index(name_or_wrapper)
                except ValueError:
                    return False
                wrapper_entry = self.slang_func.argwrappers[idx]
        else:
            wrapper_entry = name_or_wrapper

        if wrapper_entry is None:
            return False

        if isinstance(wrapper_entry, (tuple, list)):
            candidate = wrapper_entry[0]
        else:
            candidate = wrapper_entry

        if candidate is DiffTensorView or isinstance(candidate, DiffTensorView):
            return True

        name = getattr(candidate, "__name__", "")
        if name == "DiffTensorView":
            return True

        fields = getattr(candidate, "_fields", None)
        diff_fields = getattr(DiffTensorView, "_fields", None)
        if fields and diff_fields and fields == diff_fields:
            return True

        candidate_repr = repr(candidate)
        return "DiffTensorView" in candidate_repr

    @staticmethod
    def _unwrap_tensor_value(value: Any) -> Any:
        """Unwrap known tensor-like wrappers to their underlying tensor payload."""
        if isinstance(value, DiffTensorView):
            value = value.value
        if hasattr(value, "as_tensor") and callable(value.as_tensor):
            return value.as_tensor()
        return value

    def make_forward(
        self,
        call_params: Dict[str, Any],
        out_constructors: Dict[str, Callable],
        dispatch_grid_size: Dict[str, Any],
    ):
        """Generate forward functions"""
        self._arg_wrappers_by_name = dict(
            zip(self.slang_func.argnames, self.slang_func.argwrappers)
        )
        return_ann = List[torch.Tensor]
        params = [
            Parameter(name, Parameter.POSITIONAL_OR_KEYWORD, annotation=type(var))
            for name, var in call_params.items()
        ]
        sig = Signature(params, return_annotation=return_ann)

        def impl(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            in_args = {name: bound.arguments[name] for name in call_params.keys()}
            out_args = {
                name: self._unwrap_tensor_value(func())
                for name, func in out_constructors.items()
            }
            slang_args = in_args | out_args
            kernel = self.slang_func(**slang_args)
            block_size = dispatch_grid_size.get(
                BLOCK_SIZE_ARG_NAME, DEFAULT_FWD_BLOCK_SIZE
            )
            first_out = next(iter(out_args.values()))
            dispatch_size = dispatch_grid_size.get(
                DISPATCH_SIZE_ARG_NAME,
                [
                    first_out.shape[0],
                    first_out.shape[2],
                    first_out.shape[3],
                ],
            )
            grid_x = int((np.prod(dispatch_size) + block_size - 1) // block_size)
            kernel.launchRaw(blockSize=(block_size, 1, 1), gridSize=(grid_x, 1, 1))
            return tuple(out_args.values())

        impl.__signature__ = sig
        impl.__annotations__ = {p.name: p.annotation for p in params}
        impl.__annotations__["return"] = return_ann

        return torch.compiler.disable(impl)

    def make_backward(
        self,
        call_params: Dict[str, Any],
        call_params_grad: Dict[str, Any],
        out_constructors: Dict[str, Callable],
        dispatch_grid_size: Dict[str, Any],
    ):
        """Generate backward functions"""
        arg_names = list(self.slang_func.argnames)
        self._arg_wrappers_by_name = dict(zip(arg_names, self.slang_func.argwrappers))

        provided_args = set((call_params | call_params_grad | out_constructors).keys())
        expected_args = set(arg_names)
        if provided_args != expected_args:
            raise ValueError(
                "Mismatch between provided arguments and Slang function signature: "
                f"expected {sorted(expected_args)}, got {sorted(provided_args)}"
            )

        def _annotation_for(name: str):
            return (
                List[torch.Tensor]
                if self._is_diff_tensor_wrapper(name)
                else torch.Tensor
            )

        params = [
            Parameter(
                name, Parameter.POSITIONAL_OR_KEYWORD, annotation=_annotation_for(name)
            )
            for name in arg_names
        ]

        out_names = list(out_constructors.keys())
        if not out_names:
            raise ValueError(
                "`out_constructors` must define at least one output for backward execution."
            )
        if len(out_names) > 1:
            ret_signature = List[torch.Tensor]
        else:
            ret_signature = _annotation_for(out_names[0])

        sig = Signature(params, return_annotation=ret_signature)

        def impl(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            in_args = {name: bound.arguments[name] for name in call_params.keys()}
            in_args_grad = {}
            for name in call_params_grad.keys():
                base_tensor = self._unwrap_tensor_value(bound.arguments[name])
                in_args_grad[name] = DiffTensorView(
                    base_tensor, torch.zeros_like(base_tensor)
                )

            out_args = {}
            for name in out_constructors:
                value = bound.arguments[name]
                out_args[name] = (
                    value
                    if isinstance(value, DiffTensorView)
                    else self._unwrap_tensor_value(value)
                )
            slang_args = in_args | in_args_grad | out_args
            kernel = self.slang_func.bwd(**slang_args)
            block_size = dispatch_grid_size.get(
                BLOCK_SIZE_ARG_NAME, DEFAULT_BWD_BLOCK_SIZE
            )
            first_tensor = self._unwrap_tensor_value(next(iter(out_args.values())))
            dispatch_size = dispatch_grid_size.get(
                DISPATCH_SIZE_ARG_NAME,
                [first_tensor.shape[0], first_tensor.shape[2], first_tensor.shape[3]],
            )
            grid_x = int((np.prod(dispatch_size) + block_size - 1) // block_size)
            kernel.launchRaw(blockSize=(block_size, 1, 1), gridSize=(grid_x, 1, 1))
            return [v.grad for v in in_args_grad.values()]

        impl.__signature__ = sig
        impl.__annotations__ = {p.name: p.annotation for p in params}
        impl.__annotations__["return"] = ret_signature

        return torch.compiler.disable(impl)

    def _process_args(self, **kwargs):
        call_params = {
            name: var for name, var in kwargs.items() if name not in CONSTRUCTOR_ARGS
        }
        # call_params_grad is a subset of call_params. They are not disjoint
        call_params_grad = {
            name: var
            for name, var in call_params.items()
            if self._is_diff_tensor_wrapper(name)
        }
        out_constructors = kwargs.get(OUT_CONSTRUCTOR_ARG_NAME, {})
        dispatch_grid_size = {}
        dispatch_size = kwargs.get(DISPATCH_SIZE_ARG_NAME)
        if dispatch_size is not None:
            dispatch_grid_size[DISPATCH_SIZE_ARG_NAME] = dispatch_size
        if BLOCK_SIZE_ARG_NAME in kwargs:
            dispatch_grid_size[BLOCK_SIZE_ARG_NAME] = kwargs[BLOCK_SIZE_ARG_NAME]
        return call_params, call_params_grad, out_constructors, dispatch_grid_size

    def _create_autograd_functions(self, *args, **kwargs):
        self.autograd_tested = False

        def forward(ctx, *inputs):
            outputs = custom_ops_fwd(*inputs)
            ctx.backward_fn = custom_ops_bwd

            tensor_inputs = []
            tensor_indices = []
            for i, v in enumerate(inputs):
                if isinstance(v, torch.Tensor):
                    tensor_inputs.append(v)
                    tensor_indices.append(i)
                else:
                    setattr(ctx, f"non_tensor_{i}", v)

            ctx.tensor_indices = tensor_indices
            ctx.num_inputs = len(inputs)

            tensors_to_save = tensor_inputs + (
                list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            )
            ctx.save_for_backward(*tensors_to_save)

            if len(outputs) == 1:
                return outputs[0]
            return outputs

        def backward(ctx, *grads):
            def _to_diff_grad_tensor(out_name, saved_output, grad_value):
                if grad_value is None:
                    return torch.zeros_like(saved_output)
                if isinstance(grad_value, torch.Tensor):
                    return grad_value
                if isinstance(grad_value, (list, tuple)):
                    return (
                        grad_value[0]
                        if len(grad_value) > 0
                        else torch.zeros_like(saved_output)
                    )
                raise TypeError(
                    f"Unsupported gradient type '{type(grad_value).__name__}' "
                    + f"for DiffTensorView output '{out_name}'."
                )

            saved = list(ctx.saved_tensors)
            tensor_indices = getattr(ctx, "tensor_indices", [])
            tensor_count = len(tensor_indices)
            tensor_inputs = saved[:tensor_count]
            saved_outputs = saved[tensor_count:]

            grouped = []
            for i, grad_value in enumerate(grads):
                out_name = self.arg_out_pos_mapping[i]
                saved_output = saved_outputs[i]
                if self._is_diff_tensor_wrapper(out_name):
                    grad_tensor = _to_diff_grad_tensor(
                        out_name=out_name,
                        saved_output=saved_output,
                        grad_value=grad_value,
                    )
                    grouped.append(DiffTensorView(saved_output, grad_tensor))
                else:
                    grouped.append(saved_output)

            tensor_inputs_by_index = dict(zip(tensor_indices, tensor_inputs))
            num_inputs = len(self.arg_in_pos_mapping)
            call_inputs = [
                (
                    tensor_inputs_by_index[i]
                    if i in tensor_inputs_by_index
                    else getattr(ctx, f"non_tensor_{i}")
                )
                for i in range(num_inputs)
            ]
            cat_inputs = call_inputs + grouped

            grad_outputs = ctx.backward_fn(*cat_inputs)

            diff_input_mask = [
                self._is_diff_tensor_wrapper(self.arg_in_pos_mapping[i])
                for i in range(num_inputs)
            ]
            grad_iter = iter(grad_outputs)
            return tuple(
                next(grad_iter) if is_diff else None for is_diff in diff_input_mask
            )

        self.auto_grad_class = type(
            f"{self.func_name}_auto_grad_func",
            (torch.autograd.Function,),
            {"forward": staticmethod(forward), "backward": staticmethod(backward)},
        )
        (
            call_params,
            call_params_grad,
            out_constructors,
            dispatch_grid_size,
        ) = self._process_args(*args, **kwargs)

        # pylint: disable=unnecessary-comprehension
        self.arg_in_pos_mapping = {i: name for i, name in enumerate(call_params)}
        self.arg_out_pos_mapping = {i: name for i, name in enumerate(out_constructors)}
        # pylint: enable=unnecessary-comprehension

        custom_ops_fwd = self.make_forward(
            call_params=call_params,
            out_constructors=out_constructors,
            dispatch_grid_size=dispatch_grid_size,
        )

        custom_ops_bwd = self.make_backward(
            call_params=call_params,
            call_params_grad=call_params_grad,
            out_constructors=out_constructors,
            dispatch_grid_size=dispatch_grid_size,
        )
        self.initialised = True

    def _smoke_test_autograd_functions(self, *args, **kwargs):
        class _AutogradTestContext:
            """Minimal autograd-like context used by wrapper smoke tests.

            Stores tensors passed through ``save_for_backward`` so the generated
            backward function can be invoked without relying on PyTorch internals.
            """

            def __init__(self):
                self.saved_tensors = ()

            def save_for_backward(self, *tensors):
                """Mimic `torch.autograd` context storage for smoke tests.

                Persists the provided tensors on `saved_tensors` so the synthetic
                backward pass can consume the same interface as a real autograd
                context.
                """
                self.saved_tensors = tensors

        def _clone_input(value):
            if isinstance(value, torch.Tensor):
                return value.clone()
            if isinstance(value, (list, tuple)):
                return type(value)(_clone_input(v) for v in value)
            return value

        def _zeros_like_structure(value):
            if isinstance(value, torch.Tensor):
                return torch.zeros_like(value)
            if isinstance(value, (list, tuple)):
                return tuple(_zeros_like_structure(v) for v in value)
            raise TypeError(
                f"Unsupported output type '{type(value).__name__}' for autograd smoke test."
            )

        call_params, _, _, _ = self._process_args(*args, **kwargs)
        test_inputs = tuple(_clone_input(v) for v in call_params.values())

        ctx = _AutogradTestContext()
        outputs = self.auto_grad_class.forward(ctx, *test_inputs)
        if isinstance(outputs, torch.Tensor):
            outputs_seq = (outputs,)
        else:
            outputs_seq = tuple(outputs)

        grad_placeholders = tuple(_zeros_like_structure(out) for out in outputs_seq)
        self.auto_grad_class.backward(ctx, *grad_placeholders)
        self.autograd_tested = True

    @torch.compiler.disable
    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            raise ValueError(
                "Auto-generated autograd functions do not support positional arguments."
            )

        if not self.initialised:
            self._create_autograd_functions(*args, **kwargs)
        if not self.autograd_tested and self.slang_func.bwd_wrapped_fn is not None:
            self._smoke_test_autograd_functions(*args, **kwargs)

        call_params, _, _, _ = self._process_args(*args, **kwargs)

        return self.auto_grad_class.apply(*call_params.values())


class WrappedSlangFunctionManager:
    """
    Manages multiple WrappedSlangFunctions based on output tensor signatures.
    This allows to use the same Slang function name for multiple output types.
    """

    def __init__(self, var_name, func):
        self.var_name: str = var_name
        self.func: Callable = func
        # Map [hash -> WrappedSlangFunction]
        self._funcs: Dict[str, WrappedSlangFunction] = {}

    @staticmethod
    def _dict_signature(data: dict) -> str:
        value = ""
        for k in data:
            value += joblib.hash(k + joblib.hash(data[k]))
        return joblib.hash(value)

    def __call__(self, *args, **kwargs):
        out_constructors = kwargs.get(OUT_CONSTRUCTOR_ARG_NAME, {})
        sig = self._dict_signature(out_constructors)
        if sig not in self._funcs:
            # New signature
            logging.debug(
                f"Slang loader: Created new WrappedSlangFunction for Slang func: {self.var_name}"
            )
            func = WrappedSlangFunction(self.var_name + str(sig), self.func)
            self._funcs[sig] = func
            return func(*args, **kwargs)

        # Else not new, call correct function
        return self._funcs[sig](*args, **kwargs)


def convert_slang_wrapped_function(module):
    """Main entry point into this module."""
    members = module.__dict__
    slang_functions = {
        var_name: WrappedSlangFunctionManager(var_name, func)
        for var_name, func in members.items()
        if isinstance(func, WrappedFunction)
    }
    return type(module.__name__, (object,), slang_functions)
