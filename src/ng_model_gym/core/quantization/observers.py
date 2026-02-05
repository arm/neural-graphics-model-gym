# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import types

import torch
from torchao.quantization.pt2e import FusedMovingAvgObsFakeQuantize
from torchao.quantization.pt2e.fake_quantize import FakeQuantizeBase
from torchao.quantization.pt2e.observer import ObserverBase


class FusedMovingAvgObsFakeQuantizeFix(FusedMovingAvgObsFakeQuantize):
    """
    Implements fix for exporting zero pt and scale values when lowering
    PyTorch via `convert_pt2e`.

    During QAT / PyTorch inference the forward pass of `FusedMovingAvgObsFakeQuantize` calculates
    it's own quantization parameters in a call to `torch.fused_moving_avg_obs_fake_quant`
    which populates `self.scale` and `self.zero_point` here:
    https://github.com/pytorch/ao/blob/eb8617756682b35f98727ea43e5c6bc3d37e416e/torchao/quantization/pt2e/fake_quantize.py#L417

    For CPU, this results in calling this method here:
    https://github.com/pytorch/pytorch/blob/63360e64da814de8ce271f1e4b6e2380a03b585e/aten/src/ATen/native/quantized/cpu/fused_obs_fake_quant.cpp#L143

    Which calculates the Qparams here:
    https://github.com/pytorch/pytorch/blob/63360e64da814de8ce271f1e4b6e2380a03b585e/aten/src/ATen/native/quantized/cpu/fused_obs_fake_quant.cpp#L60

    With the business code here:
    https://github.com/pytorch/pytorch/blob/63360e64da814de8ce271f1e4b6e2380a03b585e/aten/src/ATen/native/quantized/cpu/QuantUtils.h#L70C33-L70C57

    We're yet to rigorously compare, to work out the difference,
    but `calculate_qparams` eventually calls this method here:
    https://github.com/pytorch/ao/blob/eb8617756682b35f98727ea43e5c6bc3d37e416e/torchao/quantization/pt2e/observer.py#L351

    The results returned from `_calculate_qparams` have small differences between
    the `self.scale` and `self.zero_point` used during training.
    So, for now we're ignoring this method in favour of returning the exact
    `zero_point` and `scale` computed during QAT:
    """

    @torch.jit.export
    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Getter for scale and zero points."""
        return self.scale, self.zero_point


def _freeze_one(m: torch.nn.Module):
    """
    Disable stats collection for a single observer / fake-quant module.
    """
    # --- Fake-quant blocks (wrap an internal observer) ----------------------
    if isinstance(m, FakeQuantizeBase):
        if hasattr(m, "disable_observer"):
            m.disable_observer()
            return

    # --- Stand-alone observers ---------------------------------------------
    if isinstance(m, ObserverBase):
        if hasattr(m, "disable"):
            m.disable()  # newer observers
            return

    # --- Common flags used in older / experimental code --------------------
    for flag in ("observer_enabled", "enabled"):
        if hasattr(m, flag):
            attr = getattr(m, flag)
            if isinstance(attr, torch.Tensor):
                attr.zero_()
            else:
                setattr(m, flag, False)
            return

    # --- Fallback: make the observer a no-op -------------------------------
    if m.__class__.__name__.lower().endswith("observer"):
        m.forward = types.MethodType(lambda self, x: x, m)


def freeze_all_observers(model: torch.nn.Module):
    """
    Walk every module (recursively) and freeze any pt2e observer / fake-quant.
    Call *after* model.eval().
    """
    for mod in model.modules():
        _freeze_one(mod)


def _enable_one(m: torch.nn.Module):
    """
    Re-enable statistics collection for a single observer / fake-quant module.
    Handles all the paths that freeze_all_observers() covered.
    """
    # --- Fake-quant blocks --------------------------------------------------
    if isinstance(m, FakeQuantizeBase):
        if hasattr(m, "enable_observer"):
            m.enable_observer()
            return

    # --- Stand-alone observers ---------------------------------------------
    if isinstance(m, ObserverBase):
        if hasattr(m, "enable"):
            m.enable()
            return

    # --- Common flags -------------------------------------------------------
    for flag in ("observer_enabled", "enabled"):
        if hasattr(m, flag):
            attr = getattr(m, flag)
            if isinstance(attr, torch.Tensor):
                attr.fill_(1)  # tensor buffer
            else:
                setattr(m, flag, True)
            return

    # --- Restore forward if it was monkey-patched to a no-op ---------------
    if m.__class__.__name__.lower().endswith("observer") and hasattr(
        m, "_orig_forward"
    ):
        m.forward = m._orig_forward


def enable_all_observers(model: torch.nn.Module):
    """
    Walks *recursively* over the model and re-activates any pt2e
    observers / fake-quant modules so their min-max stats update again.

    Call before resuming QAT or further calibration.
    """
    for mod in model.modules():
        _enable_one(mod)
