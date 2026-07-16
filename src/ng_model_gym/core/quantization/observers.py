# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import logging
import types

import torch
from torchao.quantization.pt2e import (
    FixedQParamsFakeQuantize,
    FusedMovingAvgObsFakeQuantize,
)
from torchao.quantization.pt2e.fake_quantize import FakeQuantizeBase
from torchao.quantization.pt2e.observer import ObserverBase

logger = logging.getLogger(__name__)


def _sanitize_fake_quant_qparams(m: FakeQuantizeBase) -> None:
    """Keep fake-quant buffers within the range accepted by Torch kernels."""
    observer = getattr(m, "activation_post_process", None)
    if observer is None or not hasattr(m, "scale") or not hasattr(m, "zero_point"):
        return

    with torch.no_grad():
        m.scale.nan_to_num_(nan=1.0, posinf=1.0, neginf=1.0)
        m.scale.clamp_(min=float(getattr(m, "eps", 2e-12)))

        quant_min = getattr(observer, "quant_min", None)
        quant_max = getattr(observer, "quant_max", None)
        if quant_min is None or quant_max is None:
            return

        if torch.is_floating_point(m.zero_point):
            m.zero_point.nan_to_num_(
                nan=0.0,
                posinf=float(quant_max),
                neginf=float(quant_min),
            )
        m.zero_point.clamp_(quant_min, quant_max)


def _set_fake_quant_observer_enabled(m: FakeQuantizeBase, enabled: bool) -> None:
    """Set an observer flag without replacing a registered tensor buffer."""
    observer_enabled = getattr(m, "observer_enabled", None)
    if isinstance(observer_enabled, torch.Tensor):
        observer_enabled.fill_(1 if enabled else 0)
    elif observer_enabled is not None:
        setattr(m, "observer_enabled", enabled)


def _is_zero_point_range_error(error: RuntimeError) -> bool:
    return "`zero_point` must be between `quant_min` and `quant_max`" in str(error)


def _should_sanitize_after_fused_fake_quant() -> bool:
    return not torch.compiler.is_compiling()


def _disable_fake_quant_observer_after_qparam_error(
    m: FakeQuantizeBase, reason: str
) -> None:
    _sanitize_fake_quant_qparams(m)
    _set_fake_quant_observer_enabled(m, False)
    m._disable_observer_after_zero_point_error = True
    if not getattr(m, "_logged_zero_point_observer_disable", False):
        logger.warning(
            "Disabling observer updates for %s after %s.",
            m.__class__.__name__,
            reason,
        )
        m._logged_zero_point_observer_disable = True


def _fake_quant_flag_enabled(m: FakeQuantizeBase, flag: str) -> bool:
    """Read a fake-quant flag on eager-only paths."""
    value = getattr(m, flag, None)
    if isinstance(value, torch.Tensor):
        return bool(value[0].item())
    return bool(value)


def _update_qparams_from_observer(m: FakeQuantizeBase, x: torch.Tensor) -> None:
    """Update registered qparam buffers, recovering from invalid observations."""
    old_min = getattr(m.activation_post_process, "min_val", None)
    old_max = getattr(m.activation_post_process, "max_val", None)
    old_min = old_min.clone() if isinstance(old_min, torch.Tensor) else None
    old_max = old_max.clone() if isinstance(old_max, torch.Tensor) else None

    try:
        m.activation_post_process(x.detach())
        scale, zero_point = m.activation_post_process.calculate_qparams()
    except (AssertionError, RuntimeError) as error:
        if old_min is not None and hasattr(m.activation_post_process, "min_val"):
            m.activation_post_process.min_val.copy_(old_min)
        if old_max is not None and hasattr(m.activation_post_process, "max_val"):
            m.activation_post_process.max_val.copy_(old_max)
        _disable_fake_quant_observer_after_qparam_error(
            m,
            f"observer qparam calculation failed: {error}",
        )
        return

    scale = scale.to(m.scale.device)
    zero_point = zero_point.to(m.zero_point.device)
    if m.scale.shape != scale.shape:
        m.scale.resize_(scale.shape)
        m.zero_point.resize_(zero_point.shape)
    m.scale.copy_(scale)
    m.zero_point.copy_(zero_point)


def _apply_fake_quant(m: FakeQuantizeBase, x: torch.Tensor) -> torch.Tensor:
    if not _fake_quant_flag_enabled(m, "fake_quant_enabled"):
        return x

    if getattr(m, "is_per_channel", False):
        return torch.fake_quantize_per_channel_affine(
            x,
            m.scale,
            m.zero_point,
            m.ch_axis,
            m.activation_post_process.quant_min,
            m.activation_post_process.quant_max,
        )

    return torch.fake_quantize_per_tensor_affine(
        x,
        m.scale,
        m.zero_point,
        m.activation_post_process.quant_min,
        m.activation_post_process.quant_max,
    )


def _forward_unfused_with_sanitized_qparams(
    m: FakeQuantizeBase, x: torch.Tensor
) -> torch.Tensor:
    """Run eager fake quant with qparam sanitation and one recovery retry."""
    if getattr(m, "_disable_observer_after_zero_point_error", False):
        _set_fake_quant_observer_enabled(m, False)

    _sanitize_fake_quant_qparams(m)
    if _fake_quant_flag_enabled(m, "observer_enabled"):
        _update_qparams_from_observer(m, x)
        _sanitize_fake_quant_qparams(m)

    try:
        return _apply_fake_quant(m, x)
    except RuntimeError as error:
        if not _is_zero_point_range_error(error):
            raise
        _disable_fake_quant_observer_after_qparam_error(
            m,
            "torch fake-quant produced an out-of-range zero_point",
        )
        return _apply_fake_quant(m, x)
    finally:
        _sanitize_fake_quant_qparams(m)


def _forward_fused_with_sanitized_qparams(
    m: FusedMovingAvgObsFakeQuantize, x: torch.Tensor
) -> torch.Tensor:
    if getattr(m, "_disable_observer_after_zero_point_error", False):
        _set_fake_quant_observer_enabled(m, False)

    try:
        output = torch.fused_moving_avg_obs_fake_quant(
            x,
            m.observer_enabled,
            m.fake_quant_enabled,
            m.activation_post_process.min_val,
            m.activation_post_process.max_val,
            m.scale,
            m.zero_point,
            m.activation_post_process.averaging_constant,
            m.activation_post_process.quant_min,
            m.activation_post_process.quant_max,
            m.ch_axis,
            m.is_per_channel,
            m.is_symmetric_quant,
        )
    except RuntimeError as error:
        if not _is_zero_point_range_error(error):
            raise
        _disable_fake_quant_observer_after_qparam_error(
            m,
            "torch fused fake-quant produced an out-of-range zero_point",
        )
        return _forward_unfused_with_sanitized_qparams(m, x)

    if _should_sanitize_after_fused_fake_quant():
        _sanitize_fake_quant_qparams(m)
    return output


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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization using sanitized quantization parameters."""
        return _forward_fused_with_sanitized_qparams(self, X)


class FixedQParamsFakeQuantizeFix(FixedQParamsFakeQuantize):
    """Compile-friendly, sanitized fixed-qparams fake quantization."""

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization using sanitized quantization parameters."""
        if not torch.compiler.is_compiling():
            return _forward_unfused_with_sanitized_qparams(self, X)

        observer = self.activation_post_process
        if getattr(self, "is_per_channel", False):
            quantized = torch.fake_quantize_per_channel_affine(
                X,
                self.scale,
                self.zero_point,
                self.ch_axis,
                observer.quant_min,
                observer.quant_max,
            )
        else:
            quantized = torch.fake_quantize_per_tensor_affine(
                X,
                self.scale,
                self.zero_point,
                observer.quant_min,
                observer.quant_max,
            )
        enabled = self.fake_quant_enabled.to(dtype=torch.bool)
        return torch.where(enabled, quantized, X)


def replace_fixed_qparams_fake_quant(module: torch.nn.Module) -> int:
    """Recursively replace stock fixed-qparams fake-quant modules."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, FixedQParamsFakeQuantizeFix):
            continue
        if isinstance(child, FixedQParamsFakeQuantize):
            observer_ctr = getattr(child, "_observer_ctr", None)
            if observer_ctr is None:
                continue
            replacement = FixedQParamsFakeQuantizeFix(observer=observer_ctr)
            replacement.load_state_dict(child.state_dict(), strict=True)
            replacement.to(device=child.scale.device)
            replacement.train(child.training)
            setattr(module, name, replacement)
            replaced += 1
            continue
        replaced += replace_fixed_qparams_fake_quant(child)
    return replaced


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
