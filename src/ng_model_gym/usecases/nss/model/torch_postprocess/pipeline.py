# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
"""Native Torch orchestration for NSS v1 postprocessing."""

# pylint: disable=too-many-arguments,too-many-locals

from typing import NamedTuple

import torch

from .filter import filter_color
from .sampling import (
    EPS,
    fused_multiply_add,
    load_motion,
    MAX_HALF,
    output_coordinates,
    reproject_history_uv,
    sample_bilinear,
    sample_catmull_rom,
    sample_temporal_params,
)

_SLANG_BACKWARD_BLOCK_SIZE = 256


class _PostprocessDimensions(NamedTuple):
    """Validated dimensions required by postprocess orchestration."""

    batch: int
    output_size_yx: tuple[int, int]
    preprocess_size_yx: torch.Tensor


class _SlangBackwardIdentity(torch.autograd.Function):
    """Preserve values while matching Slang's wrapped CUDA backward launch."""

    @staticmethod
    def forward(
        _context, tensor: torch.Tensor
    ) -> torch.Tensor:  # pylint: disable=arguments-differ
        """Return the exact input without allocating another tensor."""

        return tensor

    @staticmethod
    def backward(_context, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        """Multiply gradients by each wrapped output thread's visit count."""

        batch, _, height, width = gradient.shape
        thread_count = batch * height * width
        launched_threads = (
            (thread_count + _SLANG_BACKWARD_BLOCK_SIZE - 1)
            // _SLANG_BACKWARD_BLOCK_SIZE
            * _SLANG_BACKWARD_BLOCK_SIZE
        )
        positions = torch.arange(thread_count, device=gradient.device)
        multiplicity = 1 + (launched_threads - 1 - positions) // thread_count
        multiplicity = multiplicity.reshape(batch, 1, height, width).to(gradient.dtype)
        return (gradient * multiplicity,)


def slang_backward_identity(tensor: torch.Tensor) -> torch.Tensor:
    """Apply Slang's modulo-wrapped padded-thread multiplicity in backward."""

    return _SlangBackwardIdentity.apply(tensor)


def karis_forward(linear_rgb: torch.Tensor) -> torch.Tensor:
    """Apply the shader's component-ordered Karis tone map."""

    nonnegative = torch.maximum(linear_rgb, torch.zeros_like(linear_rgb))
    maximum = torch.maximum(
        torch.maximum(nonnegative[:, 0:1], nonnegative[:, 1:2]),
        nonnegative[:, 2:3],
    )
    mapped = nonnegative * torch.reciprocal(1.0 + maximum)
    return mapped.clamp(0.0, 1.0)


def karis_inverse(mapped_rgb: torch.Tensor) -> torch.Tensor:
    """Invert Karis after clamping to the shader's finite HDR range."""

    nonnegative = torch.maximum(mapped_rgb, torch.zeros_like(mapped_rgb))
    hdr_limit = nonnegative.new_full((1, 3, 1, 1), MAX_HALF)
    clamped = torch.minimum(nonnegative, karis_forward(hdr_limit))
    maximum = torch.maximum(
        torch.maximum(clamped[:, 0:1], clamped[:, 1:2]), clamped[:, 2:3]
    )
    return clamped * torch.reciprocal(1.0 - maximum)


def rectify_history(
    m1: torch.Tensor,
    m2: torch.Tensor,
    warped_history: torch.Tensor,
    theta: torch.Tensor,
    gamma: torch.Tensor,
    reset: torch.Tensor | float,
    onscreen: torch.Tensor | float,
    *,
    contract_variance: bool = True,
) -> torch.Tensor:
    """Clamp history to moments, then apply reset and onscreen gates."""

    if contract_variance:
        variance = fused_multiply_add(-m1, m1, m2)
    else:
        # The bilinear shader specialization keeps this subtraction unfused.
        variance = m2 - m1 * m1
    variance = torch.maximum(torch.abs(variance), variance.new_tensor(EPS))
    sigma = torch.sqrt(variance) * gamma
    reset_value = (
        reset.detach()
        if isinstance(reset, torch.Tensor)
        else m1.new_tensor(reset).detach()
    )
    history_clamped = torch.lerp(
        m1,
        torch.maximum(torch.minimum(warped_history, m1 + sigma), m1 - sigma),
        reset_value,
    )
    return torch.lerp(history_clamped, warped_history, theta * onscreen * reset_value)


def _validate_float_nchw(name: str, tensor: torch.Tensor) -> None:
    """Validate the shared public float32 NCHW tensor contract."""

    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
        raise ValueError(f"{name} must be a torch.Tensor in NCHW layout.")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must use torch.float32; got {tensor.dtype}.")
    if any(size <= 0 for size in tensor.shape):
        raise ValueError(f"{name} must have positive N, C, H, and W dimensions.")


def _validate_postprocess_inputs(  # pylint: disable=too-many-branches
    *,
    in_color: torch.Tensor,
    in_history: torch.Tensor,
    in_kpn_params: torch.Tensor,
    in_temporal_params: torch.Tensor,
    in_motion: torch.Tensor,
    in_nearest_depth_off: torch.Tensor,
    in_exposure: torch.Tensor,
    in_offset_lut: torch.Tensor,
    in_idx_modulo: torch.Tensor,
    in_reset: torch.Tensor,
    output_shape: tuple[int, int, int, int],
    preprocess_half_res_input: bool,
    packed_nearest_offset_quad: bool,
    filter_kernel_taps: int,
) -> _PostprocessDimensions:
    """Validate shape-derived requirements before sampling or gathering."""

    named_tensors = {
        "in_color": in_color,
        "in_history": in_history,
        "in_kpn_params": in_kpn_params,
        "in_temporal_params": in_temporal_params,
        "in_motion": in_motion,
        "in_nearest_depth_off": in_nearest_depth_off,
        "in_exposure": in_exposure,
        "in_offset_lut": in_offset_lut,
        "in_idx_modulo": in_idx_modulo,
        "in_reset": in_reset,
    }
    for name, tensor in named_tensors.items():
        _validate_float_nchw(name, tensor)

    if (
        not isinstance(output_shape, tuple)
        or len(output_shape) != 4
        or any(not isinstance(size, int) or size <= 0 for size in output_shape)
    ):
        raise ValueError("output_shape must contain four positive integers.")
    if not isinstance(filter_kernel_taps, int) or filter_kernel_taps <= 0:
        raise ValueError("filter_kernel_taps must be a positive integer.")

    batch, color_channels, input_height, input_width = in_color.shape
    if output_shape[0] != batch or output_shape[1] != 3:
        raise ValueError("output_shape must match the color batch and have 3 channels.")
    channel_minima = {
        "in_color": (color_channels, 3),
        "in_history": (in_history.shape[1], 3),
        "in_kpn_params": (in_kpn_params.shape[1], 1),
        "in_temporal_params": (in_temporal_params.shape[1], 3),
        "in_motion": (in_motion.shape[1], 2),
    }
    for name, (actual, minimum) in channel_minima.items():
        if actual < minimum:
            raise ValueError(f"{name} must have at least {minimum} channels.")
    expected_offset_channels = 2 if packed_nearest_offset_quad else 1
    if in_nearest_depth_off.shape[1] != expected_offset_channels:
        raise ValueError(
            "in_nearest_depth_off has the wrong channel count for its encoding."
        )
    if in_offset_lut.shape[1] != 6:
        raise ValueError("in_offset_lut must have exactly 6 channels.")
    if in_offset_lut.shape[-1] != filter_kernel_taps:
        raise ValueError(
            "in_offset_lut tap count must equal filter_kernel_taps; got "
            f"{in_offset_lut.shape[-1]} and {filter_kernel_taps}."
        )

    for name in (
        "in_history",
        "in_kpn_params",
        "in_temporal_params",
        "in_motion",
        "in_nearest_depth_off",
        "in_exposure",
        "in_offset_lut",
        "in_reset",
    ):
        if named_tensors[name].shape[0] != batch:
            raise ValueError(f"{name} batch must match in_color batch {batch}.")
    expected_idx_batch = batch if preprocess_half_res_input else None
    if expected_idx_batch is not None and in_idx_modulo.shape[0] != batch:
        raise ValueError("half-resolution in_idx_modulo must match the color batch.")
    if not preprocess_half_res_input and in_idx_modulo.shape[0] not in (1, batch):
        raise ValueError(
            "full-resolution in_idx_modulo batch must be 1 or color batch."
        )

    devices = {tensor.device for tensor in named_tensors.values()}
    if len(devices) != 1:
        raise ValueError("All postprocess inputs must use the same device.")
    if in_history.shape[-2:] != output_shape[-2:]:
        raise ValueError("in_history spatial shape must match output_shape.")
    if in_motion.shape[-2:] != (input_height, input_width):
        raise ValueError("in_motion spatial shape must match in_color.")
    if in_exposure.numel() != batch or in_reset.numel() != batch:
        raise ValueError("in_exposure and in_reset must hold one scalar per batch.")

    idx_value_count = (
        in_idx_modulo.shape[1] * in_idx_modulo.shape[2] * (in_idx_modulo.shape[3])
    )
    required_idx_values = 4 if preprocess_half_res_input else 2
    if idx_value_count < required_idx_values:
        raise ValueError(
            f"in_idx_modulo must contain at least {required_idx_values} values."
        )
    if preprocess_half_res_input:
        logical_height, logical_width = input_height // 2, input_width // 2
        preprocess_size = in_idx_modulo.detach().reshape(batch, -1)[:, 2:4]
    else:
        logical_height, logical_width = input_height, input_width
        preprocess_size = (
            in_color.new_tensor((input_height, input_width))
            .reshape(1, 2)
            .expand(batch, -1)
        )
    if (
        in_nearest_depth_off.shape[-2] < logical_height
        or in_nearest_depth_off.shape[-1] < logical_width
    ):
        raise ValueError("in_nearest_depth_off does not cover the logical extent.")

    return _PostprocessDimensions(
        batch=batch,
        output_size_yx=output_shape[-2:],
        preprocess_size_yx=preprocess_size,
    )


@torch.compiler.disable
def postprocess_torch(
    *,
    in_color: torch.Tensor,
    in_history: torch.Tensor,
    in_kpn_params: torch.Tensor,
    in_temporal_params: torch.Tensor,
    in_motion: torch.Tensor,
    in_nearest_depth_off: torch.Tensor,
    in_exposure: torch.Tensor,
    in_offset_lut: torch.Tensor,
    in_idx_modulo: torch.Tensor,
    in_reset: torch.Tensor,
    output_shape: tuple[int, int, int, int],
    preprocess_half_res_input: bool,
    use_sparse_filter_2x2: bool,
    use_history_catmull: bool,
    packed_nearest_offset_quad: bool,
    sharp_theta: bool,
    filter_kernel_taps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run differentiable native PyTorch NSS v1 postprocessing."""

    dimensions = _validate_postprocess_inputs(
        in_color=in_color,
        in_history=in_history,
        in_kpn_params=in_kpn_params,
        in_temporal_params=in_temporal_params,
        in_motion=in_motion,
        in_nearest_depth_off=in_nearest_depth_off,
        in_exposure=in_exposure,
        in_offset_lut=in_offset_lut,
        in_idx_modulo=in_idx_modulo,
        in_reset=in_reset,
        output_shape=output_shape,
        preprocess_half_res_input=preprocess_half_res_input,
        packed_nearest_offset_quad=packed_nearest_offset_quad,
        filter_kernel_taps=filter_kernel_taps,
    )
    batch = dimensions.batch
    output_size_yx = dimensions.output_size_yx
    exposure = in_exposure.detach().reshape(batch, 1, 1, 1)
    inverse_exposure = torch.reciprocal(exposure)
    reset = in_reset.detach().reshape(batch, 1, 1, 1)

    filtered = filter_color(
        in_color,
        in_kpn_params,
        in_offset_lut,
        in_idx_modulo,
        exposure,
        output_size_yx,
        tuple(in_temporal_params.shape[-2:]),
        preprocess_half_res_input=preprocess_half_res_input,
        use_sparse_filter_2x2=use_sparse_filter_2x2,
        filter_kernel_taps=filter_kernel_taps,
    )
    temporal = sample_temporal_params(
        in_temporal_params,
        output_size_yx,
        dimensions.preprocess_size_yx,
        sharp_theta=sharp_theta,
    )
    motion = load_motion(
        in_motion,
        in_nearest_depth_off,
        output_size_yx,
        preprocess_half_res_input=preprocess_half_res_input,
        packed_nearest_offset_quad=packed_nearest_offset_quad,
    )

    coordinates = output_coordinates(batch, output_size_yx, in_history.device)
    inverse_output_size = torch.reciprocal(in_history.new_tensor(output_size_yx))
    uv = (coordinates.to(in_history.dtype) + 0.5) * inverse_output_size
    reprojected_uv = reproject_history_uv(
        uv,
        motion.permute(0, 2, 3, 1),
        inverse_output_size,
    )
    onscreen = (
        (
            torch.all(reprojected_uv >= 0.0, dim=-1, keepdim=True)
            & torch.all(reprojected_uv <= 1.0, dim=-1, keepdim=True)
        )
        .permute(0, 3, 1, 2)
        .to(in_history.dtype)
        .detach()
    )

    history = in_history[:, :3]
    if use_history_catmull:
        warped_history = sample_catmull_rom(history, reprojected_uv)
    else:
        warped_history = sample_bilinear(history, reprojected_uv, clamp_to_edge=False)
    warped_history = torch.minimum(
        warped_history * exposure, warped_history.new_tensor(MAX_HALF)
    )
    rectified = rectify_history(
        filtered.m1,
        filtered.m2,
        warped_history,
        temporal.theta,
        temporal.gamma,
        reset,
        onscreen,
        contract_variance=use_history_catmull,
    )

    rectified_mapped = karis_forward(rectified)
    center_mapped = karis_forward(filtered.center_sample[:, :3])
    alpha = temporal.alpha * filtered.center_sample[:, 3:4] * reset
    accumulated = torch.lerp(rectified_mapped, center_mapped, alpha)
    accumulated = accumulated.clamp(0.0, 1.0 - EPS)
    output_linear = karis_inverse(accumulated) * inverse_exposure
    out_filtered_linear = filtered.m1 * inverse_exposure
    return (
        slang_backward_identity(output_linear),
        slang_backward_identity(out_filtered_linear),
    )
