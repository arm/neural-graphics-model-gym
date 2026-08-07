# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
"""Torch translations of the NSS v1 temporal instability detectors."""

import torch

from .sampling import bilinear_sample, reflect_indices


def _lerp(start: torch.Tensor, end: torch.Tensor, weight) -> torch.Tensor:
    """Match the shader's ``lerp(start, end, weight)`` expression."""

    if not isinstance(start, torch.Tensor):
        start = torch.full_like(weight, start)
    if not isinstance(end, torch.Tensor):
        end = torch.full_like(weight, end)
    return torch.lerp(start, end, weight)


def _saturate(value: torch.Tensor) -> torch.Tensor:
    return torch.clamp(value, 0.0, 1.0)


def _gather_pixels(source: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Gather row/column integer coordinates into a channel-first tensor."""

    batch, channels, height, width = source.shape
    if coords.shape[0] == 1 and batch != 1:
        coords = coords.expand(batch, -1, -1, -1)
    rows = coords[..., 0].clamp(0, height - 1)
    cols = coords[..., 1].clamp(0, width - 1)
    linear = (rows * width + cols).reshape(batch, 1, -1)
    linear = linear.expand(-1, channels, -1)
    gathered = torch.gather(source.reshape(batch, channels, -1), 2, linear)
    return gathered.reshape(batch, channels, *coords.shape[1:3])


def _rgb_to_ycocg(rgb: torch.Tensor) -> torch.Tensor:
    """Translate ``RGBToYCoCg`` without reassociating its operations."""

    red = rgb[:, 0:1]
    green = rgb[:, 1:2]
    blue = rgb[:, 2:3]
    co = red - blue
    temporary = blue + co * 0.5
    cg = green - temporary
    luma = temporary + cg * 0.5
    return torch.cat((luma, co, cg), dim=1)


def _load_derivative_color(
    color: torch.Tensor,
    exposure: torch.Tensor,
    ref_coords: torch.Tensor,
    row_offset: int | torch.Tensor = 0,
    col_offset: int | torch.Tensor = 0,
) -> torch.Tensor:
    """Translate ``LoadColorForDerivativeAtPixel`` over a spatial tensor.

    Coordinates and source values are detached because the Slang input is a
    ``TensorView`` and each load is wrapped in ``no_diff``.
    """

    coords = ref_coords.detach().clone()
    coords[..., 0] = coords[..., 0] + row_offset
    coords[..., 1] = coords[..., 1] + col_offset
    coords = reflect_indices(coords, color.shape[-2:])
    sample = _gather_pixels(color.detach(), coords)
    batch_exposure = exposure.detach().reshape(color.shape[0], -1)[:, 0]
    sample = sample * batch_exposure[:, None, None, None]
    sample = torch.maximum(sample, torch.zeros_like(sample))
    return torch.sqrt(sample)


def _derivative_delta(
    ycocg_a: torch.Tensor,
    ycocg_b: torch.Tensor,
    chroma_weight: float,
) -> torch.Tensor:
    """Translate ``ComputeDerivativeDelta`` with explicit component order."""

    delta_luma = ycocg_a[:, 0:1] - ycocg_b[:, 0:1]
    delta_co = (ycocg_a[:, 1:2] - ycocg_b[:, 1:2]) * chroma_weight
    delta_cg = (ycocg_a[:, 2:3] - ycocg_b[:, 2:3]) * chroma_weight
    return torch.sqrt(
        delta_luma * delta_luma + delta_co * delta_co + delta_cg * delta_cg
    )


def calculate_ycocg_derivative(
    color: torch.Tensor,
    exposure: torch.Tensor,
    ref_coords: torch.Tensor,
    derivative_tm1: torch.Tensor,
    derivative_uv: torch.Tensor,
    derivative_inv_dims: torch.Tensor,
    disocclusion_mask: torch.Tensor,
    inv_scale: torch.Tensor,
    *,
    low_mid_luma_derivative: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate the current YCoCg ``CalculateLumaDerivative`` overload.

    ``ref_coords`` and ``derivative_uv`` use shader order ``(row, column)``.
    The returned tensors have shapes ``(N, 4, H, W)`` and ``(N, 1, H, W)``.
    Color, exposure, coordinates, disocclusion, and previous derivative state
    are explicit gradient stops, matching their Slang ``TensorView``/``no_diff``
    topology. Forward threshold fixtures and whole-preprocess parity cover this
    helper; its outputs intentionally do not participate in gradient parity.
    """

    color = color.detach()
    exposure = exposure.detach()
    ref_coords = ref_coords.detach()
    derivative_tm1 = derivative_tm1.detach()
    derivative_uv = derivative_uv.detach()
    derivative_inv_dims = derivative_inv_dims.detach()
    disocclusion_mask = disocclusion_mask.detach()
    inv_scale = inv_scale.detach()

    deriv_tm1 = bilinear_sample(
        derivative_tm1,
        derivative_uv,
        clamp_to_edge=False,
    ).detach()

    # Keep these constants local and in shader declaration order for easier
    # side-by-side review with pre_process.slang.
    derivative_dis_thresh = 0.01
    chroma_weight = 1.25
    recall_floor = 0.065
    recall_ceil = 0.420
    excursion_floor = 0.025
    excursion_ceil = 0.160
    mean_gate_floor = 0.070
    mean_gate_ceil = 0.230
    sustain_cold_floor = 0.177
    sustain_cold_ceil = 0.330
    sustain_hot_floor = 0.157
    sustain_hot_ceil = 0.305
    sustain_hysteresis_floor = 0.110
    sustain_hysteresis_ceil = 0.210
    sustain_support_alpha = 0.30
    hot_hold_floor = 0.180
    hot_hold_ceil = 0.280
    sustain_min_hot_hold = 0.12
    decay_min_hot_gate = 0.50
    sustain_strength = 0.80
    instability_rise_alpha_min = 0.08
    instability_rise_alpha_max = 0.22
    instability_fast_fall_alpha = 0.24
    instability_fall_alpha = 0.05
    spatial_support_scale = 0.75
    spatial_support_blend = 0.30

    if low_mid_luma_derivative:
        moire_temporal_floor = 0.10
        moire_temporal_ceil = 0.30
        moire_range_floor = 0.99
        moire_range_ceil = 0.999
        moire_range_scale = 0.50
        flat_temporal_floor = 0.015
        flat_temporal_ceil = 0.030
        flat_range_floor = 0.040
        flat_range_ceil = 0.120
        flat_blue_floor = 0.220
        flat_blue_ceil = 0.300
        flat_luma_floor = 0.450
        flat_luma_ceil = 1.050
        flat_rgb_b_floor = 0.800
        flat_rgb_b_ceil = 1.400
        flat_flicker_scale = 0.75

    ycocg_center = _rgb_to_ycocg(_load_derivative_color(color, exposure, ref_coords))
    ycocg_north = _rgb_to_ycocg(
        _load_derivative_color(color, exposure, ref_coords, col_offset=-1)
    )
    ycocg_south = _rgb_to_ycocg(
        _load_derivative_color(color, exposure, ref_coords, col_offset=1)
    )
    ycocg_east = _rgb_to_ycocg(
        _load_derivative_color(color, exposure, ref_coords, row_offset=1)
    )
    ycocg_west = _rgb_to_ycocg(
        _load_derivative_color(color, exposure, ref_coords, row_offset=-1)
    )

    delta_center = _derivative_delta(ycocg_center, deriv_tm1[:, :3], chroma_weight)
    delta_north = _derivative_delta(ycocg_center, ycocg_north, chroma_weight)
    delta_south = _derivative_delta(ycocg_center, ycocg_south, chroma_weight)
    delta_east = _derivative_delta(ycocg_center, ycocg_east, chroma_weight)
    delta_west = _derivative_delta(ycocg_center, ycocg_west, chroma_weight)
    spatial_delta_sum = delta_north + delta_south + delta_east + delta_west
    spatial_delta_max = torch.maximum(
        torch.maximum(delta_north, delta_south),
        torch.maximum(delta_east, delta_west),
    )
    prev_instability = deriv_tm1[:, 3:4]

    if low_mid_luma_derivative:
        temporal_input_step = torch.maximum(
            (inv_scale + 0.5).to(torch.int64),
            torch.ones_like(inv_scale, dtype=torch.int64),
        )
        row_step = temporal_input_step[0]
        col_step = temporal_input_step[1]
        ycocg_temporal_north = _rgb_to_ycocg(
            _load_derivative_color(color, exposure, ref_coords, col_offset=-col_step)
        )
        ycocg_temporal_south = _rgb_to_ycocg(
            _load_derivative_color(color, exposure, ref_coords, col_offset=col_step)
        )
        ycocg_temporal_east = _rgb_to_ycocg(
            _load_derivative_color(color, exposure, ref_coords, row_offset=row_step)
        )
        ycocg_temporal_west = _rgb_to_ycocg(
            _load_derivative_color(color, exposure, ref_coords, row_offset=-row_step)
        )

        north_uv = derivative_uv.clone()
        south_uv = derivative_uv.clone()
        east_uv = derivative_uv.clone()
        west_uv = derivative_uv.clone()
        north_uv[..., 1] -= derivative_inv_dims[1]
        south_uv[..., 1] += derivative_inv_dims[1]
        east_uv[..., 0] += derivative_inv_dims[0]
        west_uv[..., 0] -= derivative_inv_dims[0]
        deriv_tm1_north = bilinear_sample(
            derivative_tm1, north_uv, clamp_to_edge=False
        ).detach()
        deriv_tm1_south = bilinear_sample(
            derivative_tm1, south_uv, clamp_to_edge=False
        ).detach()
        deriv_tm1_east = bilinear_sample(
            derivative_tm1, east_uv, clamp_to_edge=False
        ).detach()
        deriv_tm1_west = bilinear_sample(
            derivative_tm1, west_uv, clamp_to_edge=False
        ).detach()

        delta_temporal_north = _derivative_delta(
            ycocg_temporal_north, deriv_tm1_north[:, :3], chroma_weight
        )
        delta_temporal_south = _derivative_delta(
            ycocg_temporal_south, deriv_tm1_south[:, :3], chroma_weight
        )
        delta_temporal_east = _derivative_delta(
            ycocg_temporal_east, deriv_tm1_east[:, :3], chroma_weight
        )
        delta_temporal_west = _derivative_delta(
            ycocg_temporal_west, deriv_tm1_west[:, :3], chroma_weight
        )
        range_temporal_north = _derivative_delta(
            ycocg_center, ycocg_temporal_north, chroma_weight
        )
        range_temporal_south = _derivative_delta(
            ycocg_center, ycocg_temporal_south, chroma_weight
        )
        range_temporal_east = _derivative_delta(
            ycocg_center, ycocg_temporal_east, chroma_weight
        )
        range_temporal_west = _derivative_delta(
            ycocg_center, ycocg_temporal_west, chroma_weight
        )
        temporal_moire_max = torch.maximum(
            torch.maximum(delta_center, delta_temporal_north),
            torch.maximum(
                torch.maximum(delta_temporal_south, delta_temporal_east),
                delta_temporal_west,
            ),
        )
        current_moire_range = torch.maximum(
            torch.maximum(range_temporal_north, range_temporal_south),
            torch.maximum(range_temporal_east, range_temporal_west),
        )
        flat_temporal_min = torch.minimum(
            torch.minimum(delta_center, delta_temporal_north),
            torch.minimum(
                torch.minimum(delta_temporal_south, delta_temporal_east),
                delta_temporal_west,
            ),
        )

    spatial_support = torch.clamp_min(spatial_delta_sum - spatial_delta_max, 0.0) * (
        1.0 / 3.0
    )
    supported_instability = (
        _lerp(delta_center, spatial_support, spatial_support_blend)
        * spatial_support_scale
    )

    if low_mid_luma_derivative:
        moire_temporal_gate = _saturate(
            (temporal_moire_max - moire_temporal_floor)
            * (1.0 / (moire_temporal_ceil - moire_temporal_floor))
        )
        moire_range_entry = (
            _saturate(
                (current_moire_range - moire_range_floor)
                * (1.0 / (moire_range_ceil - moire_range_floor))
            )
            * moire_temporal_gate
            * moire_range_scale
        )
        flat_temporal_gate = _saturate(
            (flat_temporal_min - flat_temporal_floor)
            * (1.0 / (flat_temporal_ceil - flat_temporal_floor))
        )
        flat_range_gate = 1.0 - _saturate(
            (current_moire_range - flat_range_floor)
            * (1.0 / (flat_range_ceil - flat_range_floor))
        )
        flat_blue_bias = (-0.75 * ycocg_center[:, 1:2]) - (0.5 * ycocg_center[:, 2:3])
        flat_blue_gate = _saturate(
            (flat_blue_bias - flat_blue_floor)
            * (1.0 / (flat_blue_ceil - flat_blue_floor))
        )
        flat_luma_gate = _saturate(
            (ycocg_center[:, 0:1] - flat_luma_floor) * (1.0 / 0.10)
        ) * (1.0 - _saturate((ycocg_center[:, 0:1] - flat_luma_ceil) * (1.0 / 0.20)))
        flat_rgb_blue = ycocg_center[:, 0:1] - (
            0.5 * (ycocg_center[:, 1:2] + ycocg_center[:, 2:3])
        )
        flat_rgb_blue_gate = _saturate(
            (flat_rgb_blue - flat_rgb_b_floor) * (1.0 / 0.10)
        ) * (1.0 - _saturate((flat_rgb_blue - flat_rgb_b_ceil) * (1.0 / 0.20)))
        flat_surface_gate = (
            flat_range_gate * flat_blue_gate * flat_luma_gate * flat_rgb_blue_gate
        )
        flat_flicker_entry = flat_temporal_gate * flat_surface_gate * flat_flicker_scale

    recall_excursion = torch.clamp_min(supported_instability - prev_instability, 0.0)
    recall_score = _saturate(
        (supported_instability - recall_floor) * (1.0 / (recall_ceil - recall_floor))
    )
    excursion_score = _saturate(
        (recall_excursion - excursion_floor)
        * (1.0 / (excursion_ceil - excursion_floor))
    )
    mean_gate = _saturate(
        (supported_instability - mean_gate_floor)
        * (1.0 / (mean_gate_ceil - mean_gate_floor))
    )
    raw_entry = torch.sqrt(recall_score) * torch.sqrt(excursion_score) * mean_gate
    if low_mid_luma_derivative:
        raw_entry = torch.maximum(
            raw_entry, torch.maximum(moire_range_entry, flat_flicker_entry)
        )

    sustain_heat = _saturate(
        (prev_instability - sustain_hysteresis_floor)
        * (1.0 / (sustain_hysteresis_ceil - sustain_hysteresis_floor))
    )
    sustain_support = _lerp(
        prev_instability, supported_instability, sustain_support_alpha
    )
    sustain_floor = _lerp(sustain_cold_floor, sustain_hot_floor, sustain_heat)
    sustain_ceil = _lerp(sustain_cold_ceil, sustain_hot_ceil, sustain_heat)
    sustain_gate = _saturate(
        (sustain_support - sustain_floor)
        * torch.reciprocal(sustain_ceil - sustain_floor)
    )
    sustain_gate = sustain_gate * sustain_gate
    hot_hold = _saturate(
        (prev_instability - hot_hold_floor) * (1.0 / (hot_hold_ceil - hot_hold_floor))
    )
    hot_hold = hot_hold * hot_hold
    hot_hold_gate = hot_hold * sustain_min_hot_hold
    carry_gate = torch.maximum(sustain_gate, hot_hold_gate)
    raw_sustain = prev_instability * carry_gate * sustain_strength
    raw_instability = torch.maximum(raw_entry, raw_sustain)

    decay_gate = torch.maximum(
        sustain_gate, sustain_heat * sustain_heat * decay_min_hot_gate
    )
    fall_alpha = _lerp(instability_fast_fall_alpha, instability_fall_alpha, decay_gate)
    rise_support = torch.sqrt(recall_score * mean_gate)
    rise_alpha = _lerp(
        instability_rise_alpha_min, instability_rise_alpha_max, rise_support
    )
    instability_alpha = torch.where(
        raw_instability > prev_instability, rise_alpha, fall_alpha
    )
    filtered_instability = _lerp(prev_instability, raw_instability, instability_alpha)

    output_alpha = torch.where(
        filtered_instability > prev_instability,
        torch.full_like(filtered_instability, 0.75),
        torch.full_like(filtered_instability, 0.80),
    )
    visible_instability = _lerp(prev_instability, filtered_instability, output_alpha)
    if low_mid_luma_derivative:
        visible_instability = torch.maximum(
            visible_instability,
            torch.maximum(moire_range_entry, flat_flicker_entry),
        )

    derivative_state = torch.cat((ycocg_center, filtered_instability), dim=1)
    disocclusion_binary = (disocclusion_mask > derivative_dis_thresh).to(color.dtype)
    uninitialized_state = (
        torch.sum(torch.abs(deriv_tm1), dim=1, keepdim=True) < 1e-4
    ).to(color.dtype)
    history_valid = 1.0 - disocclusion_binary
    visible_instability = visible_instability * history_valid

    reset_state = torch.cat(
        (ycocg_center, torch.zeros_like(filtered_instability)), dim=1
    )
    seeded_state = _lerp(derivative_state, reset_state, disocclusion_binary)
    visible_instability = _lerp(
        visible_instability, torch.zeros_like(visible_instability), uninitialized_state
    )
    state = _lerp(seeded_state, reset_state, uninitialized_state)
    return state.detach(), visible_instability.detach()


def calculate_rec709_luminance_derivative(
    unjittered_color: torch.Tensor,
    derivative_tm1: torch.Tensor,
    derivative_uv: torch.Tensor,
    disocclusion_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate the Rec.709 luminance ``CalculateLumaDerivative`` overload.

    Inputs and outputs use ``(N, C, H, W)`` tensors; ``derivative_uv`` is
    ``(N, H, W, 2)`` in row/column order. All values are detached to mirror the
    shader's non-differentiable source tensors. Forward threshold fixtures and
    whole-preprocess parity provide the required verification.
    """

    unjittered_color = unjittered_color.detach()
    derivative_tm1_sample = bilinear_sample(
        derivative_tm1.detach(),
        derivative_uv.detach(),
        clamp_to_edge=False,
    ).detach()
    disocclusion_mask = disocclusion_mask.detach()

    derivative_min = 0.05
    derivative_max = 0.3
    derivative_alpha = 0.1
    derivative_disocclusion_threshold = 0.01

    derivative_max_pow_reciprocal = 1.0 / (derivative_max * derivative_max**0.5)
    luma_tm1 = derivative_tm1_sample[:, 1:2]
    luma_derivative_tm1 = derivative_tm1_sample[:, 0:1]
    # Shader luminance is the standard Rec.709 dot product.
    luma = (
        unjittered_color[:, 0:1] * 0.2126
        + unjittered_color[:, 1:2] * 0.7152
        + unjittered_color[:, 2:3] * 0.0722
    )

    luma_derivative = torch.abs(luma - luma_tm1)
    clipped = torch.minimum(
        luma_derivative, torch.full_like(luma_derivative, derivative_max)
    )
    clipped = clipped * (luma_derivative >= derivative_min).to(clipped.dtype)
    curved = clipped * torch.sqrt(clipped) * derivative_max_pow_reciprocal

    applied_alpha = _lerp(
        derivative_alpha,
        derivative_alpha * 0.1,
        torch.clamp(luma_derivative_tm1, 0.0, derivative_max) * (1.0 / derivative_max),
    )
    filtered_derivative = _lerp(luma_derivative_tm1, curved, applied_alpha)
    filtered_derivative = filtered_derivative * (
        disocclusion_mask <= derivative_disocclusion_threshold
    ).to(filtered_derivative.dtype)
    state = torch.cat(
        (
            filtered_derivative,
            luma,
            torch.zeros_like(luma),
            torch.zeros_like(luma),
        ),
        dim=1,
    )
    return state.detach(), filtered_derivative.detach()
