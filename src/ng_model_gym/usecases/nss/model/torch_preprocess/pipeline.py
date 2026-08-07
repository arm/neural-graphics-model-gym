# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
"""Native Torch orchestration for NSS v1 preprocessing."""

# pylint: disable=too-many-branches

from typing import Mapping

import torch
import torch.nn.functional as torch_functional

from .depth import compute_depth_clip, depth_scatter, lq_disocclusion_mask
from .derivative import (
    calculate_rec709_luminance_derivative,
    calculate_ycocg_derivative,
)
from .sampling import (
    bilinear_sample,
    encode_nearest_offsets,
    encode_packed_nearest_offsets,
    find_nearest_depth_4x4,
    find_nearest_depth_4x4_from_pixels,
    gather_pixels,
    make_pixel_grid,
    reflect_indices,
)


def _karis_tonemap(color: torch.Tensor) -> torch.Tensor:
    """Translate ``tonemap_forward(..., TonemapMode.Karis)``."""

    color = torch.maximum(color, torch.zeros_like(color))
    maximum = torch.maximum(torch.maximum(color[:, 0:1], color[:, 1:2]), color[:, 2:3])
    color = color * torch.reciprocal(1.0 + maximum)
    return torch.clamp(color, 0.0, 1.0)


def _batch_values(source: torch.Tensor, channels: int) -> torch.Tensor:
    """Return per-batch metadata as ``(N, channels, 1, 1)``."""

    return source.detach().reshape(source.shape[0], -1)[:, :channels, None, None]


def _threshold_motion(motion: torch.Tensor) -> torch.Tensor:
    """Apply the shader's strict 0.1 pixel motion threshold."""

    motion = motion.detach()
    length = torch.sqrt(
        motion[:, 0:1] * motion[:, 0:1] + motion[:, 1:2] * motion[:, 1:2]
    )
    return motion * (length > 0.1).to(motion.dtype)


def _warp_history(
    history: torch.Tensor,
    uv: torch.Tensor,
    exposure: torch.Tensor,
) -> torch.Tensor:
    """Translate ``WarpHistory`` while retaining value gradients."""

    sampled = bilinear_sample(history, uv.detach(), clamp_to_edge=False)
    return _karis_tonemap(sampled * exposure.detach())


def _warped_history_for_pixels(
    history: torch.Tensor,
    motion: torch.Tensor,
    depth: torch.Tensor,
    pixels: torch.Tensor,
    inverse_input_size: torch.Tensor,
    exposure: torch.Tensor,
) -> torch.Tensor:
    """Translate ``BuildPreProcessLaneData`` followed by ``WarpHistory``."""

    _, nearest_coord, _ = find_nearest_depth_4x4_from_pixels(depth, pixels)
    lane_motion = _threshold_motion(gather_pixels(motion.detach(), nearest_coord))
    lane_uv = (pixels.to(history.dtype) + 0.5) * inverse_input_size
    reprojection_uv = lane_uv - lane_motion.permute(0, 2, 3, 1) * inverse_input_size
    return _warp_history(history, reprojection_uv, exposure)


def _average_warped_history_2x2(
    history: torch.Tensor,
    motion: torch.Tensor,
    depth: torch.Tensor,
    process_coords: torch.Tensor,
    inverse_input_size: torch.Tensor,
    exposure: torch.Tensor,
) -> torch.Tensor:
    """Translate ``AverageWarpedHistory2x2`` in literal lane order."""

    input_height, input_width = depth.shape[-2:]
    base = process_coords * 2
    lane_00 = base
    lane_10 = base + base.new_tensor((1, 0))
    lane_01 = base + base.new_tensor((0, 1))
    lane_11 = base + base.new_tensor((1, 1))

    def clamp_lane(lane: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                lane[..., 0].clamp(0, input_height - 1),
                lane[..., 1].clamp(0, input_width - 1),
            ),
            dim=-1,
        )

    history_00 = _warped_history_for_pixels(
        history,
        motion,
        depth,
        clamp_lane(lane_00),
        inverse_input_size,
        exposure,
    )
    history_10 = _warped_history_for_pixels(
        history,
        motion,
        depth,
        clamp_lane(lane_10),
        inverse_input_size,
        exposure,
    )
    history_01 = _warped_history_for_pixels(
        history,
        motion,
        depth,
        clamp_lane(lane_01),
        inverse_input_size,
        exposure,
    )
    history_11 = _warped_history_for_pixels(
        history,
        motion,
        depth,
        clamp_lane(lane_11),
        inverse_input_size,
        exposure,
    )
    return (((history_00 + history_10) + history_01) + history_11) * 0.25


def _calculate_motion_detector(
    motion: torch.Tensor,
    render_size: torch.Tensor,
) -> torch.Tensor:
    """Translate ``CalculateMotionDetector`` over a spatial tensor."""

    render_size = render_size.detach()
    pixel_min = torch.sqrt(
        (1.0 / render_size[:, 0:1]) * (1.0 / render_size[:, 0:1])
        + (1.0 / render_size[:, 1:2]) * (1.0 / render_size[:, 1:2])
    )
    pixel_max = torch.sqrt(
        (200.0 / render_size[:, 0:1]) * (200.0 / render_size[:, 0:1])
        + (200.0 / render_size[:, 1:2]) * (200.0 / render_size[:, 1:2])
    )
    pixel_denominator = torch.reciprocal(pixel_max - pixel_min)
    normalized = motion.detach() / render_size
    motion_length = torch.sqrt(
        normalized[:, 0:1] * normalized[:, 0:1]
        + normalized[:, 1:2] * normalized[:, 1:2]
    )
    clamped = torch.minimum(torch.maximum(motion_length, pixel_min), pixel_max)
    return torch.sqrt((clamped - pixel_min) * pixel_denominator).detach()


def _pad_process_tensor(
    source: torch.Tensor,
    padded_spatial: tuple[int, int],
) -> torch.Tensor:
    """Pad a logical process tensor with an exactly-zero bottom/right fringe."""

    pad_height = padded_spatial[0] - source.shape[-2]
    pad_width = padded_spatial[1] - source.shape[-1]
    return torch_functional.pad(source, (0, pad_width, 0, pad_height))


def _match_slang_input_vjp(value: torch.Tensor) -> torch.Tensor:
    """Preserve the primal while matching the generated Slang input VJP.

    The generated ``DiffTensorView`` backward wrapper accumulates two copies of
    each differentiable input load. NSS-v1 training therefore observes a 2x
    VJP for both history and temporal feedback. Express the same first-order
    contract with native autograd and an explicit detach boundary.
    """

    detached = value.detach()
    return detached + (value - detached) * 2.0


@torch.compiler.disable
def preprocess_torch(
    preprocess_input: Mapping[str, torch.Tensor],
    *,
    input_shape: tuple[int, int, int, int],
    process_shape: tuple[int, int, int, int],
    pad_shape: tuple[int, int, int, int],
    depth_shape: tuple[int, int, int, int],
    preprocess_half_res_input: bool,
    depth_scatter_quarter_res_input: bool,
    packed_nearest_offset_quad: bool,
    nss_v1_luma_derivative: bool,
    nss_v1_low_mid_luma_derivative: bool,
    motion_key: str = "motion_lr",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create NSS v1 autoencoder inputs using only native Torch operations.

    This is the eager Torch counterpart of ``depth_scatter``, optional
    ``lq_disocclusion_mask``, and ``pre_process``. Spatial coordinates retain
    the shaders' ``(row, column)`` convention. Shape tuples are calculated by
    ``NSSV1Model`` before dispatch, so this entrypoint preserves the existing
    public shape-validation boundary.

    Only ``history`` and ``temporal_params_tm1`` contribute gradients to the
    returned autoencoder input, specifically channels 0:3 and 7:11. Every
    selection coordinate, mask, metadata value, recurrent derivative value,
    and auxiliary output is detached at the corresponding shader boundary.
    Whole-pipeline forward/VJP parity tests are required for this entrypoint.
    """

    color = preprocess_input["colour_linear"]
    history = preprocess_input["history"]
    motion = preprocess_input[motion_key]
    depth = preprocess_input["depth"]
    feedback_tm1 = preprocess_input["temporal_params_tm1"]
    derivative_tm1 = preprocess_input["derivative_tm1"]
    jitter = preprocess_input["jitter"]
    exposure_input = preprocess_input["exposure"]
    render_size_input = preprocess_input["render_size"]
    depth_params = preprocess_input["depth_params"]

    if tuple(color.shape) != tuple(input_shape):
        raise ValueError(
            f"NSS-v1 Torch preprocess input shape mismatch: "
            f"expected {input_shape}, got {tuple(color.shape)}."
        )

    batch_size, _, input_height, input_width = input_shape
    process_height, process_width = process_shape[-2:]
    padded_height, padded_width = pad_shape[-2:]
    dtype = color.dtype
    device = color.device
    input_size = torch.tensor((input_height, input_width), device=device, dtype=dtype)
    process_size = torch.tensor(
        (process_height, process_width), device=device, dtype=dtype
    )
    padded_size = torch.tensor(
        (padded_height, padded_width), device=device, dtype=dtype
    )
    depth_size = torch.tensor(depth_shape[-2:], device=device, dtype=dtype)
    scale = process_size / input_size
    inv_scale = torch.reciprocal(scale)
    inverse_input_size = torch.reciprocal(input_size)
    inverse_process_size = torch.reciprocal(process_size)
    inverse_padded_size = torch.reciprocal(padded_size)
    inverse_depth_size = torch.reciprocal(depth_size)
    exposure = _batch_values(exposure_input, 1)
    render_size = _batch_values(render_size_input, 2)

    reconstructed_depth = depth_scatter(
        motion,
        depth,
        render_size_input,
        output_shape=depth_shape[-2:],
        quarter_res_input=depth_scatter_quarter_res_input,
    ).detach()
    if preprocess_half_res_input and depth_scatter_quarter_res_input:
        disocclusion_lq = lq_disocclusion_mask(
            motion,
            depth,
            reconstructed_depth,
            depth_params,
            render_size_input,
            output_shape=depth_shape[-2:],
        ).detach()
    else:
        disocclusion_lq = depth.detach()

    padded_coords = make_pixel_grid(
        (padded_height, padded_width),
        batch_size=batch_size,
        device=device,
    )
    if preprocess_half_res_input:
        process_coords = reflect_indices(padded_coords, (process_height, process_width))
        scaled_coords = (process_coords.to(dtype) + 0.5) * inv_scale
        ref_coords = torch.floor(scaled_coords).to(torch.int64)
        ref_coords = torch.stack(
            (
                ref_coords[..., 0].clamp(0, input_height - 1),
                ref_coords[..., 1].clamp(0, input_width - 1),
            ),
            dim=-1,
        )
    else:
        process_coords = padded_coords
        ref_coords = reflect_indices(padded_coords, (input_height, input_width)).to(
            torch.int64
        )

    uv = (ref_coords.to(dtype) + 0.5) * inverse_input_size
    uv_padded = (padded_coords.to(dtype) + 0.5) * inverse_padded_size

    if preprocess_half_res_input:
        (
            depth_dilated,
            nearest_coord,
            nearest_offset,
        ) = find_nearest_depth_4x4_from_pixels(depth, ref_coords)
    else:
        depth_dilated, nearest_coord, nearest_offset = find_nearest_depth_4x4(
            depth, uv, inverted=False
        )
    if preprocess_half_res_input:
        sampled_motion = gather_pixels(motion.detach(), nearest_coord)
    else:
        nearest_uv = (nearest_coord.to(dtype) + 0.5) / input_size
        sampled_motion = bilinear_sample(
            motion.detach(), nearest_uv, clamp_to_edge=True
        )
    sampled_motion = _threshold_motion(sampled_motion)
    sampled_motion_coords = sampled_motion.permute(0, 2, 3, 1)

    reprojection_uv = uv - sampled_motion_coords * inverse_input_size
    jitter_value = _batch_values(jitter, 2).permute(0, 2, 3, 1)
    unjittered_uv = uv - jitter_value * inverse_input_size

    if preprocess_half_res_input:
        scaled_depth_coords = (
            (ref_coords.to(dtype) + 0.5) * inverse_input_size * depth_size
        )
        depth_coords = torch.floor(scaled_depth_coords).to(torch.int64)
        depth_coords = torch.stack(
            (
                depth_coords[..., 0].clamp(0, depth_shape[-2] - 1),
                depth_coords[..., 1].clamp(0, depth_shape[-1] - 1),
            ),
            dim=-1,
        )
        # Keep this translation visible even though the half-res shader only
        # consumes the low-quality disocclusion UV below.
        _ = (
            depth_coords.to(dtype) + 0.5
        ) * inverse_depth_size - sampled_motion_coords * inverse_input_size
        reprojection_padded_uv = (
            uv_padded - (sampled_motion_coords * scale * inverse_padded_size).detach()
        )
        disocclusion_uv = (process_coords.to(dtype) + 0.5) * inverse_process_size
        disocclusion_mask = bilinear_sample(
            disocclusion_lq,
            disocclusion_uv,
            clamp_to_edge=True,
        )[:, 0:1].detach()
    else:
        depth_coords = torch.div(ref_coords, 2, rounding_mode="floor")
        reprojection_depth_uv = (
            depth_coords.to(dtype) + 0.5
        ) * inverse_depth_size - sampled_motion_coords * inverse_input_size
        reprojection_padded_uv = (
            uv_padded - sampled_motion_coords * inverse_padded_size
        ).detach()
        disocclusion_mask = (
            compute_depth_clip(
                reconstructed_depth,
                reprojection_depth_uv,
                render_size_input,
                depth_dilated,
                depth_params,
                inverted=False,
            )
            .unsqueeze(1)
            .detach()
        )

    unjittered_color = bilinear_sample(
        color.detach(), unjittered_uv, clamp_to_edge=True
    )
    unjittered_color = _karis_tonemap(unjittered_color * exposure).detach()

    if preprocess_half_res_input:
        warped_history = _average_warped_history_2x2(
            history,
            motion.detach(),
            depth.detach(),
            process_coords,
            inverse_input_size,
            exposure,
        )
        derivative_uv = reprojection_padded_uv
        derivative_inv_dims = inverse_padded_size
    else:
        warped_history = _warp_history(history, reprojection_uv.detach(), exposure)
        derivative_uv = reprojection_uv.detach()
        derivative_inv_dims = inverse_input_size

    if nss_v1_luma_derivative:
        derivative_state, instability = calculate_ycocg_derivative(
            color,
            exposure_input,
            ref_coords,
            derivative_tm1,
            derivative_uv,
            derivative_inv_dims,
            disocclusion_mask,
            inv_scale,
            low_mid_luma_derivative=nss_v1_low_mid_luma_derivative,
        )
    else:
        derivative_state, instability = calculate_rec709_luminance_derivative(
            unjittered_color,
            derivative_tm1,
            derivative_uv,
            disocclusion_mask,
        )

    feedback = bilinear_sample(
        feedback_tm1,
        reprojection_padded_uv.detach(),
        clamp_to_edge=False,
    )
    disocclusion_binary = (disocclusion_mask > 0.01).to(dtype)
    feedback = torch.lerp(
        feedback,
        torch.zeros_like(feedback),
        disocclusion_binary,
    )
    warped_history = _match_slang_input_vjp(warped_history)
    feedback = _match_slang_input_vjp(feedback)
    motion_detector = _calculate_motion_detector(sampled_motion, render_size)

    input_tensor = torch.cat(
        (
            warped_history,
            unjittered_color,
            motion_detector,
            feedback,
            instability,
        ),
        dim=1,
    ).contiguous()

    process_disocclusion = disocclusion_mask[:, :, :process_height, :process_width]
    disocclusion_output = torch.cat(
        (process_disocclusion, torch.zeros_like(process_disocclusion)), dim=1
    ).detach()

    if preprocess_half_res_input:
        process_derivative = derivative_state[:, :, :process_height, :process_width]
        derivative_output = _pad_process_tensor(
            process_derivative, (padded_height, padded_width)
        ).detach()
    else:
        derivative_output = derivative_state[:, :, :input_height, :input_width].detach()

    if packed_nearest_offset_quad:
        packed_base_coords = (
            make_pixel_grid(
                (process_height, process_width),
                batch_size=batch_size,
                device=device,
            )
            * 2
        )
        process_offsets = encode_packed_nearest_offsets(
            depth.detach(), packed_base_coords, dtype=dtype
        )
        nearest_offset_output = _pad_process_tensor(
            process_offsets, (padded_height, padded_width)
        ).detach()
    else:
        nearest_offset_output = encode_nearest_offsets(
            nearest_offset[:, :process_height, :process_width], dtype=dtype
        ).detach()

    return (
        input_tensor,
        derivative_output.contiguous(),
        disocclusion_output.contiguous(),
        nearest_offset_output.contiguous(),
    )
