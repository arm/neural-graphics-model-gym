# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
"""Native Torch translation of the NFRU network-input preprocessor."""

from __future__ import annotations

from numbers import Integral

import torch

from .sampling import (
    bilinear_sample,
    gather_pixels,
    make_uv_grid,
    shader_float,
    uv_to_pixels,
    validate_matrix,
    validate_nchw,
)

_DEPTH_LOG_K = 1200.0
_DISOCCLUSION_DEPTH_SEPARATION_SCALE = 1.37e-5
_DISOCCLUSION_REFERENCE_SIZE = (1080.0, 1920.0)
_RANDOM_OOB_SEED_STRIDE = 10000


def _length2(vector: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(vector[..., 0] * vector[..., 0] + vector[..., 1] * vector[..., 1])


def _length3(vector: torch.Tensor) -> torch.Tensor:
    squared_xy = vector[..., 0] * vector[..., 0] + vector[..., 1] * vector[..., 1]
    return torch.sqrt(squared_xy + vector[..., 2] * vector[..., 2])


def _normalize_depth(depth: torch.Tensor) -> torch.Tensor:
    # Avoid sharing the far-plane collision bias used only by scatter.
    return 1.0 - torch.log(-(_DEPTH_LOG_K * (depth - 1.0)) + 1.0) / torch.log(
        depth.new_tensor(_DEPTH_LOG_K + 1.0)
    )


def _matrix_vector(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nij,nhwj->nhwi", matrix, vector)


def _reproject_depth(
    uv: torch.Tensor, depth: torch.Tensor, matrix: torch.Tensor
) -> torch.Tensor:
    transformed_uv = torch.stack((uv[..., 1], 1.0 - uv[..., 0]), dim=-1)
    clip = torch.cat(
        (
            2.0 * transformed_uv - 1.0,
            depth.unsqueeze(-1),
            torch.ones_like(depth).unsqueeze(-1),
        ),
        dim=-1,
    )
    reprojected = _matrix_vector(matrix, clip)
    return 1.0 - reprojected[..., 2] / reprojected[..., 3]


def _view_depth(device_depth: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    return params[:, 1, None, None] / (device_depth - params[:, 0, None, None])


def _view_position(
    viewport_position: torch.Tensor,
    viewport_size: torch.Tensor,
    device_depth: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    z_value = _view_depth(device_depth, params)
    ndc = viewport_position / viewport_size * viewport_size.new_tensor(
        (2.0, -2.0)
    ) + viewport_size.new_tensor((-1.0, 1.0))
    x_value = params[:, 2, None, None] * ndc[..., 0] * z_value
    y_value = params[:, 3, None, None] * ndc[..., 1] * z_value
    return torch.stack((x_value, y_value, z_value), dim=-1)


def _single_tap_depth_clip(
    current_depth: torch.Tensor,
    previous_depth: torch.Tensor,
    depth_params: torch.Tensor,
    spatial_shape: tuple[int, int],
) -> torch.Tensor:
    current_view = _view_depth(current_depth, depth_params)
    previous_view = _view_depth(previous_depth, depth_params)
    difference = current_view - previous_view
    plane_depth = torch.maximum(previous_depth, current_depth)
    size = current_depth.new_tensor(spatial_shape)
    center_pixel = torch.floor(size * 0.5)
    corner_pixel = torch.zeros_like(size)
    batch = current_depth.shape[0]
    center = _view_position(
        center_pixel.view(1, 1, 1, 2).expand(batch, *current_depth.shape[1:], 2),
        size,
        plane_depth,
        depth_params,
    )
    corner = _view_position(
        corner_pixel.view(1, 1, 1, 2).expand(batch, *current_depth.shape[1:], 2),
        size,
        plane_depth,
        depth_params,
    )
    viewport_length = _length2(size)
    threshold_depth = torch.maximum(current_view, previous_view)
    field_of_view = _length3(corner) / _length3(center)
    required_separation = (
        _DISOCCLUSION_DEPTH_SEPARATION_SCALE
        * field_of_view
        * viewport_length
        * threshold_depth
    )
    reference = current_depth.new_tensor(_DISOCCLUSION_REFERENCE_SIZE)
    resolution_factor = torch.clamp(viewport_length / _length2(reference), 0.0, 1.0)
    power = 1.0 + (3.0 - 1.0) * resolution_factor
    ratio = torch.clamp(required_separation / difference, 0.0, 1.0)
    result = 1.0 - torch.pow(ratio, power)
    return torch.where(difference <= 0.0, torch.zeros_like(result), result)


def _hash_to_float(
    seed_base: torch.Tensor, channel: int, random_seed: int
) -> torch.Tensor:
    """Run the shader's uint32 spatial hash and reinterpret its mantissa bits."""

    mask = 0xFFFFFFFF
    value = (
        seed_base.to(torch.int64)
        + 0x9E3779B9 * int(channel)
        + (int(random_seed) & mask)
    ) & mask
    value = ((value ^ 61) ^ (value >> 16)) & mask
    value = (value * 9) & mask
    value = (value ^ (value >> 4)) & mask
    value = (value * 0x27D4EB2D) & mask
    value = (value ^ (value >> 15)) & mask
    bits = ((value >> 9) | 0x3F800000).to(torch.int32).contiguous()
    return bits.view(torch.float32) - 1.0


def _oob(uv: torch.Tensor) -> torch.Tensor:
    return (
        (uv[..., 0] <= 0.0)
        | (uv[..., 0] >= 1.0)
        | (uv[..., 1] <= 0.0)
        | (uv[..., 1] >= 1.0)
    )


def preprocess_torch(
    warped_flow: torch.Tensor,
    warped_mv: torch.Tensor,
    rgb_m1: torch.Tensor,
    rgb_p1: torch.Tensor,
    depth_m1: torch.Tensor,
    depth_p1: torch.Tensor,
    holes_t: torch.Tensor,
    holes_tm1: torch.Tensor,
    motion_mat_m1p1: torch.Tensor,
    motion_mat_p1m1: torch.Tensor,
    depth_params: torch.Tensor,
    timestep: float,
    random_seed: int,
) -> torch.Tensor:
    """Assemble the detached 16-channel NFRU autoencoder input."""

    validate_nchw(warped_flow, "warped_flow", 2)
    batch = warped_flow.shape[0]
    device = warped_flow.device
    for tensor, name, channels in (
        (warped_mv, "warped_mv", 2),
        (rgb_m1, "rgb_m1", 3),
        (rgb_p1, "rgb_p1", 3),
        (depth_m1, "depth_m1", 1),
        (depth_p1, "depth_p1", 1),
        (holes_t, "holes_t", 1),
        (holes_tm1, "holes_tm1", 1),
    ):
        validate_nchw(tensor, name, channels, device=device, batch=batch)
    if rgb_p1.shape != rgb_m1.shape:
        raise ValueError("rgb_p1 shape must match rgb_m1.")
    if depth_p1.shape != depth_m1.shape:
        raise ValueError("depth_p1 shape must match depth_m1.")
    if holes_t.shape[-2:] != warped_mv.shape[-2:] or holes_tm1.shape != holes_t.shape:
        raise ValueError("hole surfaces must match warped_mv batch and spatial shape.")
    validate_matrix(motion_mat_m1p1, "motion_mat_m1p1", device=device, batch=batch)
    validate_matrix(motion_mat_p1m1, "motion_mat_p1m1", device=device, batch=batch)
    if not isinstance(depth_params, torch.Tensor):
        raise TypeError("depth_params must be a torch.Tensor.")
    if depth_params.shape != (batch, 4, 1, 1):
        raise ValueError(
            f"depth_params must have shape [{batch},4,1,1]; got "
            f"{tuple(depth_params.shape)}."
        )
    if depth_params.dtype != torch.float32 or depth_params.device != device:
        raise TypeError("depth_params must be float32 on the shared tensor device.")
    scale = shader_float(timestep, "timestep")
    if isinstance(random_seed, bool) or not isinstance(random_seed, Integral):
        raise TypeError("random_seed must be an integer.")

    with torch.no_grad():
        flow = warped_flow.detach()
        motion = warped_mv.detach()
        color_m1 = rgb_m1.detach()
        color_p1 = rgb_p1.detach()
        first_depth = depth_m1.detach()
        second_depth = depth_p1.detach()
        first_transform = motion_mat_m1p1.detach()
        second_transform = motion_mat_p1m1.detach()
        params = depth_params.detach()[:, :, 0, 0]
        output_shape = flow.shape[-2:]
        uv = make_uv_grid(output_shape, batch, device)
        motion_pixels = uv_to_pixels(uv, motion.shape[-2:])
        flow_pixels = uv_to_pixels(uv, flow.shape[-2:])
        sampled_flow = gather_pixels(flow, flow_pixels, clamp_to_edge=False).permute(
            0, 2, 3, 1
        )
        sampled_motion = gather_pixels(
            motion, motion_pixels, clamp_to_edge=False
        ).permute(0, 2, 3, 1)

        uv_m1_mv = uv + sampled_motion * scale
        uv_p1_mv = uv - sampled_motion * (1.0 - scale)
        uv_m1_flow = uv - sampled_flow * scale
        uv_p1_flow = uv + sampled_flow * (1.0 - scale)
        oob_m1_mv = _oob(uv_m1_mv)
        oob_p1_mv = _oob(uv_p1_mv)
        oob_m1_flow = _oob(uv_m1_flow)
        oob_p1_flow = _oob(uv_p1_flow)

        hole_m1 = (bilinear_sample(holes_tm1.detach(), uv_m1_mv)[:, 0] != 0.0).to(
            torch.float32
        ) * (~oob_p1_mv).to(torch.float32)
        hole_t = gather_pixels(holes_t.detach(), motion_pixels, clamp_to_edge=False)[
            :, 0
        ] * (~oob_p1_mv).to(torch.float32)
        double_disocclusion = (hole_t - hole_m1 < 0.0).to(torch.float32)

        color_size = color_m1.shape[-2:]

        def load_color(source: torch.Tensor, sample_uv: torch.Tensor) -> torch.Tensor:
            return gather_pixels(
                source, uv_to_pixels(sample_uv, color_size), clamp_to_edge=False
            )

        colors = [
            load_color(color_m1, uv_m1_mv),
            load_color(color_p1, uv_p1_mv),
            load_color(color_m1, uv_m1_flow),
            load_color(color_p1, uv_p1_flow),
        ]
        masks = (oob_m1_mv, oob_p1_mv, oob_m1_flow, oob_p1_flow)
        row = torch.arange(output_shape[0], device=device, dtype=torch.int64)
        column = torch.arange(output_shape[1], device=device, dtype=torch.int64)
        row_grid, column_grid = torch.meshgrid(row, column, indexing="ij")
        seed_base = (row_grid * _RANDOM_OOB_SEED_STRIDE + column_grid).unsqueeze(0)
        seed_base = seed_base.expand(batch, -1, -1)
        for group, mask in enumerate(masks):
            hashed = torch.stack(
                tuple(
                    _hash_to_float(seed_base, group * 3 + channel, int(random_seed))
                    for channel in range(3)
                ),
                dim=1,
            )
            colors[group] = torch.where(mask.unsqueeze(1), hashed, colors[group])

        depth_size = first_depth.shape[-2:]
        depth_m1_warped = gather_pixels(
            first_depth,
            uv_to_pixels(uv_m1_mv, depth_size),
            clamp_to_edge=False,
        )[:, 0]
        depth_p1_warped = gather_pixels(
            second_depth,
            uv_to_pixels(uv_p1_mv, depth_size),
            clamp_to_edge=False,
        )[:, 0]
        normalized_depth = torch.stack(
            (_normalize_depth(depth_m1_warped), _normalize_depth(depth_p1_warped)),
            dim=1,
        )

        depth_m1_transformed = _reproject_depth(
            uv_p1_mv, 1.0 - depth_m1_warped, first_transform
        )
        depth_p1_transformed = _reproject_depth(
            uv_m1_mv, 1.0 - depth_p1_warped, second_transform
        )
        disocclusion_m1 = _single_tap_depth_clip(
            depth_p1_warped, depth_m1_transformed, params, depth_size
        )
        disocclusion_p1 = _single_tap_depth_clip(
            depth_m1_warped, depth_p1_transformed, params, depth_size
        )
        disocclusion = torch.stack((disocclusion_m1, disocclusion_p1), dim=1)
        disocclusion = torch.clamp(
            disocclusion + double_disocclusion.unsqueeze(1), 0.0, 1.0
        )
        result = torch.cat((*colors, normalized_depth, disocclusion), dim=1)
    return result
