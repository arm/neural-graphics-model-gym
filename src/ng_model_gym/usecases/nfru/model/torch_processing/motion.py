# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
"""Dynamic-mask and depth-aware motion stages for native Torch NFRU."""

from __future__ import annotations

import torch

from .quantization import decode_motion, pack_depth_motion, quantize_dynamic_motion
from .sampling import (
    gather_pixels,
    make_pixel_grid,
    make_uv_grid,
    ordered_nearest_depth,
    ordered_packed_max,
    shader_float,
    uv_to_pixels,
    validate_matrix,
    validate_nchw,
)

_DEPTH_LOG_K = 1200.0
_DEPTH_MAX = 1.1
_FAR_PLANE_COLLISION_BIAS = 0.1
_ZERO_MOTION_EPS = 1.0e-5


def _length2(vector: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(vector[..., 0] * vector[..., 0] + vector[..., 1] * vector[..., 1])


def _calculate_camera_motion(
    uv: torch.Tensor,
    depth: torch.Tensor,
    transform: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate ``CalculateCameraMotion`` including its cleanup quirks."""

    uv_transformed = torch.stack((uv[..., 1], 1.0 - uv[..., 0]), dim=-1)
    clip = torch.cat(
        (
            2.0 * uv_transformed - 1.0,
            depth.unsqueeze(-1),
            torch.ones_like(depth).unsqueeze(-1),
        ),
        dim=-1,
    )
    reprojected = torch.einsum("nij,nhwj->nhwi", transform, clip)
    invalid = reprojected[..., 3] < 0.0
    previous_uv = (reprojected[..., :2] / reprojected[..., 3:4] + 1.0) * 0.5
    velocity = uv_transformed - previous_uv
    basically_zero = torch.all(torch.abs(velocity) < _ZERO_MOTION_EPS, dim=-1)
    nonfinite = torch.any(torch.isnan(velocity), dim=-1) | torch.any(
        torch.isinf(velocity), dim=-1
    )
    velocity = torch.where(
        (basically_zero | nonfinite).unsqueeze(-1),
        torch.zeros_like(velocity),
        velocity,
    )
    motion = torch.stack((velocity[..., 1], -velocity[..., 0]), dim=-1)
    return motion, invalid.to(torch.float32)


def _dynamic_mask(
    camera_motion: torch.Tensor,
    rendered_motion: torch.Tensor,
    epsilon: float,
    tau: float,
    *,
    runtime_accurate: bool,
    spatial_shape: tuple[int, int],
) -> torch.Tensor:
    if runtime_accurate:
        size = camera_motion.new_tensor(spatial_shape)
        camera_motion = quantize_dynamic_motion(camera_motion * size)
        rendered_motion = quantize_dynamic_motion(rendered_motion * size)
        difference = _length2(camera_motion - rendered_motion)
        max_length = torch.maximum(
            torch.maximum(
                _length2(camera_motion),
                _length2(rendered_motion),
            ),
            difference.new_tensor(tau),
        )
        # Keep the shader's unreachable branch rather than simplifying it.
        return torch.where(
            max_length < tau,
            torch.zeros_like(difference),
            (difference >= epsilon).to(torch.float32),
        )

    difference = _length2(camera_motion - rendered_motion)
    denominator = torch.maximum(
        torch.maximum(
            _length2(camera_motion),
            _length2(rendered_motion),
        ),
        difference.new_tensor(tau),
    )
    return (difference / denominator >= epsilon).to(torch.float32)


def previous_dynamic_mask_torch(
    depth: torch.Tensor,
    rendered_mv: torch.Tensor,
    previous_transform: torch.Tensor,
    mv_similarity_threshold: float,
    mv_similarity_noise_threshold: float,
    runtime_accurate: bool = False,
) -> torch.Tensor:
    """Calculate the previous-frame dynamic mask without adding invalid-W bits."""

    if not isinstance(runtime_accurate, bool):
        raise TypeError("runtime_accurate must be a bool.")
    validate_nchw(depth, "depth", 1)
    validate_nchw(
        rendered_mv,
        "rendered_mv",
        2,
        device=depth.device,
        batch=depth.shape[0],
    )
    if rendered_mv.shape[-2:] != depth.shape[-2:]:
        raise ValueError("rendered_mv spatial dimensions must match depth.")
    validate_matrix(
        previous_transform,
        "previous_transform",
        device=depth.device,
        batch=depth.shape[0],
    )
    epsilon = shader_float(mv_similarity_threshold, "mv_similarity_threshold")
    tau = shader_float(mv_similarity_noise_threshold, "mv_similarity_noise_threshold")
    with torch.no_grad():
        uv = make_uv_grid(depth.shape[-2:], depth.shape[0], depth.device)
        camera_motion, _ = _calculate_camera_motion(
            uv, 1.0 - depth.detach()[:, 0], previous_transform.detach()
        )
        result = _dynamic_mask(
            camera_motion,
            rendered_mv.detach().permute(0, 2, 3, 1),
            epsilon,
            tau,
            runtime_accurate=runtime_accurate,
            spatial_shape=depth.shape[-2:],
        )
    return result.unsqueeze(1)


def _normalize_and_invert_depth(depth: torch.Tensor) -> torch.Tensor:
    """Apply log normalization, source-index far bias, then depth inversion."""

    height, width = depth.shape[-2:]
    normalized = 1.0 - torch.log(-(_DEPTH_LOG_K * (depth - 1.0)) + 1.0) / torch.log(
        depth.new_tensor(_DEPTH_LOG_K + 1.0)
    )
    pixels = make_pixel_grid((height, width), depth.shape[0], depth.device)
    # The shader deliberately uses row + column * height here.
    source_index = pixels[..., 0] + pixels[..., 1] * height
    bias = (
        source_index.to(depth.dtype) / float(height * width) * _FAR_PLANE_COLLISION_BIAS
    )
    normalized = torch.where(normalized == 1.0, normalized + bias, normalized)
    return _DEPTH_MAX - normalized


def _scatter_motion(
    packed: torch.Tensor,
    pixels: torch.Tensor,
    motion: torch.Tensor,
    depth: torch.Tensor,
    timestep: float,
    sign: float,
) -> torch.Tensor:
    """Depth-aware forward scatter using maximum complete packed codes."""

    _, _, height, width = packed.shape
    size = motion.new_tensor((height, width))
    vector = motion * size
    destination = pixels + torch.floor(vector * timestep).to(torch.int64)
    valid = (
        (destination[..., 0] >= 0)
        & (destination[..., 0] < height)
        & (destination[..., 1] >= 0)
        & (destination[..., 1] < width)
    )
    code = pack_depth_motion(sign * vector, depth)
    destination_index = destination[..., 0] * width + destination[..., 1]
    destination_index = torch.where(
        valid, destination_index, torch.zeros_like(destination_index)
    ).flatten(1)
    source = torch.where(valid, code, torch.zeros_like(code)).flatten(1)
    result = packed.flatten(1).scatter_reduce(
        1,
        destination_index,
        source,
        reduce="amax",
        include_self=True,
    )
    return result.reshape_as(packed)


def _fill_motion(packed: torch.Tensor) -> torch.Tensor:
    nearest = ordered_packed_max(packed)
    decoded = decode_motion(nearest[:, 0])
    inverse_size = decoded.new_tensor(packed.shape[-2:]).reciprocal()
    return (decoded * inverse_size).permute(0, 3, 1, 2)


def _warp_mv_packed_torch(
    current_depth: torch.Tensor,
    previous_depth: torch.Tensor,
    rendered_mv: torch.Tensor,
    previous_mask: torch.Tensor,
    previous_to_current: torch.Tensor,
    current_to_previous: torch.Tensor,
    timestep: float,
    mv_similarity_threshold: float,
    mv_similarity_noise_threshold: float,
    runtime_accurate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw MV scatter plus next mask and the two hole-marker surfaces."""

    if not isinstance(runtime_accurate, bool):
        raise TypeError("runtime_accurate must be a bool.")
    validate_nchw(current_depth, "current_depth", 1)
    batch = current_depth.shape[0]
    device = current_depth.device
    validate_nchw(previous_depth, "previous_depth", 1, device=device, batch=batch)
    validate_nchw(rendered_mv, "rendered_mv", 2, device=device, batch=batch)
    validate_nchw(previous_mask, "previous_mask", 1, device=device, batch=batch)
    if previous_depth.shape[-2:] != current_depth.shape[-2:]:
        raise ValueError("previous_depth spatial dimensions must match current_depth.")
    if previous_mask.shape[-2:] != current_depth.shape[-2:]:
        raise ValueError("previous_mask spatial dimensions must match current_depth.")
    validate_matrix(
        previous_to_current,
        "previous_to_current",
        device=device,
        batch=batch,
    )
    validate_matrix(
        current_to_previous,
        "current_to_previous",
        device=device,
        batch=batch,
    )
    scale = shader_float(timestep, "timestep")
    epsilon = shader_float(mv_similarity_threshold, "mv_similarity_threshold")
    tau = shader_float(mv_similarity_noise_threshold, "mv_similarity_noise_threshold")

    with torch.no_grad():
        spatial_shape = current_depth.shape[-2:]
        height, width = spatial_shape
        pixels = make_pixel_grid(spatial_shape, batch, device)
        uv = make_uv_grid(spatial_shape, batch, device)
        depth_pixels = uv_to_pixels(uv, current_depth.shape[-2:])
        motion_pixels = uv_to_pixels(uv, rendered_mv.shape[-2:])
        depth_current = gather_pixels(
            current_depth.detach(), depth_pixels, clamp_to_edge=False
        )[:, 0]
        depth_previous = gather_pixels(
            previous_depth.detach(), depth_pixels, clamp_to_edge=False
        )[:, 0]
        motion = gather_pixels(
            rendered_mv.detach(), motion_pixels, clamp_to_edge=False
        ).permute(0, 2, 3, 1)

        camera_current, invalid_current = _calculate_camera_motion(
            uv, 1.0 - depth_current, current_to_previous.detach()
        )
        camera_previous, invalid_previous = _calculate_camera_motion(
            uv, 1.0 - depth_previous, previous_to_current.detach()
        )
        packed = torch.zeros(batch, 1, height, width, dtype=torch.int32, device=device)
        packed = _scatter_motion(
            packed,
            pixels,
            motion,
            _normalize_and_invert_depth(depth_current),
            1.0 - scale,
            1.0,
        )

        _, nearest_offset = ordered_nearest_depth(current_depth.detach(), pixels)
        nearest_motion = gather_pixels(
            rendered_mv.detach(), motion_pixels + nearest_offset, clamp_to_edge=False
        ).permute(0, 2, 3, 1)
        size = nearest_motion.new_tensor((height, width))
        holes_t_coord = pixels + torch.floor(nearest_motion * size * (1.0 - scale)).to(
            torch.int64
        )
        holes_previous_coord = pixels + torch.floor(nearest_motion * size).to(
            torch.int64
        )

        def marker_surface(coordinates: torch.Tensor) -> torch.Tensor:
            valid = (
                (coordinates[..., 0] >= 0)
                & (coordinates[..., 0] < height)
                & (coordinates[..., 1] >= 0)
                & (coordinates[..., 1] < width)
            )
            destination = coordinates[..., 0] * width + coordinates[..., 1]
            destination = torch.where(
                valid, destination, torch.zeros_like(destination)
            ).flatten(1)
            values = valid.to(torch.float32).flatten(1)
            result = torch.zeros(batch, height * width, device=device)
            result.scatter_reduce_(
                1, destination, values, reduce="amax", include_self=True
            )
            return result.reshape(batch, 1, height, width)

        holes_t = marker_surface(holes_t_coord)
        holes_previous = marker_surface(holes_previous_coord)
        dynamic = _dynamic_mask(
            camera_current,
            motion,
            epsilon,
            tau,
            runtime_accurate=runtime_accurate,
            spatial_shape=spatial_shape,
        )
        next_mask = (dynamic + invalid_current).unsqueeze(1)

        feedback = previous_mask.detach()[:, 0]
        may_scatter = feedback + invalid_previous <= 0.0
        # Scatter with invalid source codes zeroed so masked sources cannot
        # affect destinations reached by valid sources.
        vector = camera_previous * camera_previous.new_tensor(spatial_shape)
        destination = pixels + torch.floor(vector * scale).to(torch.int64)
        onscreen = (
            may_scatter
            & (destination[..., 0] >= 0)
            & (destination[..., 0] < height)
            & (destination[..., 1] >= 0)
            & (destination[..., 1] < width)
        )
        code = pack_depth_motion(-vector, _normalize_and_invert_depth(depth_previous))
        index = destination[..., 0] * width + destination[..., 1]
        index = torch.where(onscreen, index, torch.zeros_like(index)).flatten(1)
        source = torch.where(onscreen, code, torch.zeros_like(code)).flatten(1)
        packed = (
            packed.flatten(1)
            .scatter_reduce(1, index, source, reduce="amax", include_self=True)
            .reshape_as(packed)
        )
    return packed, next_mask, holes_t, holes_previous


def warp_mv_torch(
    current_depth: torch.Tensor,
    previous_depth: torch.Tensor,
    rendered_mv: torch.Tensor,
    previous_mask: torch.Tensor,
    previous_to_current: torch.Tensor,
    current_to_previous: torch.Tensor,
    timestep: float,
    mv_similarity_threshold: float,
    mv_similarity_noise_threshold: float,
    runtime_accurate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Warp and fill motion, returning the public Slang-stage order."""

    packed, next_mask, holes_t, holes_previous = _warp_mv_packed_torch(
        current_depth,
        previous_depth,
        rendered_mv,
        previous_mask,
        previous_to_current,
        current_to_previous,
        timestep,
        mv_similarity_threshold,
        mv_similarity_noise_threshold,
        runtime_accurate,
    )
    return _fill_motion(packed), next_mask, holes_t, holes_previous


def _warp_flow_packed_torch(
    depth: torch.Tensor, flow: torch.Tensor, timestep: float
) -> torch.Tensor:
    """Return the raw packed optical-flow scatter surface."""

    validate_nchw(depth, "depth", 1)
    validate_nchw(flow, "flow", 2, device=depth.device, batch=depth.shape[0])
    scale = shader_float(timestep, "timestep")
    batch = depth.shape[0]
    spatial_shape = flow.shape[-2:]
    with torch.no_grad():
        pixels = make_pixel_grid(spatial_shape, batch, depth.device)
        uv = make_uv_grid(spatial_shape, batch, depth.device)
        sampled_depth = gather_pixels(
            depth.detach(), uv_to_pixels(uv, depth.shape[-2:]), clamp_to_edge=False
        )[:, 0]
        sampled_flow = gather_pixels(
            flow.detach(), uv_to_pixels(uv, flow.shape[-2:]), clamp_to_edge=False
        ).permute(0, 2, 3, 1)
        packed = torch.zeros(
            batch, 1, *spatial_shape, device=depth.device, dtype=torch.int32
        )
        return _scatter_motion(
            packed,
            pixels,
            sampled_flow,
            _normalize_and_invert_depth(sampled_depth),
            scale,
            1.0,
        )


def warp_flow_torch(
    depth: torch.Tensor, flow: torch.Tensor, timestep: float
) -> torch.Tensor:
    """Warp and ordered-hole-fill optical flow."""

    return _fill_motion(_warp_flow_packed_torch(depth, flow, timestep))
