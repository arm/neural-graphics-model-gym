# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
"""Depth reconstruction and disocclusion for the Torch NSS-v1 preprocessor."""

import torch

from .sampling import gather_pixels, make_pixel_grid

_INT32_MAX = torch.iinfo(torch.int32).max
_INT32_MIN = torch.iinfo(torch.int32).min
_RECONSTRUCTION_OFFSETS = ((0, 0), (1, 0), (0, 1), (1, 1))
_DEPTH_SCATTER_2X2_OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1))
_DEPTH_SCATTER_4X4_OFFSETS = (
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 0),
    (2, 1),
    (3, 0),
    (3, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
)


def _spatial_shape(output_shape: tuple[int, ...]) -> tuple[int, int]:
    if len(output_shape) == 2:
        return output_shape
    if len(output_shape) == 4:
        return output_shape[-2], output_shape[-1]
    raise ValueError("Output shape must be spatial (H, W) or NCHW.")


def _length2(vector: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(vector[..., 0] * vector[..., 0] + vector[..., 1] * vector[..., 1])


def _length3(vector: torch.Tensor) -> torch.Tensor:
    squared_xy = vector[..., 0] * vector[..., 0] + vector[..., 1] * vector[..., 1]
    return torch.sqrt(squared_xy + vector[..., 2] * vector[..., 2])


def _bilinear_data(
    uv: torch.Tensor,
    spatial_shape: tuple[int, int],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Translate NSS ``GetBilinearSamplingData`` in row/column order."""

    uv = uv.detach()
    size = uv.new_tensor(spatial_shape)
    pixel_sample = uv * size - 0.5
    base = torch.floor(pixel_sample).to(torch.int64)
    fraction = pixel_sample - torch.floor(pixel_sample)
    one_minus_row = 1.0 - fraction[..., 0]
    one_minus_column = 1.0 - fraction[..., 1]
    weights = (
        one_minus_row * one_minus_column,
        fraction[..., 0] * one_minus_column,
        one_minus_row * fraction[..., 1],
        fraction[..., 0] * fraction[..., 1],
    )
    return base, weights


def saturating_float_to_int32(value: torch.Tensor) -> torch.Tensor:
    """Truncate float values to int32 with explicit saturation.

    CUDA shader casts saturate at the positive int32 endpoint.  CPU's direct
    cast can instead wrap the float32 representation of ``INT32_MAX`` (which is
    ``2147483648``) to ``INT32_MIN``.
    """

    value64 = value.detach().to(torch.float64)
    value64 = torch.clamp(value64, float(_INT32_MIN), float(_INT32_MAX))
    return torch.trunc(value64).to(torch.int32)


def _reconstruct_previous_depth(
    depth: torch.Tensor,
    reprojected_uv: torch.Tensor,
    output_shape: tuple[int, int],
    *,
    bilinear_weight_threshold: float = 0.1,
) -> torch.Tensor:
    """Scatter minimum reconstructed depth to four ordered bilinear targets."""

    batch_size = depth.shape[0]
    output_height, output_width = output_shape
    base, weights = _bilinear_data(reprojected_uv, output_shape)
    # Match float(depth * float(0x7fffffff)) before the saturating cast.  The
    # integer constant rounds to 2**31 when represented as float32.
    int_depth = saturating_float_to_int32(
        depth.detach() * depth.new_tensor(float(_INT32_MAX))
    )
    output = torch.full(
        (batch_size * output_height * output_width,),
        _INT32_MAX,
        dtype=torch.int32,
        device=depth.device,
    )
    batch_offset = (
        torch.arange(batch_size, device=depth.device, dtype=torch.int64)
        .view(batch_size, 1, 1)
        .mul(output_height * output_width)
    )

    for (row_offset, column_offset), weight in zip(_RECONSTRUCTION_OFFSETS, weights):
        position = base + base.new_tensor((row_offset, column_offset))
        onscreen = (
            (position[..., 0] >= 0)
            & (position[..., 0] < output_height)
            & (position[..., 1] >= 0)
            & (position[..., 1] < output_width)
        )
        active = onscreen & (weight > bilinear_weight_threshold)
        flat_index = (
            batch_offset
            + position[..., 0].clamp(0, output_height - 1) * output_width
            + position[..., 1].clamp(0, output_width - 1)
        )
        output.scatter_reduce_(
            0,
            flat_index[active],
            int_depth[active],
            reduce="amin",
            include_self=True,
        )

    return output.reshape(batch_size, 1, output_height, output_width)


def depth_scatter(
    motion: torch.Tensor,
    depth: torch.Tensor,
    render_size: torch.Tensor,
    output_shape: tuple[int, ...],
    *,
    quarter_res_input: bool,
) -> torch.Tensor:
    """Translate ``depth_scatter.slang`` on CPU or CUDA.

    The candidate loop is vectorized over batches and pixels but kept
    sequential over its 4 or 16 literal candidates.  This both preserves equal
    depth replacement and avoids an ``N x 16 x H x W`` expansion.
    """

    del render_size  # Kept in the API because the shader binding requires it.
    if motion.ndim != 4 or motion.shape[1] != 2:
        raise ValueError("Motion must have shape (N, 2, H, W).")
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("Depth must have shape (N, 1, H, W).")
    if motion.shape[0] != depth.shape[0] or motion.shape[-2:] != depth.shape[-2:]:
        raise ValueError("Motion and depth batch/spatial dimensions must match.")

    motion = motion.detach()
    depth = depth.detach()
    batch_size, _, input_height, input_width = depth.shape
    output_height, output_width = _spatial_shape(output_shape)
    output_pixels = make_pixel_grid(
        (output_height, output_width),
        batch_size=batch_size,
        device=depth.device,
    )
    scale = depth.new_tensor((input_height, input_width)) / depth.new_tensor(
        (output_height, output_width)
    )
    if quarter_res_input:
        source_base = (output_pixels.to(depth.dtype) * scale).to(torch.int64)
        inverse_output_size = torch.reciprocal(
            depth.new_tensor((output_height, output_width))
        )
        inverse_source_size = inverse_output_size * torch.reciprocal(scale)

        def candidate_load_position(
            row_offset: int, column_offset: int
        ) -> torch.Tensor:
            # Reproduce the four texture-gather UV round trips.  At odd sizes
            # their float32 bases can differ from the integer anchors.
            anchor = source_base.new_tensor(
                ((row_offset // 2) * 2, (column_offset // 2) * 2)
            )
            local = source_base.new_tensor((row_offset % 2, column_offset % 2))
            gather_uv = (
                source_base.to(depth.dtype) + anchor.to(depth.dtype) + 0.5
            ) * inverse_source_size
            gather_base = torch.floor(
                gather_uv * depth.new_tensor((input_height, input_width)) - 0.5
            ).to(torch.int64)
            return gather_base + local

    else:
        # The high path obtains its source base through texture_gather.  Keep
        # the shader's reciprocal/multiply order: for odd or otherwise
        # non-integral ratios this is not generally floor(output * scale).
        inverse_output_size = torch.reciprocal(
            depth.new_tensor((output_height, output_width))
        )
        gather_uv = (output_pixels.to(depth.dtype) + 0.5) * inverse_output_size
        source_base = torch.floor(
            gather_uv * depth.new_tensor((input_height, input_width)) - 0.5
        ).to(torch.int64)

        def candidate_load_position(
            row_offset: int, column_offset: int
        ) -> torch.Tensor:
            return source_base + source_base.new_tensor((row_offset, column_offset))

    candidate_offsets = (
        _DEPTH_SCATTER_4X4_OFFSETS if quarter_res_input else _DEPTH_SCATTER_2X2_OFFSETS
    )

    initial_position = candidate_load_position(0, 0)
    nearest_depth = gather_pixels(depth, initial_position)[:, 0]
    nearest_motion = gather_pixels(motion, initial_position).permute(0, 2, 3, 1)
    for row_offset, column_offset in candidate_offsets[1:]:
        logical_position = source_base + source_base.new_tensor(
            (row_offset, column_offset)
        )
        position = candidate_load_position(row_offset, column_offset)
        candidate_depth = gather_pixels(depth, position)[:, 0]
        candidate_motion = gather_pixels(motion, position).permute(0, 2, 3, 1)
        if quarter_res_input:
            onscreen = (
                (logical_position[..., 0] >= 0)
                & (logical_position[..., 0] < input_height)
                & (logical_position[..., 1] >= 0)
                & (logical_position[..., 1] < input_width)
            )
        else:
            # texture_gather clamps its four source loads in the high path.
            onscreen = torch.ones_like(candidate_depth, dtype=torch.bool)
        take = onscreen & (candidate_depth <= nearest_depth)
        # depth_scatter.slang uses lerp rather than assignment.  Slang lowers
        # this to x + take * (y - x), which can move a selected float by one
        # ULP through cancellation even when take is exactly one.
        take_float = take.to(depth.dtype)
        nearest_depth = nearest_depth + take_float * (candidate_depth - nearest_depth)
        nearest_motion = nearest_motion + take_float.unsqueeze(-1) * (
            candidate_motion - nearest_motion
        )

    inverse_scale = torch.reciprocal(scale)
    scaled_motion = nearest_motion * inverse_scale
    threshold_motion = nearest_motion if quarter_res_input else scaled_motion
    threshold = (_length2(threshold_motion) > 0.1).to(depth.dtype)
    scaled_motion = scaled_motion * threshold.unsqueeze(-1)
    inverse_output_size = torch.reciprocal(
        depth.new_tensor((output_height, output_width))
    )
    uv = (output_pixels.to(depth.dtype) + 0.5) * inverse_output_size
    reprojected_uv = uv - scaled_motion * inverse_output_size
    return _reconstruct_previous_depth(
        nearest_depth,
        reprojected_uv,
        (output_height, output_width),
    )


def _view_space_depth(
    depth: torch.Tensor,
    device_to_view: torch.Tensor,
) -> torch.Tensor:
    return device_to_view[..., 1] / (depth - device_to_view[..., 0])


def _view_space_position(
    viewport_position: torch.Tensor,
    viewport_size: torch.Tensor,
    device_depth: torch.Tensor,
    device_to_view: torch.Tensor,
) -> torch.Tensor:
    view_depth = _view_space_depth(device_depth, device_to_view)
    scale = viewport_position / viewport_size
    ndc_row = scale[..., 0] * 2.0 - 1.0
    ndc_column = scale[..., 1] * -2.0 + 1.0
    view_row = device_to_view[..., 2] * ndc_row * view_depth
    view_column = device_to_view[..., 3] * ndc_column * view_depth
    return torch.stack((view_row, view_column, view_depth), dim=-1)


def compute_depth_clip(
    depth_tm1: torch.Tensor,
    uv: torch.Tensor,
    render_size: torch.Tensor,
    current_depth: torch.Tensor,
    depth_params: torch.Tensor,
    *,
    inverted: bool = False,
    local_view_depth_range: torch.Tensor | None = None,
    depth_gradient_tolerance_scale: float = 1.0,
    bilinear_weight_threshold: float = 0.1,
) -> torch.Tensor:
    """Translate high and low-quality ``ComputeDepthClip`` variants."""

    depth_tm1 = depth_tm1.detach()
    uv = uv.detach().to(device=depth_tm1.device, dtype=torch.float32)
    current_depth = current_depth.detach().to(dtype=torch.float32)
    batch_size, _, height, width = depth_tm1.shape
    if uv.shape[0] not in (1, batch_size):
        raise ValueError("UV batch must be one or match depth batch.")
    uv = uv.expand(batch_size, -1, -1, -1)
    output_height, output_width = uv.shape[1:3]

    device_to_view = depth_params.detach().reshape(batch_size, 4).to(torch.float32)
    device_to_view = device_to_view[:, None, None, :]
    render_size = render_size.detach().reshape(batch_size, 2).to(torch.float32)
    render_size = render_size[:, None, None, :]
    current_view_depth = _view_space_depth(current_depth, device_to_view)
    base, weights = _bilinear_data(uv, (height, width))
    accumulated_depth = torch.zeros_like(current_depth)
    weight_sum = torch.zeros_like(current_depth)
    if local_view_depth_range is None:
        local_tolerance: torch.Tensor | float = 0.0
    else:
        local_tolerance = (
            local_view_depth_range.detach().to(torch.float32)
            * depth_gradient_tolerance_scale
        )

    for (row_offset, column_offset), weight in zip(_RECONSTRUCTION_OFFSETS, weights):
        position = base + base.new_tensor((row_offset, column_offset))
        onscreen = (
            (position[..., 0] >= 0)
            & (position[..., 0] < height)
            & (position[..., 1] >= 0)
            & (position[..., 1] < width)
        )
        weight_sum = weight_sum + torch.where(
            onscreen, torch.zeros_like(weight), weight
        )
        active_weight = onscreen & (weight > bilinear_weight_threshold)
        previous_depth = gather_pixels(depth_tm1, position)[:, 0].to(torch.float32)
        previous_depth = previous_depth * torch.reciprocal(
            previous_depth.new_tensor(float(_INT32_MAX))
        )
        previous_view_depth = _view_space_depth(previous_depth, device_to_view)
        depth_difference = current_view_depth - previous_view_depth
        active = active_weight & (depth_difference > 0.0)

        if inverted:
            plane_depth = torch.minimum(previous_depth, current_depth)
        else:
            plane_depth = torch.maximum(previous_depth, current_depth)
        viewport_size = render_size.to(torch.int64).to(torch.float32)
        center_position = (render_size * 0.5).to(torch.int64).to(torch.float32)
        corner_position = torch.zeros_like(center_position)
        center = _view_space_position(
            center_position,
            viewport_size,
            plane_depth,
            device_to_view,
        )
        corner = _view_space_position(
            corner_position,
            viewport_size,
            plane_depth,
            device_to_view,
        )
        half_viewport_width = _length2(render_size)
        depth_threshold = torch.maximum(current_view_depth, previous_view_depth)
        required_separation = (
            1.37e-05
            * (_length3(corner) / _length3(center))
            * half_viewport_width
            * depth_threshold
        ) + local_tolerance
        reference_size = render_size.new_tensor((1080.0, 1920.0))
        resolution_factor = torch.clamp(
            _length2(render_size) / _length2(reference_size), 0.0, 1.0
        )
        power = 1.0 + (3.0 - 1.0) * resolution_factor
        ratio = torch.clamp(required_separation / depth_difference, 0.0, 1.0)
        depth_contribution = torch.pow(ratio, power) * weight
        accumulated_depth = accumulated_depth + torch.where(
            active, depth_contribution, torch.zeros_like(depth_contribution)
        )
        weight_sum = weight_sum + torch.where(active, weight, torch.zeros_like(weight))

    clipped = torch.where(
        weight_sum > 0.0,
        torch.clamp(1.0 - accumulated_depth / weight_sum, 0.0, 1.0),
        torch.zeros_like(weight_sum),
    )
    return clipped.reshape(batch_size, output_height, output_width).detach()


def lq_disocclusion_mask(
    motion: torch.Tensor,
    depth: torch.Tensor,
    depth_tm1: torch.Tensor,
    depth_params: torch.Tensor,
    render_size: torch.Tensor,
    output_shape: tuple[int, ...],
    *,
    depth_gradient_tolerance_scale: float = 1.0,
) -> torch.Tensor:
    """Translate ``disocclusion_lq.slang`` for low and mid quality."""

    motion = motion.detach()
    depth = depth.detach()
    batch_size, _, input_height, input_width = depth.shape
    output_height, output_width = _spatial_shape(output_shape)
    destination = make_pixel_grid(
        (output_height, output_width),
        batch_size=batch_size,
        device=depth.device,
    )
    scale = depth.new_tensor((input_height, input_width)) / depth.new_tensor(
        (output_height, output_width)
    )
    source_base = (destination.to(depth.dtype) * scale).to(torch.int64)
    source_position = torch.stack(
        (
            source_base[..., 0].clamp(0, input_height - 1),
            source_base[..., 1].clamp(0, input_width - 1),
        ),
        dim=-1,
    )
    closest_depth = gather_pixels(depth, source_position)[:, 0]
    closest_motion = gather_pixels(motion, source_position).permute(0, 2, 3, 1)
    device_to_view = depth_params.detach().reshape(batch_size, 4).to(torch.float32)
    device_to_view = device_to_view[:, None, None, :]
    initial_view_depth = _view_space_depth(closest_depth, device_to_view)
    minimum_view_depth = initial_view_depth
    maximum_view_depth = initial_view_depth

    # The shader's loop is y-major but constructs int2(x, y).  Since NSS int2
    # is (row, column), column is the outer loop here.
    for column_offset in range(4):
        for row_offset in range(4):
            position = source_base + source_base.new_tensor((row_offset, column_offset))
            onscreen = (
                (position[..., 0] >= 0)
                & (position[..., 0] < input_height)
                & (position[..., 1] >= 0)
                & (position[..., 1] < input_width)
            )
            sample_depth = gather_pixels(depth, position)[:, 0]
            sample_motion = gather_pixels(motion, position).permute(0, 2, 3, 1)
            sample_view_depth = _view_space_depth(sample_depth, device_to_view)
            minimum_view_depth = torch.where(
                onscreen,
                torch.minimum(minimum_view_depth, sample_view_depth),
                minimum_view_depth,
            )
            maximum_view_depth = torch.where(
                onscreen,
                torch.maximum(maximum_view_depth, sample_view_depth),
                maximum_view_depth,
            )
            take = onscreen & (sample_depth < closest_depth)
            closest_depth = torch.where(take, sample_depth, closest_depth)
            closest_motion = torch.where(
                take.unsqueeze(-1), sample_motion, closest_motion
            )

    local_view_depth_range = maximum_view_depth - minimum_view_depth
    motion_scale = depth.new_tensor(
        (output_height / input_height, output_width / input_width)
    )
    motion_depth_pixels = closest_motion * motion_scale
    motion_depth_pixels = motion_depth_pixels * (_length2(closest_motion) > 0.1).to(
        depth.dtype
    ).unsqueeze(-1)
    inverse_output_size = torch.reciprocal(
        depth.new_tensor((output_height, output_width))
    )
    uv = (destination.to(depth.dtype) + 0.5) * inverse_output_size
    reprojected_uv = uv - motion_depth_pixels * inverse_output_size
    clipped = compute_depth_clip(
        depth_tm1,
        reprojected_uv,
        render_size,
        closest_depth,
        depth_params,
        local_view_depth_range=local_view_depth_range,
        depth_gradient_tolerance_scale=depth_gradient_tolerance_scale,
    )
    return clipped.unsqueeze(1)
