# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
"""Sampling and coordinate primitives for native NSS v1 postprocessing."""

from typing import NamedTuple

import torch

EPS = 1.0e-7
MAX_HALF = 65504.0


class TemporalParams(NamedTuple):
    """Upsampled and transformed temporal parameters."""

    theta: torch.Tensor
    alpha: torch.Tensor
    gamma: torch.Tensor


def expand_batch(tensor: torch.Tensor, batch: int) -> torch.Tensor:
    """Expand shared metadata without materializing repeated tensor data."""

    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch, *tensor.shape[1:])
    raise ValueError(f"Expected batch size 1 or {batch}, got {tensor.shape[0]}.")


def output_coordinates(
    batch: int,
    output_size_yx: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Return output pixel coordinates in row/y then column/x order."""

    rows = torch.arange(output_size_yx[0], device=device)
    columns = torch.arange(output_size_yx[1], device=device)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    return (
        torch.stack((row_grid, column_grid), dim=-1)
        .unsqueeze(0)
        .expand(batch, -1, -1, -1)
    )


def fused_multiply_add(
    multiplier_a: torch.Tensor,
    multiplier_b: torch.Tensor,
    addend: torch.Tensor,
) -> torch.Tensor:
    """Emulate one float32 CUDA fused multiply-add rounding."""

    return (
        multiplier_a.to(torch.float64) * multiplier_b.to(torch.float64)
        + addend.to(torch.float64)
    ).to(multiplier_a.dtype)


def decode_nearest_offsets(
    nearest_depth_offset: torch.Tensor,
    input_coordinates_yx: torch.Tensor,
    input_size_yx: tuple[int, int],
    *,
    preprocess_half_res_input: bool,
    packed_nearest_offset_quad: bool,
) -> torch.Tensor:
    """Decode nearest offsets as signed row/y then column/x pairs."""

    encoded = nearest_depth_offset.detach()
    coordinates = input_coordinates_yx.detach().to(torch.long)
    batch = coordinates.shape[0]
    storage_height, storage_width = encoded.shape[-2:]

    if preprocess_half_res_input:
        preprocess_size = encoded.new_tensor(
            (input_size_yx[0] // 2, input_size_yx[1] // 2)
        )
        input_size = encoded.new_tensor(input_size_yx)
        preprocess_scale = preprocess_size * torch.reciprocal(input_size)
        texel = torch.floor(
            (coordinates.to(encoded.dtype) + 0.5) * preprocess_scale
        ).to(torch.long)
        logical_height = input_size_yx[0] // 2
        logical_width = input_size_yx[1] // 2
    else:
        texel = coordinates
        logical_height = storage_height
        logical_width = storage_width

    texel_y = texel[..., 0].clamp(0, logical_height - 1)
    texel_x = texel[..., 1].clamp(0, logical_width - 1)
    flat_texel = (texel_y * storage_width + texel_x).reshape(batch, 1, -1)
    gathered = torch.gather(
        encoded.reshape(batch, encoded.shape[1], -1),
        2,
        flat_texel.expand(-1, encoded.shape[1], -1),
    ).reshape(batch, encoded.shape[1], *coordinates.shape[1:-1])
    # Slang int(...) truncates these nonnegative normalized byte encodings.
    codes = (gathered * 255.0 + 0.5).to(torch.long)

    if packed_nearest_offset_quad:
        lanes = (coordinates[..., 0] & 1) * 2 + (coordinates[..., 1] & 1)
        byte_channel = (lanes >= 2).to(torch.long).unsqueeze(1)
        packed_byte = torch.gather(codes, 1, byte_channel).squeeze(1)
        code = torch.where(
            (lanes & 1) == 0,
            packed_byte & 0xF,
            (packed_byte >> 4) & 0xF,
        )
        row_offset = (code & 0x3) - 1
        column_offset = ((code >> 2) & 0x3) - 1
    else:
        code = codes[:, 0]
        row_offset = (code & 0x7) - 2
        column_offset = ((code >> 3) & 0x7) - 2
    return torch.stack((row_offset, column_offset), dim=-1)


def load_motion(
    motion_yx: torch.Tensor,
    nearest_depth_offset: torch.Tensor,
    output_size_yx: tuple[int, int],
    *,
    preprocess_half_res_input: bool,
    packed_nearest_offset_quad: bool,
) -> torch.Tensor:
    """Gather, scale, and threshold motion exactly as ``LoadMotion`` does."""

    motion = motion_yx.detach()
    batch, _, input_height, input_width = motion.shape
    coordinates = output_coordinates(batch, output_size_yx, motion.device)
    input_size = motion.new_tensor((input_height, input_width))
    output_size = motion.new_tensor(output_size_yx)
    scale = output_size / input_size
    inverse_scale = torch.reciprocal(scale)
    input_coordinates = torch.floor(coordinates.to(motion.dtype) * inverse_scale).to(
        torch.long
    )
    offsets = decode_nearest_offsets(
        nearest_depth_offset,
        input_coordinates,
        (input_height, input_width),
        preprocess_half_res_input=preprocess_half_res_input,
        packed_nearest_offset_quad=packed_nearest_offset_quad,
    )
    sample_coordinates = input_coordinates + offsets
    sample_y = sample_coordinates[..., 0].clamp(0, input_height - 1)
    sample_x = sample_coordinates[..., 1].clamp(0, input_width - 1)
    flat_sample = (sample_y * input_width + sample_x).reshape(batch, 1, -1)
    gathered = torch.gather(
        motion.reshape(batch, motion.shape[1], -1),
        2,
        flat_sample.expand(-1, motion.shape[1], -1),
    ).reshape(batch, motion.shape[1], *output_size_yx)
    scaled = gathered[:, :2] * scale.reshape(1, 2, 1, 1)

    second_square = scaled[:, 1:2] * scaled[:, 1:2]
    length_squared = fused_multiply_add(scaled[:, 0:1], scaled[:, 0:1], second_square)
    length = torch.sqrt(length_squared)
    return scaled * (length > 0.1).to(scaled.dtype)


def reproject_history_uv(
    uv_yx: torch.Tensor,
    motion_yx: torch.Tensor,
    inverse_output_size_yx: torch.Tensor,
) -> torch.Tensor:
    """Reproject UVs with CUDA fused multiply-add single-rounding behavior."""

    return fused_multiply_add(-motion_yx, inverse_output_size_yx, uv_yx)


def sample_bilinear(
    tensor: torch.Tensor,
    uv_yx: torch.Tensor,
    *,
    clamp_to_edge: bool = True,
) -> torch.Tensor:
    """Sample NCHW data with the shader's bilinear term and sum order."""

    batch, channels, height, width = tensor.shape
    if uv_yx.shape[0] != batch:
        raise ValueError(
            f"uv_yx batch must match sampled tensor batch {batch}; got "
            f"{uv_yx.shape[0]}."
        )
    size = uv_yx.new_tensor((height, width))
    sample_position = uv_yx * size
    grid0_float = torch.floor(sample_position - 0.5)
    grid1_float = grid0_float + 1.0
    zero = torch.zeros((), dtype=uv_yx.dtype, device=uv_yx.device)
    weight0 = torch.maximum(1.0 - torch.abs(grid0_float + 0.5 - sample_position), zero)
    weight1 = torch.maximum(1.0 - torch.abs(grid1_float + 0.5 - sample_position), zero)
    if not clamp_to_edge:
        weight0 = weight0 * ((grid0_float >= 0.0) & (grid0_float < size))
        weight1 = weight1 * ((grid1_float >= 0.0) & (grid1_float < size))

    grid0 = grid0_float.to(torch.long)
    grid1 = grid1_float.to(torch.long)
    y0 = grid0[..., 0].clamp(0, height - 1)
    x0 = grid0[..., 1].clamp(0, width - 1)
    y1 = grid1[..., 0].clamp(0, height - 1)
    x1 = grid1[..., 1].clamp(0, width - 1)
    flat_tensor = tensor.reshape(batch, channels, -1)

    def gather(y_coordinate: torch.Tensor, x_coordinate: torch.Tensor) -> torch.Tensor:
        flat_index = (y_coordinate * width + x_coordinate).reshape(batch, 1, -1)
        return torch.gather(
            flat_tensor, 2, flat_index.expand(-1, channels, -1)
        ).reshape(batch, channels, *uv_yx.shape[1:3])

    sample_tl = (
        gather(y0, x0) * weight0[..., 0].unsqueeze(1) * weight0[..., 1].unsqueeze(1)
    )
    sample_tr = (
        gather(y0, x1) * weight0[..., 0].unsqueeze(1) * weight1[..., 1].unsqueeze(1)
    )
    sample_bl = (
        gather(y1, x0) * weight1[..., 0].unsqueeze(1) * weight0[..., 1].unsqueeze(1)
    )
    sample_br = (
        gather(y1, x1) * weight1[..., 0].unsqueeze(1) * weight1[..., 1].unsqueeze(1)
    )
    return sample_tl + sample_tr + sample_bl + sample_br


def sample_catmull_rom(tensor: torch.Tensor, uv_yx: torch.Tensor) -> torch.Tensor:
    """Sample using the optimized five-cross-tap Catmull-Rom shader path."""

    spatial_size = uv_yx.new_tensor(tensor.shape[-2:])
    inverse_spatial_size = torch.reciprocal(spatial_size)
    scaled_uv = uv_yx * spatial_size
    texel_center = torch.floor(scaled_uv - 0.5) + 0.5
    fraction = scaled_uv - texel_center
    fraction_squared = fraction * fraction
    fraction_cubed = fraction_squared * fraction

    weight0 = fraction_squared - 0.5 * (fraction_cubed + fraction)
    weight1 = 1.5 * fraction_cubed - 2.5 * fraction_squared + 1.0
    weight3 = 0.5 * (fraction_cubed - fraction_squared)
    weight2 = 1.0 - weight0 - weight1 - weight3
    combined = torch.stack((weight0, weight1 + weight2, weight3), dim=-2)
    positions = torch.stack(
        (
            texel_center - 1.0,
            texel_center + weight2 / combined[..., 1, :],
            texel_center + 2.0,
        ),
        dim=-2,
    )
    sample_uvs = positions * inverse_spatial_size

    low = sample_uvs[..., 0, :]
    middle = sample_uvs[..., 1, :]
    high = sample_uvs[..., 2, :]
    cross_uvs = (
        torch.stack((middle[..., 0], low[..., 1]), dim=-1),
        torch.stack((low[..., 0], middle[..., 1]), dim=-1),
        middle,
        torch.stack((high[..., 0], middle[..., 1]), dim=-1),
        torch.stack((middle[..., 0], high[..., 1]), dim=-1),
    )
    low_weight = combined[..., 0, :]
    middle_weight = combined[..., 1, :]
    high_weight = combined[..., 2, :]
    cross_weights = (
        middle_weight[..., 0] * low_weight[..., 1],
        low_weight[..., 0] * middle_weight[..., 1],
        middle_weight[..., 0] * middle_weight[..., 1],
        high_weight[..., 0] * middle_weight[..., 1],
        middle_weight[..., 0] * high_weight[..., 1],
    )

    output_shape = (tensor.shape[0], tensor.shape[1], *uv_yx.shape[1:3])
    sample_min = torch.full(
        output_shape, MAX_HALF, dtype=tensor.dtype, device=tensor.device
    )
    sample_max = torch.full(
        output_shape, -MAX_HALF, dtype=tensor.dtype, device=tensor.device
    )
    weighted_sum = None
    corner_weight_sum = None
    for sample_uv, weight in zip(cross_uvs, cross_weights):
        sample = sample_bilinear(tensor, sample_uv, clamp_to_edge=True)
        weighted = sample * weight.unsqueeze(1)
        weighted_sum = weighted if weighted_sum is None else weighted_sum + weighted
        corner_weight_sum = (
            weight if corner_weight_sum is None else corner_weight_sum + weight
        )
        sample_min = torch.minimum(sample_min, sample)
        sample_max = torch.maximum(sample_max, sample)

    final_multiplier = torch.reciprocal(corner_weight_sum)
    color = weighted_sum * final_multiplier.unsqueeze(1)
    clamped = torch.maximum(torch.minimum(color, sample_max), sample_min)
    return torch.where(torch.any(color < 0.0, dim=1, keepdim=True), clamped, color)


def sample_temporal_params(
    temporal_params: torch.Tensor,
    output_size_yx: tuple[int, int],
    preprocess_size_yx: torch.Tensor,
    *,
    sharp_theta: bool,
) -> TemporalParams:
    """Upsample padded temporal parameters at output pixel centers."""

    batch = temporal_params.shape[0]
    coordinates = output_coordinates(batch, output_size_yx, temporal_params.device)
    inverse_output_size = torch.reciprocal(temporal_params.new_tensor(output_size_yx))
    uv = (coordinates.to(temporal_params.dtype) + 0.5) * inverse_output_size
    preprocess_size = preprocess_size_yx.detach().to(
        device=temporal_params.device, dtype=temporal_params.dtype
    )
    preprocess_size = preprocess_size.reshape(batch, 1, 1, 2)
    temporal_size = temporal_params.new_tensor(temporal_params.shape[-2:])
    padded_uv_scale = preprocess_size * torch.reciprocal(temporal_size)
    params = sample_bilinear(temporal_params[:, :3], uv * padded_uv_scale)

    theta = params[:, 0:1]
    if sharp_theta:
        theta = theta.clamp(0.0, 1.0)
        theta_squared = theta * theta
        inverse_theta = 1.0 - theta
        inverse_squared = inverse_theta * inverse_theta
        denominator = torch.maximum(
            theta_squared + inverse_squared,
            theta.new_tensor(1.0e-6),
        )
        theta = theta_squared / denominator
    alpha = params[:, 1:2] * 0.35 + 0.05
    gamma = params[:, 2:3] * 2.0
    return TemporalParams(theta=theta, alpha=alpha, gamma=gamma)
