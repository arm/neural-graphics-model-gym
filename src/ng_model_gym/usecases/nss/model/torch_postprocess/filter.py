# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
"""KPN filtering primitives for native NSS v1 postprocessing."""

from typing import NamedTuple

import torch

from .sampling import EPS, expand_batch, MAX_HALF, output_coordinates


class FilterColorResult(NamedTuple):
    """Filtered moments and the exact-center color sample."""

    m1: torch.Tensor
    m2: torch.Tensor
    center_sample: torch.Tensor


SPARSE_PRUNED_CHANNEL_MAP = (
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    0,
    1,
    2,
    3,
    -1,
    -1,
    4,
    5,
    6,
    7,
    -1,
    -1,
    8,
    9,
    10,
    11,
    -1,
    -1,
    12,
    13,
    14,
    15,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
)


def map_sparse_2x2_kpn_channels(tap_channels: torch.Tensor) -> torch.Tensor:
    """Map all 36 6x6 LUT channels to retained sparse 4x4 channels."""

    channels = tap_channels.detach().to(torch.long)
    table = channels.new_tensor(SPARSE_PRUNED_CHANNEL_MAP)
    valid = (channels >= 0) & (channels < len(SPARSE_PRUNED_CHANNEL_MAP))
    safe_channels = channels.clamp(0, len(SPARSE_PRUNED_CHANNEL_MAP) - 1)
    return torch.where(valid, table[safe_channels], -1)


def kpn_coordinates(
    lr_coordinates_yx: torch.Tensor,
    input_size_yx: tuple[int, int],
    kpn_size_yx: tuple[int, int],
    temporal_size_yx: tuple[int, int],
    *,
    use_sparse_filter_2x2: bool,
    preprocess_half_res_input: bool,
) -> torch.Tensor:
    """Map LR coordinates to KPN texels with the quality-specific shader rule."""

    coordinates = lr_coordinates_yx.detach().to(torch.long)
    if use_sparse_filter_2x2 or preprocess_half_res_input:
        mapped_y = coordinates[..., 0] * kpn_size_yx[0] // max(input_size_yx[0], 1)
        mapped_x = coordinates[..., 1] * kpn_size_yx[1] // max(input_size_yx[1], 1)
    else:
        kpn_scale = coordinates.new_tensor(kpn_size_yx, dtype=torch.float32) / (
            coordinates.new_tensor(temporal_size_yx, dtype=torch.float32)
        )
        coordinate_float = coordinates.to(torch.float32)
        mapped_y = torch.floor(
            (coordinate_float[..., 0] + 0.5 + 1.0e-3) * kpn_scale[0]
        ).to(torch.long)
        mapped_x = torch.floor(
            (coordinate_float[..., 1] + 0.5 + 1.0e-3) * kpn_scale[1]
        ).to(torch.long)
    return torch.stack(
        (
            mapped_y.clamp(0, kpn_size_yx[0] - 1),
            mapped_x.clamp(0, kpn_size_yx[1] - 1),
        ),
        dim=-1,
    )


def filter_color(  # pylint: disable=too-many-arguments,too-many-locals
    color: torch.Tensor,
    kpn_params: torch.Tensor,
    offset_lut: torch.Tensor,
    idx_modulo_yx: torch.Tensor,
    exposure: torch.Tensor,
    output_size_yx: tuple[int, int],
    temporal_size_yx: tuple[int, int],
    *,
    preprocess_half_res_input: bool,
    use_sparse_filter_2x2: bool,
    filter_kernel_taps: int,
) -> FilterColorResult:
    """Translate the shader's ordered sparse LUT/KPN filter accumulation."""

    batch, _, input_height, input_width = color.shape
    output_height, output_width = output_size_yx
    coordinates = output_coordinates(batch, output_size_yx, color.device)

    modulo = idx_modulo_yx.detach().reshape(idx_modulo_yx.shape[0], -1)[:, :2]
    if not preprocess_half_res_input:
        modulo = modulo[0:1]
    modulo = expand_batch(modulo, batch).to(torch.long)
    lut_index = torch.remainder(coordinates[..., 0], modulo[:, None, None, 0])
    lut_index = lut_index * modulo[:, None, None, 1] + torch.remainder(
        coordinates[..., 1], modulo[:, None, None, 1]
    )

    lut = offset_lut.detach()
    flat_lut = lut.reshape(batch, 6, -1)
    input_size = lut.new_tensor((input_height, input_width))
    output_size = lut.new_tensor(output_size_yx)
    scale = output_size / input_size
    inverse_scale = torch.reciprocal(scale)
    batch_exposure = exposure.detach().reshape(batch, -1)[:, 0]
    flat_color = torch.minimum(
        color[:, :3].detach() * batch_exposure[:, None, None, None],
        color.new_tensor(MAX_HALF),
    ).reshape(batch, 3, -1)

    m1_sum = flat_color.new_zeros((batch, 3, output_height, output_width))
    m2_sum = torch.zeros_like(m1_sum)
    weight_sum = flat_color.new_zeros((batch, output_height, output_width))
    center_color = torch.zeros_like(m1_sum)
    center_valid = torch.zeros(
        (batch, output_height, output_width),
        dtype=torch.bool,
        device=color.device,
    )

    kpn_height, kpn_width = kpn_params.shape[-2:]
    flat_kpn = kpn_params.reshape(batch, -1)
    for tap_index in range(filter_kernel_taps):
        flat_lut_index = lut_index * filter_kernel_taps + tap_index
        tap = torch.gather(
            flat_lut,
            2,
            flat_lut_index.reshape(batch, 1, -1).expand(-1, 6, -1),
        ).reshape(batch, 6, output_height, output_width)

        tap_position = torch.stack((tap[:, 3], tap[:, 4]), dim=-1).to(torch.long)
        output_tap = coordinates + tap_position
        lr_coordinates = torch.floor(
            (output_tap.to(lut.dtype) + 0.5) * inverse_scale + 1.0e-3
        ).to(torch.long)
        tile_offset = torch.stack((tap[:, 0], tap[:, 1]), dim=-1).to(torch.long)
        lr_coordinates = lr_coordinates + tile_offset
        lr_y = lr_coordinates[..., 0].clamp(0, input_height - 1)
        lr_x = lr_coordinates[..., 1].clamp(0, input_width - 1)
        lr_coordinates = torch.stack((lr_y, lr_x), dim=-1)

        flat_color_index = (lr_y * input_width + lr_x).reshape(batch, 1, -1)
        color_tap = torch.gather(
            flat_color, 2, flat_color_index.expand(-1, 3, -1)
        ).reshape(batch, 3, output_height, output_width)
        is_center = (tap[:, 3] == 0.0) & (tap[:, 4] == 0.0)
        center_color = torch.where(is_center[:, None], color_tap, center_color)
        center_valid = center_valid | is_center

        kpn_coord = kpn_coordinates(
            lr_coordinates,
            (input_height, input_width),
            (kpn_height, kpn_width),
            temporal_size_yx,
            use_sparse_filter_2x2=use_sparse_filter_2x2,
            preprocess_half_res_input=preprocess_half_res_input,
        )
        raw_channel = tap[:, 5].to(torch.long)
        if use_sparse_filter_2x2:
            mapped_channel = map_sparse_2x2_kpn_channels(raw_channel)
            valid_channel = mapped_channel >= 0
        else:
            mapped_channel = raw_channel
            valid_channel = torch.ones_like(mapped_channel, dtype=torch.bool)
        mapped_channel = mapped_channel.clamp(0, kpn_params.shape[1] - 1)
        flat_kpn_index = (
            mapped_channel * (kpn_height * kpn_width)
            + kpn_coord[..., 0] * kpn_width
            + kpn_coord[..., 1]
        )
        raw_weight = torch.gather(
            flat_kpn, 1, flat_kpn_index.reshape(batch, -1)
        ).reshape(batch, output_height, output_width)
        clamped_weight = torch.maximum(raw_weight, raw_weight.new_tensor(EPS))
        weight = torch.where(
            valid_channel, clamped_weight, torch.zeros_like(clamped_weight)
        )
        weight = weight * tap[:, 2]

        m1_sum = torch.addcmul(m1_sum, color_tap, weight[:, None])
        m2_sum = torch.addcmul(m2_sum, color_tap * color_tap, weight[:, None])
        weight_sum = weight_sum + weight

    center_sample = torch.cat(
        (center_color, center_valid[:, None].to(center_color.dtype)), dim=1
    )
    denominator = torch.maximum(weight_sum, weight_sum.new_tensor(EPS)).unsqueeze(1)
    return FilterColorResult(
        m1=m1_sum / denominator,
        m2=m2_sum / denominator,
        center_sample=center_sample,
    )
