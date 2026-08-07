# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
"""Sampling and nearest-depth helpers for the Torch NSS-v1 preprocessor.

Spatial vectors in the NSS shaders are ordered as ``(height, width)``.  This
module deliberately keeps that convention instead of translating through the
usual PyTorch ``grid_sample`` ``(x, y)`` convention.
"""

from collections.abc import Sequence

import torch

_HIGH_NEAREST_OFFSETS = (
    (0, 0),
    (1, 0),
    (0, 1),
    (0, -1),
    (-1, 0),
    (-1, 1),
    (1, 1),
    (-1, -1),
    (1, -1),
    (-1, 2),
    (0, 2),
    (1, 2),
    (2, 2),
    (2, 1),
    (2, 0),
    (2, -1),
)

_LOW_NEAREST_OFFSETS = (
    (0, 0),
    (0, -1),
    (-1, 0),
    (-1, -1),
    (0, 1),
    (-1, 1),
    (-1, 2),
    (0, 2),
    (1, 0),
    (1, -1),
    (2, 0),
    (2, -1),
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
)


def make_pixel_grid(
    spatial_shape: tuple[int, int],
    *,
    batch_size: int = 1,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return an ``(N, H, W, 2)`` integer grid in ``(row, column)`` order."""

    height, width = spatial_shape
    rows = torch.arange(height, device=device, dtype=torch.int64)
    columns = torch.arange(width, device=device, dtype=torch.int64)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    grid = torch.stack((row_grid, column_grid), dim=-1)
    return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)


def make_uv_grid(
    spatial_shape: tuple[int, int],
    *,
    batch_size: int = 1,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return pixel-centre UVs with shape ``(N, H, W, 2)``."""

    height, width = spatial_shape
    pixels = make_pixel_grid(
        spatial_shape,
        batch_size=batch_size,
        device=device,
    ).to(dtype=dtype)
    size = pixels.new_tensor((height, width))
    return (pixels + 0.5) * torch.reciprocal(size)


def reflect_indices(coordinates: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Apply the shader's single-step edge reflection to spatial coordinates."""

    result = coordinates.detach()
    spatial_size = torch.as_tensor(size, device=result.device, dtype=result.dtype)
    result = torch.where(result < 0, -result - 1, result)
    return torch.where(result >= spatial_size, 2 * spatial_size - result - 1, result)


def gather_pixels(source: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
    """Gather ``source`` at integer row/column coordinates.

    ``source`` has shape ``(N, C, H, W)`` and ``pixels`` has shape
    ``(N|1, OH, OW, 2)``.  Coordinates are clamped before the load, matching
    the safe-load pattern used by the shaders.  The returned shape is
    ``(N, C, OH, OW)``.
    """

    if source.ndim != 4 or pixels.ndim != 4 or pixels.shape[-1] != 2:
        raise ValueError("Expected source NCHW and pixels (N, H, W, 2).")
    batch_size, channels, height, width = source.shape
    if pixels.shape[0] not in (1, batch_size):
        raise ValueError("Pixel-coordinate batch must be one or match source batch.")

    pixels = pixels.detach().to(device=source.device, dtype=torch.int64)
    pixels = pixels.expand(batch_size, -1, -1, -1)
    rows = pixels[..., 0].clamp(0, height - 1)
    columns = pixels[..., 1].clamp(0, width - 1)
    flat_index = (rows * width + columns).flatten(1)
    flat_index = flat_index.unsqueeze(1).expand(-1, channels, -1)
    gathered = source.flatten(2).gather(2, flat_index)
    return gathered.reshape(batch_size, channels, pixels.shape[1], pixels.shape[2])


def bilinear_sample(
    source: torch.Tensor,
    uv: torch.Tensor,
    *,
    clamp_to_edge: bool = True,
) -> torch.Tensor:
    """Explicitly sample an NCHW tensor at row/column UV coordinates.

    This translates ``sample_tensor`` from ``tensor.slang``.  Coordinates and
    all indexing decisions are detached, while gradients through the four
    sampled values remain native PyTorch gradients.  Multiplication and
    summation follow the shader's TL/TR/BL/BR order.
    """

    if source.ndim != 4 or uv.ndim != 4 or uv.shape[-1] != 2:
        raise ValueError("Expected source NCHW and uv (N, H, W, 2).")
    batch_size, _, height, width = source.shape
    if uv.shape[0] not in (1, batch_size):
        raise ValueError("UV batch must be one or match source batch.")

    uv = uv.detach().to(device=source.device, dtype=source.dtype)
    uv = uv.expand(batch_size, -1, -1, -1)
    size = uv.new_tensor((height, width))
    sample_position = uv * size
    base = torch.floor(sample_position - 0.5)
    next_position = base + 1.0

    # lerp_weight(g + 0.5, sample_position) is 1 - abs(...) for these taps.
    weight_base = 1.0 - torch.abs((base + 0.5) - sample_position)
    weight_next = 1.0 - torch.abs((next_position + 0.5) - sample_position)
    weight_base = torch.clamp_min(weight_base, 0.0)
    weight_next = torch.clamp_min(weight_next, 0.0)
    if not clamp_to_edge:
        weight_base = weight_base * ((base >= 0.0) & (base < size)).to(source.dtype)
        weight_next = weight_next * (
            (next_position >= 0.0) & (next_position < size)
        ).to(source.dtype)

    base_int = base.to(torch.int64)
    next_int = next_position.to(torch.int64)
    top_left = torch.stack((base_int[..., 0], base_int[..., 1]), dim=-1)
    top_right = torch.stack((base_int[..., 0], next_int[..., 1]), dim=-1)
    bottom_left = torch.stack((next_int[..., 0], base_int[..., 1]), dim=-1)
    bottom_right = next_int

    value_tl = gather_pixels(source, top_left)
    value_tr = gather_pixels(source, top_right)
    value_bl = gather_pixels(source, bottom_left)
    value_br = gather_pixels(source, bottom_right)

    row_base = weight_base[..., 0].unsqueeze(1)
    column_base = weight_base[..., 1].unsqueeze(1)
    row_next = weight_next[..., 0].unsqueeze(1)
    column_next = weight_next[..., 1].unsqueeze(1)
    term_tl = value_tl * row_base * column_base
    term_tr = value_tr * row_base * column_next
    term_bl = value_bl * row_next * column_base
    term_br = value_br * row_next * column_next
    return term_tl + term_tr + term_bl + term_br


def _find_nearest_from_offsets(
    depth: torch.Tensor,
    pixels: torch.Tensor,
    offsets: Sequence[tuple[int, int]],
    *,
    replace_equal: bool,
    inverted: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run an ordered nearest-depth reduction over spatial tensors."""

    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError("Depth must have shape (N, 1, H, W).")
    batch_size, _, height, width = depth.shape
    if pixels.shape[0] not in (1, batch_size):
        raise ValueError("Pixel-coordinate batch must be one or match depth batch.")

    pixels = pixels.detach().to(device=depth.device, dtype=torch.int64)
    pixels = pixels.expand(batch_size, -1, -1, -1)
    nearest_coord = pixels.clone()
    nearest_offset = torch.zeros_like(pixels)
    nearest_depth = gather_pixels(depth.detach(), pixels)[:, 0]

    for row_offset, column_offset in offsets[1:]:
        offset = pixels.new_tensor((row_offset, column_offset))
        sample_position = pixels + offset
        onscreen = (
            (sample_position[..., 0] >= 0)
            & (sample_position[..., 0] < height)
            & (sample_position[..., 1] >= 0)
            & (sample_position[..., 1] < width)
        )
        sample_depth = gather_pixels(depth.detach(), sample_position)[:, 0]
        if inverted:
            closer = (
                sample_depth >= nearest_depth
                if replace_equal
                else sample_depth > nearest_depth
            )
        else:
            closer = (
                sample_depth <= nearest_depth
                if replace_equal
                else sample_depth < nearest_depth
            )
        take = onscreen & closer
        nearest_depth = torch.where(take, sample_depth, nearest_depth)
        nearest_coord = torch.where(take.unsqueeze(-1), sample_position, nearest_coord)
        expanded_offset = offset.view(1, 1, 1, 2).expand_as(nearest_offset)
        nearest_offset = torch.where(
            take.unsqueeze(-1), expanded_offset, nearest_offset
        )

    return nearest_depth, nearest_coord, nearest_offset


def find_nearest_depth_4x4(
    depth: torch.Tensor,
    uv: torch.Tensor,
    *,
    inverted: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Translate high-quality ``FindNearestDepth_4x4``.

    Equal values retain the first candidate because the shader uses strict
    comparisons in this path.
    """

    height, width = depth.shape[-2:]
    size = uv.detach().new_tensor((height, width))
    pixels = (uv.detach() * size).to(torch.int64)
    return _find_nearest_from_offsets(
        depth,
        pixels,
        _HIGH_NEAREST_OFFSETS,
        replace_equal=False,
        inverted=inverted,
    )


def find_nearest_depth_4x4_from_pixels(
    depth: torch.Tensor,
    pixels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Translate low/mid ``FindNearestDepth_4x4_FromPixel``.

    This path uses ``step(sample, nearest)`` and therefore replaces on equal
    depth: the last equal candidate in the literal shader order wins.
    """

    height, width = depth.shape[-2:]
    pixels = pixels.detach().to(device=depth.device, dtype=torch.int64)
    pixels = torch.stack(
        (
            pixels[..., 0].clamp(0, height - 1),
            pixels[..., 1].clamp(0, width - 1),
        ),
        dim=-1,
    )
    return _find_nearest_from_offsets(
        depth,
        pixels,
        _LOW_NEAREST_OFFSETS,
        replace_equal=True,
    )


def encode_nearest_offsets(
    offsets: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Encode row/column offsets into the high-quality byte layout."""

    offsets = offsets.detach().to(torch.int64).clamp(-2, 2)
    code = ((offsets[..., 1] + 2) << 3) | (offsets[..., 0] + 2)
    return (code.to(dtype=dtype) / 255.0).unsqueeze(1)


def _encode_nearest_offset_nibbles(offsets: torch.Tensor) -> torch.Tensor:
    offsets = offsets.detach().to(torch.int64).clamp(-1, 2)
    return ((offsets[..., 1] + 1) << 2) | (offsets[..., 0] + 1)


def pack_nearest_offsets(
    offset_00: torch.Tensor,
    offset_10: torch.Tensor,
    offset_01: torch.Tensor,
    offset_11: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Pack four low/mid offsets into the shader's two byte channels."""

    byte_r = _encode_nearest_offset_nibbles(offset_00) | (
        _encode_nearest_offset_nibbles(offset_10) << 4
    )
    byte_g = _encode_nearest_offset_nibbles(offset_01) | (
        _encode_nearest_offset_nibbles(offset_11) << 4
    )
    return torch.stack((byte_r, byte_g), dim=1).to(dtype=dtype) / 255.0


def encode_packed_nearest_offsets(
    depth: torch.Tensor,
    base_pixels: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Find and pack the four 2x2-lane offsets used by low/mid quality."""

    height, width = depth.shape[-2:]
    base_pixels = base_pixels.detach().to(device=depth.device, dtype=torch.int64)
    lane_00 = base_pixels
    lane_10 = base_pixels + base_pixels.new_tensor((0, 1))
    lane_01 = base_pixels + base_pixels.new_tensor((1, 0))
    lane_11 = base_pixels + base_pixels.new_tensor((1, 1))

    def clamp_lane(lane: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                lane[..., 0].clamp(0, height - 1),
                lane[..., 1].clamp(0, width - 1),
            ),
            dim=-1,
        )

    offset_00 = find_nearest_depth_4x4_from_pixels(depth, clamp_lane(lane_00))[2]
    offset_10 = find_nearest_depth_4x4_from_pixels(depth, clamp_lane(lane_10))[2]
    offset_01 = find_nearest_depth_4x4_from_pixels(depth, clamp_lane(lane_01))[2]
    offset_11 = find_nearest_depth_4x4_from_pixels(depth, clamp_lane(lane_11))[2]
    return pack_nearest_offsets(
        offset_00,
        offset_10,
        offset_01,
        offset_11,
        dtype=dtype,
    )
