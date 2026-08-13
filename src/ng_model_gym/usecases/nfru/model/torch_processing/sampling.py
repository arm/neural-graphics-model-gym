# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
"""Shader-coordinate sampling helpers for the native NFRU Torch backend."""

# Shader translations intentionally keep usecase-specific tap and grid semantics local.
# pylint: disable=duplicate-code

from __future__ import annotations

import math
from numbers import Real

import torch

HOLE_FILL_OFFSETS = (
    (0, 0),
    (1, 0),
    (0, 1),
    (0, -1),
    (-1, 0),
    (-1, 1),
    (1, 1),
    (-1, -1),
    (1, -1),
)


def shader_float(value: Real, name: str) -> float:
    """Validate a scalar and round it to the shader's float32 semantics."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real value.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real value.")
    return torch.tensor(result, dtype=torch.float32).item()


def validate_nchw(
    tensor: torch.Tensor,
    name: str,
    channels: int,
    *,
    device: torch.device | None = None,
    batch: int | None = None,
) -> None:
    """Validate a floating NCHW stage input without copying it."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tensor.ndim != 4 or tensor.shape[1] != channels:
        raise ValueError(
            f"{name} must have shape [N,{channels},H,W]; got {tuple(tensor.shape)}."
        )
    if tensor.shape[0] < 1 or tensor.shape[2] < 1 or tensor.shape[3] < 1:
        raise ValueError(f"{name} must have positive batch and spatial dimensions.")
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must use torch.float32; got {tensor.dtype}.")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}; got tensor on {tensor.device}.")
    if batch is not None and tensor.shape[0] != batch:
        raise ValueError(f"{name} batch must be {batch}; got {tensor.shape[0]}.")


def validate_matrix(
    tensor: torch.Tensor,
    name: str,
    *,
    device: torch.device,
    batch: int,
) -> None:
    """Validate an NFRU batch of 4x4 float32 transforms."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tensor.shape != (batch, 4, 4):
        raise ValueError(
            f"{name} must have shape [{batch},4,4]; got {tuple(tensor.shape)}."
        )
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must use torch.float32; got {tensor.dtype}.")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}; got tensor on {tensor.device}.")


def make_pixel_grid(
    spatial_shape: tuple[int, int],
    batch: int,
    device: torch.device,
) -> torch.Tensor:
    """Return integer pixels in the shaders' ``(height, width)`` order."""

    rows = torch.arange(spatial_shape[0], device=device, dtype=torch.int64)
    columns = torch.arange(spatial_shape[1], device=device, dtype=torch.int64)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    return (
        torch.stack((row_grid, column_grid), dim=-1)
        .unsqueeze(0)
        .expand(batch, -1, -1, -1)
    )


def make_uv_grid(
    spatial_shape: tuple[int, int],
    batch: int,
    device: torch.device,
) -> torch.Tensor:
    """Return float32 pixel-center UVs in ``(height, width)`` order."""

    pixels = make_pixel_grid(spatial_shape, batch, device).to(torch.float32)
    size = pixels.new_tensor(spatial_shape)
    return (pixels + 0.5) * torch.reciprocal(size)


def uv_to_pixels(uv: torch.Tensor, spatial_shape: tuple[int, int]) -> torch.Tensor:
    """Convert normalized shader UVs to nearest-load integer coordinates."""

    return torch.floor(uv.detach() * uv.new_tensor(spatial_shape)).to(torch.int64)


def gather_pixels(
    source: torch.Tensor,
    pixels: torch.Tensor,
    *,
    clamp_to_edge: bool,
) -> torch.Tensor:
    """Gather NCHW values at integer ``(height, width)`` coordinates."""

    if source.ndim != 4 or pixels.ndim != 4 or pixels.shape[-1] != 2:
        raise ValueError("Expected source NCHW and pixels [N,H,W,2].")
    batch, channels, height, width = source.shape
    if pixels.shape[0] not in (1, batch):
        raise ValueError("Pixel batch must be one or match the source batch.")
    pixels = pixels.detach().to(device=source.device, dtype=torch.int64)
    pixels = pixels.expand(batch, -1, -1, -1)
    valid = (
        (pixels[..., 0] >= 0)
        & (pixels[..., 0] < height)
        & (pixels[..., 1] >= 0)
        & (pixels[..., 1] < width)
    )
    rows = pixels[..., 0].clamp(0, height - 1)
    columns = pixels[..., 1].clamp(0, width - 1)
    flat_index = (rows * width + columns).flatten(1)
    flat_index = flat_index.unsqueeze(1).expand(-1, channels, -1)
    result = (
        source.flatten(2)
        .gather(2, flat_index)
        .reshape(batch, channels, pixels.shape[1], pixels.shape[2])
    )
    if not clamp_to_edge:
        result = result * valid.unsqueeze(1).to(result.dtype)
    return result


def nearest_sample(source: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Sample normalized UVs with the TensorView zero-OOB nearest semantics."""

    return gather_pixels(
        source,
        uv_to_pixels(uv, source.shape[-2:]),
        clamp_to_edge=False,
    )


def bilinear_sample(source: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Translate ``sample_tensor`` with edge clamping and shader term order."""

    if source.ndim != 4 or uv.ndim != 4 or uv.shape[-1] != 2:
        raise ValueError("Expected source NCHW and uv [N,H,W,2].")
    batch, _, height, width = source.shape
    if uv.shape[0] not in (1, batch):
        raise ValueError("UV batch must be one or match the source batch.")
    coordinates = uv.detach().to(device=source.device, dtype=source.dtype)
    coordinates = coordinates.expand(batch, -1, -1, -1)
    position = coordinates * coordinates.new_tensor((height, width))
    grid0_float = torch.floor(position - 0.5)
    grid1_float = grid0_float + 1.0
    weight0 = torch.clamp_min(1.0 - torch.abs(grid0_float + 0.5 - position), 0.0)
    weight1 = torch.clamp_min(1.0 - torch.abs(grid1_float + 0.5 - position), 0.0)
    grid0 = grid0_float.to(torch.int64)
    grid1 = grid1_float.to(torch.int64)

    top_left = grid0
    top_right = torch.stack((grid0[..., 0], grid1[..., 1]), dim=-1)
    bottom_left = torch.stack((grid1[..., 0], grid0[..., 1]), dim=-1)
    bottom_right = grid1
    tl = gather_pixels(source, top_left, clamp_to_edge=True)
    tr = gather_pixels(source, top_right, clamp_to_edge=True)
    bl = gather_pixels(source, bottom_left, clamp_to_edge=True)
    br = gather_pixels(source, bottom_right, clamp_to_edge=True)

    row0 = weight0[..., 0].unsqueeze(1)
    column0 = weight0[..., 1].unsqueeze(1)
    row1 = weight1[..., 0].unsqueeze(1)
    column1 = weight1[..., 1].unsqueeze(1)
    return (
        tl * row0 * column0
        + tr * row0 * column1
        + bl * row1 * column0
        + br * row1 * column1
    )


def ordered_nearest_depth(
    depth: torch.Tensor, pixels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the strict-min depth and offset from the ordered nine shader taps."""

    nearest_depth = gather_pixels(depth, pixels, clamp_to_edge=True)[:, 0]
    nearest_offset = torch.zeros_like(pixels)
    for row_offset, column_offset in HOLE_FILL_OFFSETS[1:]:
        offset = pixels.new_tensor((row_offset, column_offset))
        candidate = gather_pixels(depth, pixels + offset, clamp_to_edge=True)[:, 0]
        take = candidate < nearest_depth
        nearest_depth = torch.where(take, candidate, nearest_depth)
        nearest_offset = torch.where(
            take.unsqueeze(-1), offset.view(1, 1, 1, 2), nearest_offset
        )
    return nearest_depth, nearest_offset


def ordered_packed_max(packed: torch.Tensor) -> torch.Tensor:
    """Gather the ordered nine packed taps, retaining the first equal maximum."""

    batch, _, height, width = packed.shape
    pixels = make_pixel_grid((height, width), batch, packed.device)
    nearest = gather_pixels(packed, pixels, clamp_to_edge=True)[:, 0]
    for row_offset, column_offset in HOLE_FILL_OFFSETS[1:]:
        offset = pixels.new_tensor((row_offset, column_offset))
        candidate = gather_pixels(packed, pixels + offset, clamp_to_edge=True)[:, 0]
        nearest = torch.where(candidate > nearest, candidate, nearest)
    return nearest.unsqueeze(1)
