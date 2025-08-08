# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import torch


def compute_luminance(image):
    """ITU-R BT.709: `0.2126 * R + 0.7152 * G + 0.0722 * B`"""
    if len(image.shape) == 3:
        weights = torch.tensor([0.2126, 0.7152, 0.0722], device=image.device).view(
            3, 1, 1
        )
        luminance = (weights * image).sum(dim=0)
    # batch case: (N, C, H, W)
    elif len(image.shape) == 4:
        weights = torch.tensor([0.2126, 0.7152, 0.0722], device=image.device).view(
            1, 3, 1, 1
        )
        luminance = (weights * image).sum(dim=1, keepdim=True)
    else:
        raise ValueError("Image must be either of shape [C, H, W] or [N, C, H, W].")

    return luminance


def length(vector) -> torch.Tensor:
    """Assuming 2ch vector, calculates: `sqrt(x^2 + y^2)`"""
    if vector.dim() == 4:
        # B, C, H, W
        y, x = torch.split(vector, 1, dim=1)
        return torch.sqrt(x**2 + y**2)
    if vector.dim() == 2:
        # Split along the last dimension into two chunks, each of size 1
        y, x = torch.split(vector, 1, dim=-1)
        return torch.sqrt(x**2 + y**2)
    return None


def lerp(x, y, a):
    """Performs: `x * (1 - a) + y * a`"""
    return x * (1 - a) + y * a


def linear_01_depth(depth: torch.Tensor, z_near: float, z_far: float) -> torch.Tensor:
    """Linearises the depth data between [0,1]"""
    return 1.0 / (depth * (1.0 - z_far / z_near) + z_far / z_near)


def normalize_mvs_fixed(mvs: torch.Tensor) -> torch.Tensor:
    """Normalize motion."""
    if mvs.dim() == 4:
        v, u = torch.split(mvs, 1, dim=1)
        v = v / 540.0
        u = u / 960.0
        vu = torch.cat([v, u], dim=1)
        return vu
    if mvs.dim() == 2:
        v, u = torch.split(mvs, 1, dim=-1)
        v = v / 540.0
        u = u / 960.0
        vu = torch.cat([v, u], dim=-1)
        return vu
    return None


def normalize_mvs(mvs: torch.Tensor) -> torch.Tensor:
    """Normalizes motion vector by shape: `v = v / height`, `u = u / width`"""
    # B, C, H, W
    sh = mvs.shape
    v, u = torch.split(mvs, 1, dim=1)
    v = v / sh[2]
    u = u / sh[3]
    vu = torch.cat([v, u], dim=1)
    return vu


def fixed_normalize_mvs(
    mvs: torch.Tensor, dim: int = 1, height: int = 544, width: int = 960
) -> torch.Tensor:
    """Normalizes motion vector by shape: `v = v / height`, `u = u / width`"""
    v, u = torch.split(mvs, split_size_or_sections=1, dim=dim)
    norm_v = v / height
    norm_u = u / width
    mvs_norm = torch.concat([norm_v, norm_u], axis=dim)
    return mvs_norm


def generate_lr_to_hr_lut(scale: float, jitter: torch.Tensor) -> torch.Tensor:
    """
    Generates a low-resolution (LR) to high-resolution (HR) lookup table for sparse upsampling.

    This function computes how jittered LR pixel samples map onto a higher-resolution grid,
    based on a given upsampling scale (e.g., 1.5x, 2x). It identifies which LR pixel contributes
    to each HR pixel after jittered projection and reverse-mapping, producing a spatially sparse
    representation suitable for reconstruction or filtering.

    For each valid HR pixel, the output LUT encodes:
      - `dy, dx`: the integer offset to apply to the back-projected LR pixel
      - `mask`: a binary flag indicating whether this HR pixel has a corresponding source sample

    The LUT is padded to 4 channels (dy, dx, valid, 0) for ease of GPU access (e.g., push constants)

    Args:
        scale (float): The upsampling factor (e.g., 1.5, 2.0).
                       Must match a supported rational scale preset internally.
        jitter (Tensor): A tensor of shape (n, 2, h_lr, w_lr) representing subpixel jitter offsets
                         to apply to LR grid positions before projection.

    Returns:
        Tuple[Tensor, int]:
            - Tensor of shape (n, 4, h_hr, w_hr), containing (dy, dx, valid, 0) per HR pixel.
            - Integer idx_mod: the modulo grid size used to tile HR space for kernel filtering.
    """
    scales = {
        2.0: ((1, 1), (2, 2), 2),
        1.5: ((2, 2), (3, 3), 3),
        1.3: ((3, 3), (4, 4), 4),
    }

    if scale not in scales:
        raise ValueError(f"Unsupported scale: {scale}")

    (h_lr, w_lr), (h_hr, w_hr), idx_mod = scales[scale]
    n, _, _, _ = jitter.shape
    device = jitter.device
    sf = torch.tensor(scale, device=device)
    idx_mod = torch.tensor(idx_mod, dtype=torch.int32, device=device)

    # Generate LR pixel grid (n, 2, h_lr, w_lr)
    y, x = torch.meshgrid(
        torch.arange(h_lr, device=device),
        torch.arange(w_lr, device=device),
        indexing="ij",
    )
    lr_grid = torch.stack((y, x), dim=0).float().unsqueeze(0).expand(n, -1, -1, -1)

    # Jitter + scale → projected HR pixel index (n, 2, h_lr, w_lr)
    jitter = jitter.expand(n, 2, h_lr, w_lr)
    hr_pos = (lr_grid + jitter + 0.5) * sf
    hr_idx = hr_pos.floor().int()

    # Back-project and compute offset
    lr_coord = ((hr_idx + 0.5) / sf).floor().int()
    offset = lr_grid.int() - lr_coord  # (n, 2, h_lr, w_lr)

    # Flatten everything
    hy, hx = hr_idx[:, 0].reshape(-1), hr_idx[:, 1].reshape(-1)
    n_idx = (
        torch.arange(n, device=device).view(-1, 1, 1).expand(n, h_lr, w_lr).reshape(-1)
    )
    offset_flat = offset.permute(0, 2, 3, 1).reshape(-1, 2)

    # Filter valid HR coordinates
    valid = (hy >= 0) & (hy < h_hr) & (hx >= 0) & (hx < w_hr)
    n_idx = n_idx[valid]
    hy, hx = hy[valid], hx[valid]
    dy, dx = offset_flat[valid].T

    # Build output, we pad to 4 just for ease of using as push constants later
    out = torch.zeros((n, 4, h_hr, w_hr), dtype=torch.float32, device=device)
    out[n_idx, 0, hy, hx] = dy.float()
    out[n_idx, 1, hy, hx] = dx.float()
    out[n_idx, 2, hy, hx] = 1.0

    return out, idx_mod


def swizzle(x: torch.Tensor, pattern: str) -> torch.Tensor:
    """
    Swizzle the columns of x according to a GLSL-like pattern.

    Supports up to 4D vectors and the standard characters:
      - x, y, z, w
      - r, g, b, a
      - s, t, p, q

    E.g. pattern='yx' for a 2D vector, pattern='xyzw' for a 4D vector, etc.
    """
    # Map GLSL swizzle characters to indices
    # x, r, s -> 0
    # y, g, t -> 1
    # z, b, p -> 2
    # w, a, q -> 3
    # fmt:off
    idx_map = {
        'x': 0, 'r': 0, 's': 0,
        'y': 1, 'g': 1, 't': 1,
        'z': 2, 'b': 2, 'p': 2,
        'w': 3, 'a': 3, 'q': 3,
    }
    # fmt:on

    # Convert each character in the pattern to the corresponding index
    indices = [idx_map[c] for c in pattern]

    # Gather columns according to the pattern; assume features in dim=1
    # If your tensor has features in another dim, change accordingly
    return x[:, indices]


def compute_jitter_tile_offset(
    jitter: torch.Tensor, scale: torch.Tensor, idx_mod: torch.Tensor
) -> Tuple[int, int]:
    """Compute the offset from jitter"""
    jitter_hr = (jitter + torch.tensor(0.5).to(jitter.device)) * torch.tensor(scale).to(
        jitter.device
    )
    offset = (
        jitter_hr.to(torch.int32) - torch.ones_like(jitter_hr).to(torch.int32)
    ) % torch.tensor(idx_mod, dtype=torch.int32).to(jitter.device)

    return offset.to(torch.float32)  # TODO: type casting to match current slang
