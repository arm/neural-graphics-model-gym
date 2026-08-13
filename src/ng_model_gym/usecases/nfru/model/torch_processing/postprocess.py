# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
"""Native Torch translation of NFRU learned compositing."""

from __future__ import annotations

import torch

from .sampling import (
    bilinear_sample,
    gather_pixels,
    make_uv_grid,
    shader_float,
    uv_to_pixels,
    validate_nchw,
)

_FLOAT_MIN = 1.175494351e-38


def postprocess_torch(
    warped_flow: torch.Tensor,
    warped_mv: torch.Tensor,
    rgb_m1: torch.Tensor,
    rgb_p1: torch.Tensor,
    learnt_params: torch.Tensor,
    timestep: float,
) -> torch.Tensor:
    """Upsample blend parameters, warp four colors, and composite them."""

    validate_nchw(rgb_m1, "rgb_m1", 3)
    batch = rgb_m1.shape[0]
    device = rgb_m1.device
    validate_nchw(rgb_p1, "rgb_p1", 3, device=device, batch=batch)
    validate_nchw(warped_flow, "warped_flow", 2, device=device, batch=batch)
    validate_nchw(warped_mv, "warped_mv", 2, device=device, batch=batch)
    validate_nchw(learnt_params, "learnt_params", 4, device=device, batch=batch)
    if rgb_p1.shape != rgb_m1.shape:
        raise ValueError("rgb_p1 shape must match rgb_m1.")
    scale = shader_float(timestep, "timestep")

    uv = make_uv_grid(rgb_m1.shape[-2:], batch, device)
    # TensorView inputs have no Slang VJP. Keep only in_params differentiable.
    flow = gather_pixels(
        warped_flow.detach(),
        uv_to_pixels(uv, warped_flow.shape[-2:]),
        clamp_to_edge=False,
    ).permute(0, 2, 3, 1)
    motion = gather_pixels(
        warped_mv.detach(),
        uv_to_pixels(uv, warped_mv.shape[-2:]),
        clamp_to_edge=False,
    ).permute(0, 2, 3, 1)
    params = bilinear_sample(learnt_params, uv)
    uv_m1_mv = uv + motion * scale
    uv_p1_mv = uv - motion * (1.0 - scale)
    uv_m1_flow = uv - flow * scale
    uv_p1_flow = uv + flow * (1.0 - scale)
    candidates = torch.stack(
        (
            bilinear_sample(rgb_m1.detach(), uv_m1_mv),
            bilinear_sample(rgb_p1.detach(), uv_p1_mv),
            bilinear_sample(rgb_m1.detach(), uv_m1_flow),
            bilinear_sample(rgb_p1.detach(), uv_p1_flow),
        ),
        dim=1,
    )
    maximum = torch.maximum(params[:, 0:1], params[:, 1:2])
    maximum = torch.maximum(maximum, params[:, 2:3])
    maximum = torch.maximum(maximum, params[:, 3:4])
    stable = params - maximum
    exponent = torch.exp(stable)
    denominator = exponent[:, 0:1] + exponent[:, 1:2]
    denominator = denominator + exponent[:, 2:3]
    denominator = denominator + exponent[:, 3:4]
    weights = exponent / (denominator + _FLOAT_MIN)
    weighted = candidates * weights.unsqueeze(2)
    result = weighted[:, 0]
    result = result + weighted[:, 1]
    result = result + weighted[:, 2]
    result = result + weighted[:, 3]
    return torch.nan_to_num(result)
