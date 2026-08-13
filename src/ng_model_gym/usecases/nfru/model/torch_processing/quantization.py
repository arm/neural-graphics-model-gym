# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
"""NFRU motion/depth quantization translated from ``quant.slang``."""

from __future__ import annotations

import torch

BITS_EXP = 4
BITS_X = 7
BITS_Y = 7
MAX_VAL = 1023.0
DEPTH_MAX = 1.1


def symmetric_ceil(value: torch.Tensor) -> torch.Tensor:
    """Round magnitudes upward while preserving sign."""

    return torch.sign(value) * torch.ceil(torch.abs(value))


def quantize_dynamic_motion(value: torch.Tensor) -> torch.Tensor:
    """Apply the runtime dynamic-mask vector quantization literally."""

    max_int = (1 << (BITS_X + BITS_Y)) - 1
    scale = value.new_tensor(float(max_int) / MAX_VAL)
    quantized = symmetric_ceil(value * scale).clamp(-MAX_VAL, MAX_VAL)
    return quantized.to(torch.int32).to(value.dtype) / scale


def pack_depth_motion(motion: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """Pack pixel-space motion and inverted normalized depth into 31 bits."""

    full_bits = BITS_X + BITS_Y
    max_int = (1 << full_bits) - 1
    scale = motion.new_tensor(float(max_int) / MAX_VAL)
    ix = symmetric_ceil(motion[..., 0] * scale).clamp(-max_int, max_int).to(torch.int64)
    iy = symmetric_ceil(motion[..., 1] * scale).clamp(-max_int, max_int).to(torch.int64)

    depth_bits = 31 - full_bits - 2 - BITS_EXP
    max_depth_int = (1 << depth_bits) - 1
    depth_scale = depth.new_tensor(float(max_depth_int) / DEPTH_MAX)
    depth_code = torch.floor(depth * depth_scale + 0.5).to(torch.int64)

    absolute_max = torch.maximum(torch.abs(ix), torch.abs(iy))
    exponent = torch.where(
        absolute_max == 0,
        torch.zeros_like(absolute_max),
        torch.floor(torch.log2(absolute_max.to(torch.float32))).to(torch.int64),
    ).clamp(0, (1 << BITS_EXP) - 1)
    shift = (exponent - (BITS_X - 1)).clamp_min(0)
    mantissa_x = torch.bitwise_right_shift(torch.abs(ix), shift).clamp(
        0, (1 << BITS_X) - 1
    )
    mantissa_y = torch.bitwise_right_shift(torch.abs(iy), shift).clamp(
        0, (1 << BITS_Y) - 1
    )

    sign_y_shift = BITS_X + BITS_Y
    sign_x_shift = sign_y_shift + 1
    exponent_shift = sign_x_shift + 1
    depth_shift = exponent_shift + BITS_EXP
    code = (depth_code & max_depth_int) << depth_shift
    code |= (exponent & ((1 << BITS_EXP) - 1)) << exponent_shift
    code |= (ix < 0).to(torch.int64) << sign_x_shift
    code |= (iy < 0).to(torch.int64) << sign_y_shift
    code |= (mantissa_x & ((1 << BITS_X) - 1)) << BITS_Y
    code |= mantissa_y & ((1 << BITS_Y) - 1)
    return (code & 0x7FFFFFFF).to(torch.int32)


def decode_motion(code: torch.Tensor) -> torch.Tensor:
    """Decode a packed code to its quantized pixel-space motion vector."""

    value = code.to(torch.int64) & 0x7FFFFFFF
    sign_y_shift = BITS_X + BITS_Y
    sign_x_shift = sign_y_shift + 1
    exponent_shift = sign_x_shift + 1
    sign_x = 1 - 2 * ((value >> sign_x_shift) & 1)
    sign_y = 1 - 2 * ((value >> sign_y_shift) & 1)
    exponent = (value >> exponent_shift) & ((1 << BITS_EXP) - 1)
    mantissa_x = (value >> BITS_Y) & ((1 << BITS_X) - 1)
    mantissa_y = value & ((1 << BITS_Y) - 1)
    shift = (exponent - (BITS_X - 1)).clamp_min(0)
    ix = sign_x * torch.bitwise_left_shift(mantissa_x, shift)
    iy = sign_y * torch.bitwise_left_shift(mantissa_y, shift)
    inverse_scale = MAX_VAL / float((1 << (BITS_X + BITS_Y)) - 1)
    return torch.stack((ix, iy), dim=-1).to(torch.float32) * inverse_scale
