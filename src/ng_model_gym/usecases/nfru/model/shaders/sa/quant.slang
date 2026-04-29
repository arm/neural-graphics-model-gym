// SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
// its affiliates <open-source-office@arm.com></text>
// SPDX-License-Identifier: Apache-2.0

#include "constants.slang"
#include "glsl_mappings.slang"

#ifndef QUANT_DEF
#define QUANT_DEF


float sym_ceil(float x) {
    return sign(x) * ceil(abs(x));
}

// Pack flow (vec2) and depth -> 32-bit int code
// Layout (MSB->LSB): depth | exponent | signX | signY | mantissaX | mantissaY
// -----------------------------------------------------------------------------
uint32_t qFloat(float2 flow, float depth) {
    int32_t full_bits = BITS_X + BITS_Y;
    int32_t max_int = (1 << full_bits) - 1;
    float scale = float(max_int) / MAX_VAL;

    int32_t depth_bits = 31 - full_bits - 2 - BITS_EXP;
    int32_t max_int_d = (1 << depth_bits) - 1;
    float depth_max = float(max_int_d) / DEPTH_MAX;

    int32_t ix = int32_t(clamp(sym_ceil(flow.x * scale), -float(max_int), float(max_int)));
    int32_t iy = int32_t(clamp(sym_ceil(flow.y * scale), -float(max_int), float(max_int)));
    int32_t id = int32_t(round(depth * depth_max));

    int32_t absmax = max(abs(ix), abs(iy));
    int32_t exp = clamp(findMSB(absmax), 0, (1 << BITS_EXP) - 1);

    // exp needs to be cast to int32 here to avoid underflow
    int32_t shift = max(exp - (int32_t(BITS_X) - 1), 0);
    int32_t mx = clamp((abs(ix) >> shift), 0, (1 << BITS_X) - 1);
    int32_t my = clamp((abs(iy) >> shift), 0, (1 << BITS_Y) - 1);

    int32_t sY_shift = BITS_X + BITS_Y;
    int32_t sX_shift = sY_shift + 1;
    int32_t exp_shift = sX_shift + 1;
    int32_t depth_shift = exp_shift + BITS_EXP;

    uint32_t code = 0;
    code |= (id & max_int_d) << depth_shift;
    code |= (exp & ((1 << BITS_EXP) - 1)) << exp_shift;
    code |= ((ix < 0) ? 1 : 0) << sX_shift;
    code |= ((iy < 0) ? 1 : 0) << sY_shift;
    code |= (mx & ((1 << BITS_X) - 1)) << BITS_Y;
    code |= (my & ((1 << BITS_Y) - 1));

    code &= 0x7FFFFFFF;

    return code;
}

uint32_t qFloat(float2 flow, int depth) {
    int32_t full_bits = BITS_X + BITS_Y;
    int32_t max_int = (1 << full_bits) - 1;
    float scale = float(max_int) / MAX_VAL;

    int32_t depth_bits = 31 - full_bits - 2 - BITS_EXP;
    int32_t max_int_d = (1 << depth_bits) - 1;
    float depth_max = float(max_int_d) / DEPTH_MAX;

    int32_t ix = int32_t(clamp(sym_ceil(flow.x * scale), -float(max_int), float(max_int)));
    int32_t iy = int32_t(clamp(sym_ceil(flow.y * scale), -float(max_int), float(max_int)));
    int32_t id = depth;

    int32_t absmax = max(abs(ix), abs(iy));
    int32_t exp = clamp(findMSB(absmax), 0, (1 << BITS_EXP) - 1);

    // exp needs to be cast to int32 here to avoid underflow
    int32_t shift = max(exp - (int32_t(BITS_X) - 1), 0);
    int32_t mx = clamp((abs(ix) >> shift), 0, (1 << BITS_X) - 1);
    int32_t my = clamp((abs(iy) >> shift), 0, (1 << BITS_Y) - 1);

    int32_t sY_shift = BITS_X + BITS_Y;
    int32_t sX_shift = sY_shift + 1;
    int32_t exp_shift = sX_shift + 1;
    int32_t depth_shift = exp_shift + BITS_EXP;

    uint32_t code = 0;
    code |= (id & max_int_d) << depth_shift;
    code |= (exp & ((1 << BITS_EXP) - 1)) << exp_shift;
    code |= ((ix < 0) ? 1 : 0) << sX_shift;
    code |= ((iy < 0) ? 1 : 0) << sY_shift;
    code |= (mx & ((1 << BITS_X) - 1)) << BITS_Y;
    code |= (my & ((1 << BITS_Y) - 1));

    code &= 0x7FFFFFFF;

    return code;
}

float2 dqFloatMv(uint32_t c) {
    c &= 0x7FFFFFFF;
    uint32_t sY_shift = BITS_X + BITS_Y;
    uint32_t sX_shift = sY_shift + 1;
    uint32_t exp_shift = sX_shift + 1;

    int32_t sx = 1 - 2 * int32_t((c >> sX_shift) & 1);
    int32_t sy = 1 - 2 * int32_t((c >> sY_shift) & 1);

    uint32_t exp = (c >> exp_shift) & ((1 << BITS_EXP) - 1);
    uint32_t mx = (c >> BITS_Y) & ((1 << BITS_X) - 1);
    uint32_t my = c & ((1 << BITS_Y) - 1);

    // exp needs to be cast to int32 here to avoid underflow
    uint32_t shift = uint32_t(max(int32_t(exp) - (int32_t(BITS_X) - 1), 0));
    int32_t ix = sx * int32_t(mx << shift);
    int32_t iy = sy * int32_t(my << shift);

    const float inv_scale = MAX_VAL / float((1 << (BITS_X + BITS_Y)) - 1);
    return float2(float(ix), float(iy)) * inv_scale;
}

#endif
