// SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
// its affiliates <open-source-office@arm.com></text>
// SPDX-License-Identifier: Apache-2.0

#include "glsl_mappings.slang"
#include "constants.slang"

import "tonemap";
import "../../../../common/model/shaders/tensor";
import "../utils";

#ifndef COMMON
#define COMMON


/*------------------.
| :: HELPER FUNCS :: |
'------------------*/

bool IsOnScreen(int2 pos, int2 size)
{
    return all(lessThan(pos, size)) &&
           all(greaterThanEqual(pos, int2(0)));
}

float NormalizeDepth(float depth)
{
    return 1.0 - (log(-(DEPTH_LOG_K * (depth - 1.0)) + 1) /
                  log(DEPTH_LOG_K + 1));
}

float2 CalculateCameraMotion(float2 uv, float depth, float4x4 MotionMat, out float mask)
{
    // Calculate "camera motion vectors" based on camera motion via ViewProjection mat. Fails for animated objects
    // out mask used to filter out pixels where the calculation fails.
    float2 uvT = float2(uv.y, 1.0-uv.x); // CHANGE: Swizzle
    float4 clip = float4(2.0f * uvT - 1.0f, depth, 1.0f);
    float4 reproj = MatrixVectorMultiplication(MotionMat, clip);
    mask = float(reproj.w < 0);
    float2 uvTm2 = ((reproj.xy / reproj.w) + 1.0f) * 0.5f;

    float2 velocity = uvT - uvTm2;

    // Remove outliers
    float basically_0_motion = ZERO_MOTION_EPS;
    velocity = all(abs(velocity) < float2(basically_0_motion)) ? float2(0.0f) : velocity;
    velocity = any(isnan(velocity)) ? float2(0.0) : velocity;
    velocity = any(isinf(velocity)) ? float2(0.0) : velocity;

    // CHANGE: swizzle back
    return -float2(-float(velocity.y), float(velocity.x));
}

/*--------------.
| :: LOADERS :: |
'--------------*/


[Differentiable]
float3 LoadColour(uint b, uint2 pixel, TensorView<float> texture)
{
    float3 colour = sample_tensor_nearest(texture, b, uint3(0, 1, 2), pixel);
#if LINEAR_COLOUR
    colour = TonemapExp(colour, LINEAR_COLOUR_TONEMAP_EXPOSURE);
#endif
    return colour;
}

[Differentiable]
float3 LoadColour(uint b, float2 uv, TensorView<float> texture)
{
    float3 colour = sample_tensor(texture, b, uint3(0, 1, 2), uv);
#if LINEAR_COLOUR
    colour = TonemapExp(colour, LINEAR_COLOUR_TONEMAP_EXPOSURE);
#endif
    return colour;
}


[Differentiable]
float2 LoadMotion(uint b, int2 pixel, TensorView<float> texture)
{
    float2 motion = sample_tensor_nearest(texture, b, uint2(0, 1), pixel);
#if REVERSE_MV
    motion *= -1;
#endif
    return motion;
}

[Differentiable]
float LoadDepth(uint b, int2 pixel, TensorView<float> texture)
{
    float depth = sample_tensor_nearest(texture, b, uint(0), pixel);
    // TODO: fix this hard-coded size
#if INVERTED_DEPTH
    int2 size = int2(INVERTED_DEPTH_WIDTH, INVERTED_DEPTH_HEIGHT);
    depth = 1.0 - depth;
    depth = !IsOnScreen(pixel, size) ? 0.0 : depth;
#endif
    return depth;
}


float2 sym_ceil(float2 x)
{
    return sign(x) * ceil(abs(x));
}

float2 quantise(float2 x) {
    const int32_t full_bits = BITS_X + BITS_Y;
    const int32_t max_int = (1 << full_bits) - 1;
    const float c = float(max_int) / MAX_VAL;
    const int32_t2 q = int32_t2(clamp(sym_ceil(x * c), -MAX_VAL, MAX_VAL));
    return float2(q) / c;
}

float dynamic_mask_v5(float2 a, float2 b, float eps, float tau) {
    a = quantise(a);
    b = quantise(b);
    float diff = length(a - b);
    float max_len = max(max(length(a), length(b)), tau);
    return max_len < tau ? 0.0f : float(diff >= eps);
}

float dynamic_mask_v4(float2 a, float2 b, float eps, float tau)
{
    float diff = length(a - b);
    float denom = max(max(length(a), length(b)), tau);
    return float((diff / denom) >= eps);
}


#endif // COMMON
