// SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
// its affiliates <open-source-office@arm.com></text>
// SPDX-License-Identifier: Apache-2.0

#include "constants.slang"

import "common";
import "quant";

#ifndef WARP_FLOW_COMMON
#define WARP_FLOW_COMMON


float InvertAndNormalizeDepth(float depth, int32_t output_pixel_cont, float inv_output_area)
{
    // Normalize depth with log approach
    depth = NormalizeDepth(depth);
    // Slight hack to prevent scatter 'collisions' at far plane (where all depths are the same)
    depth = depth == 1.0
        ? depth + (output_pixel_cont * inv_output_area * FAR_PLANE_COLLISION_BIAS)
        : depth;
    // Invert - this is needed to use AtomicMax (intermediate buffers are initialised to 0s)
    return DEPTH_MAX - depth;
}

void Warp(uint batch, TensorView<int> _OutputTexture, int32_t2 dtId, float2 motion, float depth, int32_t2 output_dims, float t, float s)
{
    // Scatter depth and motion vector

    // Calculate target forward warp uv location
    float2 vector = float2(motion) * float2(output_dims);

    // Calculate target forward warp texel location
    // Scale by intermediate timestep
    int32_t2 coord = dtId + int32_t2(floor(vector * t));

    if (!IsOnScreen(coord, output_dims)) return;

    // Pack data for scattering. This packing has the layout:
    // (MSB->LSB): depth | exponent | signX | signY | mantissaX | mantissaY
    uint32_t qdata = qFloat(s * vector, depth);

    int old_qdata;
    _OutputTexture.InterlockedMax(uint4(batch, 0, coord), qdata, old_qdata);
}

#endif // WARP_FLOW_COMMON
