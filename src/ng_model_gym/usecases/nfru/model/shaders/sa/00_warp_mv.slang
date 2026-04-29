// SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
// its affiliates <open-source-office@arm.com></text>
// SPDX-License-Identifier: Apache-2.0

#include "glsl_mappings.slang"
#include "constants.slang"

import "common";
import "../../../../common/model/shaders/tensor";
import "warp_flow_common";

// ─────────────────────────────────────────────────────────────────────────────
// Argument plumbing
// ─────────────────────────────────────────────────────────────────────────────
#define WARP_MV_PARAMS(X)                     \
    X(TensorView<float>, in_depth),           \
    X(TensorView<float>, in_depth_m1),        \
    X(TensorView<float>, in_motion),          \
    X(TensorView<float>, in_dynamic_mask),    \
    X(TensorView<float>, in_motion_mat_m1p1), \
    X(TensorView<float>, in_motion_mat_p1m1), \
    X(no_diff float,     in_timestep),        \
    X(TensorView<int>,   out_packed_mv),      \
    X(TensorView<float>, out_dynamic_mask),   \
    X(TensorView<float>, out_holes_t),        \
    X(TensorView<float>, out_holes_tm1)       \

#define DECL(type, name) type name
#define NAME_ONLY(_, name) name
#define WARP_MV_ARGS WARP_MV_PARAMS(NAME_ONLY)


struct Bindings {
    // inputs
    TensorView<float> _DepthTexture;
    TensorView<float> _DepthTextureM1;
    TensorView<float> _VectorTexture;
    TensorView<float> _DynamicMask;

    // outputs
    TensorView<int>  _OutputTexture;
    TensorView<float>  _OutputDynamicMask;
    TensorView<float>  _OutputHolesT;
    TensorView<float>  _OutputHolesTm1;
};


struct PushConsts {
    float4x4 _MotionMatM1P1;
    float4x4 _MotionMatP1M1;
    int32_t2 _OutputDims;
    float2   _InvOutputDims;
    float    _InvOutputArea;
    float    _Timestep;
};


struct Globals {
    Bindings   b;
    PushConsts pc;
};


[Differentiable]
Globals makeGlobals(WARP_MV_PARAMS(DECL), uint batch)
{
    Globals g;

    // resources
    g.b._DepthTexture      = in_depth;
    g.b._DepthTextureM1    = in_depth_m1;
    g.b._VectorTexture     = in_motion;
    g.b._DynamicMask       = in_dynamic_mask;
    g.b._OutputTexture     = out_packed_mv;
    g.b._OutputDynamicMask = out_dynamic_mask;
    g.b._OutputHolesT      = out_holes_t;
    g.b._OutputHolesTm1    = out_holes_tm1;

    // tensor sizes
    const int2 out_dims   = int2(tensor_size(out_packed_mv).zw);
    const float4x4 motion_mat_m1p1 = tensor_to_float4x4(in_motion_mat_m1p1, batch);
    const float4x4 motion_mat_p1m1 = tensor_to_float4x4(in_motion_mat_p1m1, batch);

    // push constants
    g.pc._MotionMatM1P1     = motion_mat_m1p1;
    g.pc._MotionMatP1M1     = motion_mat_p1m1;
    g.pc._OutputDims        = out_dims;
    g.pc._InvOutputDims     = rcp(float2(out_dims));
    g.pc._InvOutputArea     = rcp(float(out_dims.x * out_dims.y));
    g.pc._Timestep          = in_timestep;

    return g;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────


void WriteDynamicMask(Globals g, uint b, int32_t2 pixel, uint8_t value)
{
    imageStore(b, g.b._OutputDynamicMask, pixel, float4(float(value), 0, 0, 1));
}


// ─────────────────────────────────────────────────────────────────────────────
// Kernels
// ─────────────────────────────────────────────────────────────────────────────

// TODO add slang only previous dynamic mask code here

[AutoPyBindCUDA]
[CUDAKernel]
public void warp_mv(WARP_MV_PARAMS(DECL))
{
    uint tId = (cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim()).x;
    uint3 size = tensor_size(out_packed_mv).xzw;
    uint3 idx = tensor_idx(tId, size);

    if (any(idx >= size)) return;

    int32_t2 output_pixel = int32_t2(idx.yz);

    uint batch = idx.x;
    Globals g = makeGlobals(WARP_MV_ARGS, batch);

    int32_t output_pixel_cont = output_pixel.x + output_pixel.y * g.pc._OutputDims.x;
    float2 uv = (float2(output_pixel.xy) + 0.5f) * g.pc._InvOutputDims;
    int32_t2 iDepthCoord = int32_t2(floor(uv * float2(textureSize(g.b._DepthTexture, 0))));
    int32_t2 iMvCoord = int32_t2(floor(uv * float2(textureSize(g.b._VectorTexture, 0))));

    // Load depth, normalise - by default, we assume 0 is closer to the camera
    float depth_p1 = LoadDepth(batch, iDepthCoord, g.b._DepthTexture);
    // Calculate camera motion vectors BEFORE normalising depth!
    float cm_p1_f30_m1_mask;
    float2 cm_p1_f30_m1 = CalculateCameraMotion(uv, 1.0 - depth_p1, g.pc._MotionMatP1M1, cm_p1_f30_m1_mask);
    depth_p1 = InvertAndNormalizeDepth(depth_p1, output_pixel_cont, g.pc._InvOutputArea);

    // Load game rendered mv
    float2 mv_p1_f30_m1 = LoadMotion(batch, iMvCoord, g.b._VectorTexture);

    // Perform depth-aware forward warp
    Warp(batch, g.b._OutputTexture, output_pixel, mv_p1_f30_m1, depth_p1, g.pc._OutputDims, 1.0f - g.pc._Timestep, 1.0f);

    /*---------------------------.
    | :: DD mask holes writes :: |
    '---------------------------*/

    const int32_t iSampleCount = HOLE_FILL_SAMPLE_COUNT;
    float offset_depth[iSampleCount];
    const int32_t2 offsets[HOLE_FILL_SAMPLE_COUNT] = HOLE_FILL_OFFSETS_INT32;

    // pull out the depth loads to allow SC to batch them
    offset_depth[0] = float(texelFetchOffset(batch, g.b._DepthTexture, output_pixel, offsets[0]).r);
    offset_depth[1] = float(texelFetchOffset(batch, g.b._DepthTexture, output_pixel, offsets[1]).r);
    offset_depth[2] = float(texelFetchOffset(batch, g.b._DepthTexture, output_pixel, offsets[2]).r);
    offset_depth[3] = float(texelFetchOffset(batch, g.b._DepthTexture, output_pixel, offsets[3]).r);
    offset_depth[4] = float(texelFetchOffset(batch, g.b._DepthTexture, output_pixel, offsets[4]).r);
    offset_depth[5] = float(texelFetchOffset(batch, g.b._DepthTexture, output_pixel, offsets[5]).r);
    offset_depth[6] = float(texelFetchOffset(batch, g.b._DepthTexture, output_pixel, offsets[6]).r);
    offset_depth[7] = float(texelFetchOffset(batch, g.b._DepthTexture, output_pixel, offsets[7]).r);
    offset_depth[8] = float(texelFetchOffset(batch, g.b._DepthTexture, output_pixel, offsets[8]).r);

    // find closest depth
    int32_t nearest_offset_index = 0;
    float nearest_depth = offset_depth[nearest_offset_index];
    [ForceUnroll]
    for (int32_t iSampleIndex = 1; iSampleIndex < iSampleCount; ++iSampleIndex) {
        float tap = offset_depth[iSampleIndex];
        if (tap < nearest_depth) {
            nearest_depth = tap;
            nearest_offset_index = iSampleIndex;
        }
    }

    float2 nearest_motion = LoadMotion(batch, iMvCoord + offsets[nearest_offset_index], g.b._VectorTexture);
    imageStore(batch, g.b._OutputHolesT, output_pixel + int32_t2(floor(nearest_motion * float2(g.pc._OutputDims) * (1.0f-g.pc._Timestep))), float4(1, 0, 0, 0));
    imageStore(batch, g.b._OutputHolesTm1, output_pixel + int32_t2(floor(nearest_motion * float2(g.pc._OutputDims))), float4(1, 0, 0, 0));

    /*-------------------------------.
    | :: Bidirectional scattering :: |
    '-------------------------------*/

    // Load depth. Don't invert here as we need it for calculating camera motion vectors
    float depth_m1 = LoadDepth(batch, iDepthCoord, g.b._DepthTextureM1);

    // Calculate camera motion vectors BEFORE normalising depth
    float cm_m1_f30_p1_mask;
    float2 cm_m1_f30_p1 = CalculateCameraMotion(uv, 1.0 - depth_m1, g.pc._MotionMatM1P1, cm_m1_f30_p1_mask);

    // Get feedback dynamic mask, and calculate dynamic for next frame
    // 1 == Invalid vector
    // 0 == Valid vector to scatter
    uint8_t dynamic_mask_m1 = uint8_t(sample_tensor_nearest(g.b._DynamicMask, batch, uint(0), output_pixel).r);
    uint8_t dynamic_mask_p1 = uint8_t(
#if NEW_DYNAMIC_MASK
        dynamic_mask_v5(cm_p1_f30_m1 * float2(g.pc._OutputDims), mv_p1_f30_m1 * float2(g.pc._OutputDims), MV_SIMILARITY_THRESHOLD, MV_SIMILARITY_NOISE_THRESHOLD) > 0.0
#else
        dynamic_mask_v4(cm_p1_f30_m1, mv_p1_f30_m1, MV_SIMILARITY_THRESHOLD, MV_SIMILARITY_NOISE_THRESHOLD) > 0.0
#endif
    );

    // Save dynamic mask to feedback for next set of frames
    WriteDynamicMask(g, batch, output_pixel, dynamic_mask_p1 + uint8_t(cm_p1_f30_m1_mask));

    // Dont scatter invalid mvs
    if (dynamic_mask_m1 + uint8_t(cm_m1_f30_p1_mask) > 0) return;

    // Normalise depth before scattering
    depth_m1 = InvertAndNormalizeDepth(depth_m1, output_pixel_cont, g.pc._InvOutputArea);

    // Perform depth-aware forward warp
    Warp(batch, g.b._OutputTexture, output_pixel, cm_m1_f30_p1, depth_m1, g.pc._OutputDims, g.pc._Timestep, -1.0f);
}
