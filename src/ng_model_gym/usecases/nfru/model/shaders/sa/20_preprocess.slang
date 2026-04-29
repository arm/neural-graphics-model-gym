// SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
// its affiliates <open-source-office@arm.com></text>
// SPDX-License-Identifier: Apache-2.0

#include "common.slang"
#include "glsl_mappings.slang"

import "common";
import "../../../../common/model/shaders/tensor";
import "../utils";

// ─────────────────────────────────────────────────────────────────────────────
// Argument plumbing
// ─────────────────────────────────────────────────────────────────────────────
#define PREPROCESS_PARAMS(X)                     \
    X(TensorView<float>, in_flow_t_f30_xx),           \
    X(TensorView<float>, in_mv_t_f30_m1),        \
    X(TensorView<float>, in_rgb_m1),          \
    X(TensorView<float>, in_rgb_p1),    \
    X(TensorView<float>, in_depth_m1), \
    X(TensorView<float>, in_depth_p1), \
    X(TensorView<float>, in_depth_p1_warp_t), \
    X(TensorView<float>, in_depth_p1_warp_p1), \
    X(TensorView<float>, in_motion_mat_m1p1), \
    X(TensorView<float>, in_motion_mat_p1m1), \
    X(TensorView<float>, in_depth_params), \
    X(no_diff float,     in_timestep),        \
    X(no_diff int,       in_random_seed),        \
    X(DiffTensorView,    network_in)

#define DECL(type, name) type name
#define NAME_ONLY(_, name) name
#define PREPROCESS_ARGS PREPROCESS_PARAMS(NAME_ONLY)


struct Bindings {
    // inputs
    TensorView<float> _WarpedFlowTm1;
    TensorView<float> _WarpedMotionTp1;
    TensorView<float> _ColourTm1;
    TensorView<float> _ColourTp1;
    TensorView<float> _DepthTm1;
    TensorView<float> _DepthTp1;
    TensorView<float> _HolesT;
    TensorView<float> _HolesTm1;

    // outputs
    DiffTensorView _InputTensor;
};


struct PushConsts {
    float4x4 _MotionMatM1P1;
    float4x4 _MotionMatP1M1;
    float4 _DeviceToViewDepth;
    int32_t2 _OutputDims;
    float2   _InvOutputDims;
    float    _Timestep;
    uint    _RandomSeed;
};


struct Globals {
    Bindings   b;
    PushConsts pc;
};


[Differentiable]
Globals makeGlobals(PREPROCESS_PARAMS(DECL), uint batch)
{
    Globals g;

    // resources
    g.b._WarpedFlowTm1 = in_flow_t_f30_xx;
    g.b._WarpedMotionTp1 = in_mv_t_f30_m1;
    g.b._ColourTm1 = in_rgb_m1;
    g.b._ColourTp1 = in_rgb_p1;
    g.b._DepthTm1 = in_depth_m1;
    g.b._DepthTp1 = in_depth_p1;
    g.b._HolesT = in_depth_p1_warp_t;
    g.b._HolesTm1 = in_depth_p1_warp_p1;
    g.b._InputTensor = network_in;

    // tensor sizes
    const int2 out_dims   = int2(tensor_size(network_in).zw);
    const float4x4 motion_mat_m1p1 = tensor_to_float4x4(in_motion_mat_m1p1, batch);
    const float4x4 motion_mat_p1m1 = tensor_to_float4x4(in_motion_mat_p1m1, batch);

    float4 device_to_view = float4(
        no_diff(in_depth_params[uint4(batch, 0, 0, 0)]),
        no_diff(in_depth_params[uint4(batch, 1, 0, 0)]),
        no_diff(in_depth_params[uint4(batch, 2, 0, 0)]),
        no_diff(in_depth_params[uint4(batch, 3, 0, 0)]),
    );

    // push constants
    g.pc._MotionMatM1P1     = motion_mat_m1p1;
    g.pc._MotionMatP1M1     = motion_mat_p1m1;
    g.pc._DeviceToViewDepth = device_to_view;
    g.pc._OutputDims        = out_dims;
    g.pc._InvOutputDims     = rcp(float2(out_dims));
    g.pc._Timestep          = in_timestep;
    g.pc._RandomSeed        = in_random_seed;

    return g;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

float2 DepthSize(Globals g)
{
    return float2(textureSize(g.b._DepthTm1, 0));
}

float2 ComputeNdc(float2 fPxPos, int32_t2 iSize)
{
    /*
        Closely based on:
        https://github.com/arm/accuracy-super-resolution-generic-library/blob/
        38697a58a6e7818ec9d28774bc073f537abb9178/include/gpu/fsr2/ffxm_fsr2_common.h#L457
    */

    // CHANGE: removed swizzle
    return fPxPos / float2(iSize) * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f);
}


float GetViewSpaceDepth(Globals g, float fDeviceDepth)
{
    /*
        Closely based on:
        https://github.com/arm/accuracy-super-resolution-generic-library/blob/
        38697a58a6e7818ec9d28774bc073f537abb9178/include/gpu/fsr2/ffxm_fsr2_common.h#L462

        `fDeviceToViewDepth` / `_DeviceToViewDepth` details found in:
        https://github.com/GPUOpen-Effects/FidelityFX-FSR2/blob/
        1680d1edd5c034f88ebbbb793d8b88f8842cf804/src/ffx-fsr2-api/ffx_fsr2.cpp#L663
    */

    const float4 fDeviceToViewDepth = g.pc._DeviceToViewDepth;

    return (fDeviceToViewDepth[1] / (fDeviceDepth - fDeviceToViewDepth[0]));
}


float3 GetViewSpacePosition(Globals g, int32_t2 iViewportPos, int32_t2 iViewportSize, float fDeviceDepth)
{
    /*
        Closely based on:
        https://github.com/arm/accuracy-super-resolution-generic-library/blob/
        38697a58a6e7818ec9d28774bc073f537abb9178/include/gpu/fsr2/ffxm_fsr2_common.h#L475
    */

    const float4 fDeviceToViewDepth = g.pc._DeviceToViewDepth;

    const float Z = GetViewSpaceDepth(g, fDeviceDepth);

    const float2 fNdcPos = ComputeNdc(iViewportPos, iViewportSize);
    const float X = fDeviceToViewDepth[2] * fNdcPos.x * Z;
    const float Y = fDeviceToViewDepth[3] * fNdcPos.y * Z;

    return float3(X, Y, Z);
}

float SingleTapDepthClip(Globals g, float current_depth, float previous_depth){

    /*
        Closely based on:
        https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/f4c1da8e92f3fe563b5c28c44e6267ce6b6b8eb2/
        Kits/FidelityFX/framegeneration/fsr3/include/gpu/frameinterpolation/ffx_frameinterpolation_disocclusion_mask.h#L65
    */

    const float2 render_size = DepthSize(g);
    const float fCurrentDepthViewSpace = GetViewSpaceDepth(g, current_depth);

    const float fPrevDepthSample = previous_depth;
    const float fPrevNearestDepthViewSpace = GetViewSpaceDepth(g, fPrevDepthSample);

    const float fDepthDiff = fCurrentDepthViewSpace - fPrevNearestDepthViewSpace;
    if (fDepthDiff <= 0) return 0;

    const float fPlaneDepth = max(fPrevDepthSample, current_depth);

    const float3 fCenter = GetViewSpacePosition(g, int32_t2(render_size * 0.5f), int32_t2(render_size), fPlaneDepth);
    const float3 fCorner = GetViewSpacePosition(g, int32_t2(0, 0), int32_t2(render_size), fPlaneDepth);

    const float fHalfViewportWidth = length(float2(render_size));
    const float fDepthThreshold = max(fCurrentDepthViewSpace, fPrevNearestDepthViewSpace);

    const float Ksep = DISOCCLUSION_DEPTH_SEPARATION_SCALE;
    const float Kfov = length(fCorner) / length(fCenter);
    const float fRequiredDepthSeparation = Ksep * Kfov * fHalfViewportWidth * fDepthThreshold;

    const float2 fReferenceResolution = float2(
        DISOCCLUSION_REFERENCE_HEIGHT, DISOCCLUSION_REFERENCE_WIDTH
    );
    const float fResolutionFactor = saturate(length(float2(render_size)) / length(fReferenceResolution));
    const float fPower = lerp(1.0f, 3.0f, fResolutionFactor);
    const float fDepth = pow(saturate(float(fRequiredDepthSeparation / fDepthDiff)), fPower) ;

    return 1.0 - fDepth;
}

// ─────────────────────────────────────────────────────────────────────────────
// WARNING: uv conventions are different in slang. Therefore, this section
// contains changes to the implementation vs the original GLSL code.
// ─────────────────────────────────────────────────────────────────────────────

// Transform tp1 depth values to tm1 frame's screen space
float reproject_depth_fwd(Globals g, float2 uv, float depth)
{
    float4x4 motionMat = g.pc._MotionMatP1M1;
    float2 uvT = float2(uv.y, 1.0-uv.x); // CHANGE: Swizzle
    float4 clip = float4(2.0f * uvT - 1.0f, depth, 1.0f);
    float4 reproj = MatrixVectorMultiplication(motionMat, clip);
    return 1.0f - (reproj.z / reproj.w);
}

// Transform tm1 depth values to tp1 frame's screen space
float reproject_depth_bwd(Globals g, float2 uv, float depth)
{
    float4x4 motionMat = g.pc._MotionMatM1P1;
    float2 uvT = float2(uv.y, 1.0-uv.x); // CHANGE: Swizzle
    float4 clip = float4(2.0f * uvT - 1.0f, depth, 1.0f);
    float4 reproj = MatrixVectorMultiplication(motionMat, clip);
    return 1.0f - (reproj.z / reproj.w);
}
// ─────────────────────────────────────────────────────────────────────────────

float2 SingleTapCalculateDisocclusionMask(
    Globals g,
    float depth_m1_warp_t_mv,
    float depth_p1_warp_t_mv,
    float2 uv_m1_mv,
    float2 uv_p1_mv
) {
    // Reproject/transform depth to respective opposite depth screen space
    float depth_m1_warp_t_mv_transform_p1 = reproject_depth_bwd(g, uv_p1_mv, 1. - depth_m1_warp_t_mv);
    float depth_p1_warp_t_mv_transform_m1 = reproject_depth_fwd(g, uv_m1_mv, 1. - depth_p1_warp_t_mv);

    float dis_mask_m1 = SingleTapDepthClip(
        g,
        depth_p1_warp_t_mv,
        depth_m1_warp_t_mv_transform_p1
    );
    float dis_mask_p1 = SingleTapDepthClip(
        g,
        depth_m1_warp_t_mv,
        depth_p1_warp_t_mv_transform_m1
    );

    return float2(dis_mask_m1, dis_mask_p1);
}

uint32_t hash32_fast(uint32_t v)
{
    v = (v ^ 61u) ^ (v >> 16);
    v *= 9u;
    v ^= v >> 4;
    v *= 0x27d4eb2du;
    v ^= v >> 15;
    return v;
}

[NoDiffThis]
float hash_to_float(Globals g, uint32_t x, uint32_t y)
{
    uint32_t v = x + 0x9e3779b9u * y + g.pc._RandomSeed;
    uint32_t h = hash32_fast(v);
    return float(asfloat((h >> 9) | 0x3f800000u) - 1.0f);
}


// ─────────────────────────────────────────────────────────────────────────────
// Kernels
// ─────────────────────────────────────────────────────────────────────────────
[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
public void preprocess(PREPROCESS_PARAMS(DECL))
{
    uint tId = (cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim()).x;
    uint3 size = tensor_size(network_in).xzw;
    uint3 idx = tensor_idx(tId, size);

    if (any(idx >= size)) return;

    int32_t2 coord = int32_t2(idx.yz);

    uint batch = idx.x;
    Globals g = makeGlobals(PREPROCESS_ARGS, batch);

    float2 uv = (coord + 0.5f) * g.pc._InvOutputDims;

    int32_t2 iMvCoord = int32_t2(floor(uv * float2(textureSize(g.b._WarpedMotionTp1, 0))));      
    int32_t2 iFlowCoord = int32_t2(floor(uv * float2(textureSize(g.b._WarpedFlowTm1, 0))));  

    // Read Vectors warped to "t" position - sampling nearest
    float2 flow_t_f30_p1 = LoadMotion(batch, iFlowCoord, g.b._WarpedFlowTm1);
    float2 mv_t_f30_m1 = LoadMotion(batch, iMvCoord, g.b._WarpedMotionTp1);

    // Calculate warped game mv positions
    float2 uv_m1_mv = uv + float2(mv_t_f30_m1 * g.pc._Timestep);
    float2 uv_p1_mv = uv - float2(mv_t_f30_m1 * (1.0f-g.pc._Timestep));

    // Calculate warped optical flow positions
    float2 uv_m1_flow = uv - float2(flow_t_f30_p1 * g.pc._Timestep);
    float2 uv_p1_flow = uv + float2(flow_t_f30_p1 * (1.0f-g.pc._Timestep));

    // Calculate OOB areas
    bool uv_p1_mv_oob = (uv_p1_mv.x <= 0.0 || uv_p1_mv.x >= 1.0) || (uv_p1_mv.y <= 0.0 || uv_p1_mv.y >= 1.0);
    bool uv_m1_mv_oob = (uv_m1_mv.x <= 0.0 || uv_m1_mv.x >= 1.0) || (uv_m1_mv.y <= 0.0 || uv_m1_mv.y >= 1.0);
    bool uv_m1_flow_oob = (uv_m1_flow.x <= 0.0 || uv_m1_flow.x >= 1.0) || (uv_m1_flow.y <= 0.0 || uv_m1_flow.y >= 1.0);
    bool uv_p1_flow_oob = (uv_p1_flow.x <= 0.0 || uv_p1_flow.x >= 1.0) || (uv_p1_flow.y <= 0.0 || uv_p1_flow.y >= 1.0);

    // Calculate double dissoclusion mask
    float depth_p1_warp_m1_warp_t = sample_tensor(g.b._HolesTm1, batch, uint(0), uv_m1_mv).r;
    float hole_mask_m1 = float(depth_p1_warp_m1_warp_t != 0.0f)  * (1.0-float(uv_p1_mv_oob));

    float depth_p1_warp_t = sample_tensor_nearest(g.b._HolesT, batch, uint(0), iMvCoord).r;
    float hole_mask_t = depth_p1_warp_t * (1.0-float(uv_p1_mv_oob));

    float dd_mask = float((hole_mask_t - hole_mask_m1) < 0.0f);

    // Warp colour using gather approach with warped vectors
    float2 external_size = float2(textureSize(g.b._ColourTm1, 0));

    float3 rgb_m1_warp_t_mv = float3(0.0f);
    float3 rgb_p1_warp_t_mv = float3(0.0f);
    float3 rgb_m1_warp_t_flow = float3(0.0f);
    float3 rgb_p1_warp_t_flow = float3(0.0f);

    if (!uv_m1_mv_oob) rgb_m1_warp_t_mv = LoadColour(batch, int32_t2(floor(uv_m1_mv * external_size)), g.b._ColourTm1);
    if (!uv_p1_mv_oob) rgb_p1_warp_t_mv = LoadColour(batch, int32_t2(floor(uv_p1_mv * external_size)), g.b._ColourTp1);
    if (!uv_m1_flow_oob) rgb_m1_warp_t_flow = LoadColour(batch, int32_t2(floor(uv_m1_flow * external_size)), g.b._ColourTm1);
    if (!uv_p1_flow_oob) rgb_p1_warp_t_flow = LoadColour(batch, int32_t2(floor(uv_p1_flow * external_size)), g.b._ColourTp1);

    // Randomize OOB RGB values to prevent network from learning spurious correlations
    // Uses spatial hash (y*stride+z) combined with random_seed for temporal variation
    uint32_t seed_base = uint32_t(coord.x) * RANDOM_OOB_SEED_STRIDE + uint32_t(coord.y);

#if !(DISABLE_RANDOM_OOB)
    if (uv_m1_mv_oob) {
        rgb_m1_warp_t_mv = float3(
            no_diff(hash_to_float(g, seed_base, NETWORK_INPUT_RGB_M1_MV_BASE)),
            no_diff(hash_to_float(g, seed_base, NETWORK_INPUT_RGB_M1_MV_BASE + 1)),
            no_diff(hash_to_float(g, seed_base, NETWORK_INPUT_RGB_M1_MV_BASE + 2))
        );
    }
    if (uv_p1_mv_oob) {
        rgb_p1_warp_t_mv = float3(
            no_diff(hash_to_float(g, seed_base, NETWORK_INPUT_RGB_P1_MV_BASE)),
            no_diff(hash_to_float(g, seed_base, NETWORK_INPUT_RGB_P1_MV_BASE + 1)),
            no_diff(hash_to_float(g, seed_base, NETWORK_INPUT_RGB_P1_MV_BASE + 2))
        );
    }
    if (uv_m1_flow_oob) {
        rgb_m1_warp_t_flow = float3(
            no_diff(hash_to_float(g, seed_base, NETWORK_INPUT_RGB_M1_FLOW_BASE)),
            no_diff(hash_to_float(g, seed_base, NETWORK_INPUT_RGB_M1_FLOW_BASE + 1)),
            no_diff(hash_to_float(g, seed_base, NETWORK_INPUT_RGB_M1_FLOW_BASE + 2))
        );
    }
    if (uv_p1_flow_oob) {
        rgb_p1_warp_t_flow = float3(
            no_diff(hash_to_float(g, seed_base, NETWORK_INPUT_RGB_P1_FLOW_BASE)),
            no_diff(hash_to_float(g, seed_base, NETWORK_INPUT_RGB_P1_FLOW_BASE + 1)),
            no_diff(hash_to_float(g, seed_base, NETWORK_INPUT_RGB_P1_FLOW_BASE + 2))
        );
    }
#endif

    // Warp depth
    float2 depth_size = DepthSize(g);
    float depth_m1_warp_t_mv = LoadDepth(batch, int32_t2(floor(uv_m1_mv * depth_size)), g.b._DepthTm1);
    float depth_p1_warp_t_mv = LoadDepth(batch, int32_t2(floor(uv_p1_mv * depth_size)), g.b._DepthTp1);

    // Normalize Depth
    float depth_m1_warp_t_mv_norm = no_diff(NormalizeDepth(depth_m1_warp_t_mv));
    float depth_p1_warp_t_mv_norm = no_diff(NormalizeDepth(depth_p1_warp_t_mv));
    float2 depth = float2(depth_m1_warp_t_mv_norm, depth_p1_warp_t_mv_norm);

    // Calculate disocclusion mask
    float2 dis_mask = no_diff(SingleTapCalculateDisocclusionMask(
        g,
        depth_m1_warp_t_mv,
        depth_p1_warp_t_mv,
        uv_m1_mv,
        uv_p1_mv
    ));

    // Encode double disocclusion mask as dis_mask.x = dis_mask.y = 1
    dis_mask.x = saturate(dis_mask.x + dd_mask);
    dis_mask.y = saturate(dis_mask.y + dd_mask);

    // Write Tensor
    g.b._InputTensor.storeOnce(uint4(idx.x, NETWORK_INPUT_RGB_M1_MV_BASE, idx.yz), rgb_m1_warp_t_mv.r);
    g.b._InputTensor.storeOnce(uint4(idx.x, NETWORK_INPUT_RGB_M1_MV_BASE + 1, idx.yz), rgb_m1_warp_t_mv.g);
    g.b._InputTensor.storeOnce(uint4(idx.x, NETWORK_INPUT_RGB_M1_MV_BASE + 2, idx.yz), rgb_m1_warp_t_mv.b);

    g.b._InputTensor.storeOnce(uint4(idx.x, NETWORK_INPUT_RGB_P1_MV_BASE, idx.yz), rgb_p1_warp_t_mv.r);
    g.b._InputTensor.storeOnce(uint4(idx.x, NETWORK_INPUT_RGB_P1_MV_BASE + 1, idx.yz), rgb_p1_warp_t_mv.g);
    g.b._InputTensor.storeOnce(uint4(idx.x, NETWORK_INPUT_RGB_P1_MV_BASE + 2, idx.yz), rgb_p1_warp_t_mv.b);

    g.b._InputTensor.storeOnce(uint4(idx.x, NETWORK_INPUT_RGB_M1_FLOW_BASE, idx.yz), rgb_m1_warp_t_flow.r);
    g.b._InputTensor.storeOnce(uint4(idx.x, NETWORK_INPUT_RGB_M1_FLOW_BASE + 1, idx.yz), rgb_m1_warp_t_flow.g);
    g.b._InputTensor.storeOnce(uint4(idx.x, NETWORK_INPUT_RGB_M1_FLOW_BASE + 2, idx.yz), rgb_m1_warp_t_flow.b);

    g.b._InputTensor.storeOnce(uint4(idx.x, NETWORK_INPUT_RGB_P1_FLOW_BASE, idx.yz), rgb_p1_warp_t_flow.r);
    g.b._InputTensor.storeOnce(uint4(idx.x, NETWORK_INPUT_RGB_P1_FLOW_BASE + 1, idx.yz), rgb_p1_warp_t_flow.g);
    g.b._InputTensor.storeOnce(uint4(idx.x, NETWORK_INPUT_RGB_P1_FLOW_BASE + 2, idx.yz), rgb_p1_warp_t_flow.b);

    g.b._InputTensor.storeOnce(uint4(idx.x, NETWORK_INPUT_DEPTH_BASE, idx.yz), depth_m1_warp_t_mv_norm);
    g.b._InputTensor.storeOnce(uint4(idx.x, NETWORK_INPUT_DEPTH_BASE + 1, idx.yz), depth_p1_warp_t_mv_norm);

    g.b._InputTensor.storeOnce(uint4(idx.x, NETWORK_INPUT_DISOCCLUSION_BASE, idx.yz), dis_mask.x);
    g.b._InputTensor.storeOnce(uint4(idx.x, NETWORK_INPUT_DISOCCLUSION_BASE + 1, idx.yz), dis_mask.y);
}
