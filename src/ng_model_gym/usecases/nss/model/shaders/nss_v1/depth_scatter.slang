// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0

import "../../../../common/model/shaders/tensor";
import "nss_common";

#ifndef NSS_DEPTH_SCATTER_QUARTER_RES_INPUT
#define NSS_DEPTH_SCATTER_QUARTER_RES_INPUT 0
#endif // !NSS_DEPTH_SCATTER_QUARTER_RES_INPUT

// ─────────────────────────────────────────────────────────────────────────────
// Argument plumbing
// ─────────────────────────────────────────────────────────────────────────────
#define DEPTH_SCATTER_PARAMS(X)                 \
    X(TensorView<float>, in_motion),            \
    X(TensorView<float>, in_depth),             \
    X(TensorView<float>, in_render_size),       \
    X(TensorView<int>,  out_tensor)

#define DECL(type, name) type name
#define NAME_ONLY(_, name) name
#define DEPTH_SCATTER_ARGS DEPTH_SCATTER_PARAMS(NAME_ONLY)

struct Bindings {
    TensorView<float> _MotionTex;
    TensorView<float> _DepthTex;
    TensorView<float> _RenderSizeTex;
    TensorView<int> _OutDepth;
};

struct PushConsts {
    // Source depth/motion extent divided by output depth-scatter extent.
    float2 _Scale;
    // Inverse of `_Scale`.
    float2 _InvScale;
    // Previous-depth output dimensions.
    int32_t2 _OutputDims;
    // Render-resolution dimensions used by downstream parity paths.
    float2 _RenderDims;
    // Inverse of `_OutputDims`.
    float2 _InvOutputDims;
};

struct Globals {
    Bindings b;
    PushConsts pc;
};


[NoDiffThis]
Globals makeGlobals(DEPTH_SCATTER_PARAMS(DECL), uint batch)
{
    Globals g;
    g.b._MotionTex = in_motion;
    g.b._DepthTex = in_depth;
    g.b._RenderSizeTex = in_render_size;
    g.b._OutDepth = out_tensor;

    const int2 out_dims = int2(tensor_size(out_tensor).zw);
    const int2 motion_dims = int2(tensor_size(in_motion).zw);

    g.pc._OutputDims = out_dims;
    g.pc._InvOutputDims = rcp(float2(out_dims));
    g.pc._Scale = float2(motion_dims) / float2(out_dims);
    g.pc._InvScale = rcp(g.pc._Scale);
    g.pc._RenderDims = float2(
        g.b._RenderSizeTex[uint4(batch, 0, 0, 0)],
        g.b._RenderSizeTex[uint4(batch, 1, 0, 0)]
    );

    return g;
}


// ─────────────────────────────────────────────────────────────────────────────
// Kernel
// ─────────────────────────────────────────────────────────────────────────────
[AutoPyBindCUDA]
[CUDAKernel]
[NoDiffThis]
public void depth_scatter(DEPTH_SCATTER_PARAMS(DECL))
{
    uint tId = (cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim()).x;
    uint3 size = tensor_size(out_tensor).xzw;
    uint3 idx = tensor_idx(tId, size);
    if (any(idx >= size)) return;

    Globals g = makeGlobals(DEPTH_SCATTER_ARGS, idx.x);

    float2 uv = (float2(idx.yz) + 0.5f) * g.pc._InvOutputDims;
    const float2 render_size = float2(g.pc._RenderDims);

    float3 result = float3(0.0f);

#if NSS_DEPTH_SCATTER_QUARTER_RES_INPUT
    const int2 src_size = int2(round(float2(g.pc._OutputDims) * g.pc._Scale));
    const int2 src_base = int2(float2(idx.yz) * g.pc._Scale);
    const float2 inv_src_size = g.pc._InvOutputDims * g.pc._InvScale;

    float2 uv00 = (float2(src_base + int2(0, 0)) + 0.5f) * inv_src_size;
    float2 uv10 = (float2(src_base + int2(0, 2)) + 0.5f) * inv_src_size;
    float2 uv01 = (float2(src_base + int2(2, 0)) + 0.5f) * inv_src_size;
    float2 uv11 = (float2(src_base + int2(2, 2)) + 0.5f) * inv_src_size;

    float4 d00 = texture_gather(g.b._DepthTex, idx.x, 0, uv00);
    float4 y00 = texture_gather(g.b._MotionTex, idx.x, 0, uv00);
    float4 x00 = texture_gather(g.b._MotionTex, idx.x, 1, uv00);
    float4 d10 = texture_gather(g.b._DepthTex, idx.x, 0, uv10);
    float4 y10 = texture_gather(g.b._MotionTex, idx.x, 0, uv10);
    float4 x10 = texture_gather(g.b._MotionTex, idx.x, 1, uv10);
    float4 d01 = texture_gather(g.b._DepthTex, idx.x, 0, uv01);
    float4 y01 = texture_gather(g.b._MotionTex, idx.x, 0, uv01);
    float4 x01 = texture_gather(g.b._MotionTex, idx.x, 1, uv01);
    float4 d11 = texture_gather(g.b._DepthTex, idx.x, 0, uv11);
    float4 y11 = texture_gather(g.b._MotionTex, idx.x, 0, uv11);
    float4 x11 = texture_gather(g.b._MotionTex, idx.x, 1, uv11);

    result = float3(y00.x, x00.x, d00.x);

    #define NSS_UPDATE_NEAREST_STEP(OFF_X, OFF_Y, DEPTH, MOTION_Y, MOTION_X)               \
        {                                                                                   \
            int2 src = src_base + int2((OFF_X), (OFF_Y));                                  \
            float d = (DEPTH);                                                              \
            float take = float(all(src >= 0) && all(src < src_size)) * step(d, result.z);  \
            result = lerp(result, float3((MOTION_Y), (MOTION_X), d), take);                \
        }

    NSS_UPDATE_NEAREST_STEP(0, 1, d00.y, y00.y, x00.y);
    NSS_UPDATE_NEAREST_STEP(1, 0, d00.z, y00.z, x00.z);
    NSS_UPDATE_NEAREST_STEP(1, 1, d00.w, y00.w, x00.w);
    NSS_UPDATE_NEAREST_STEP(0, 2, d10.x, y10.x, x10.x);
    NSS_UPDATE_NEAREST_STEP(0, 3, d10.y, y10.y, x10.y);
    NSS_UPDATE_NEAREST_STEP(1, 2, d10.z, y10.z, x10.z);
    NSS_UPDATE_NEAREST_STEP(1, 3, d10.w, y10.w, x10.w);
    NSS_UPDATE_NEAREST_STEP(2, 0, d01.x, y01.x, x01.x);
    NSS_UPDATE_NEAREST_STEP(2, 1, d01.y, y01.y, x01.y);
    NSS_UPDATE_NEAREST_STEP(3, 0, d01.z, y01.z, x01.z);
    NSS_UPDATE_NEAREST_STEP(3, 1, d01.w, y01.w, x01.w);
    NSS_UPDATE_NEAREST_STEP(2, 2, d11.x, y11.x, x11.x);
    NSS_UPDATE_NEAREST_STEP(2, 3, d11.y, y11.y, x11.y);
    NSS_UPDATE_NEAREST_STEP(3, 2, d11.z, y11.z, x11.z);
    NSS_UPDATE_NEAREST_STEP(3, 3, d11.w, y11.w, x11.w);

    #undef NSS_UPDATE_NEAREST_STEP
#else
    float4 dquad = texture_gather(g.b._DepthTex, idx.x, 0, uv);
    float4 yquad = texture_gather(g.b._MotionTex, idx.x, 0, uv);
    float4 xquad = texture_gather(g.b._MotionTex, idx.x, 1, uv);

    result = float3(yquad.x, xquad.x, dquad.x);
    result = lerp(result, float3(yquad.y, xquad.y, dquad.y), step(dquad.y, result.z));
    result = lerp(result, float3(yquad.z, xquad.z, dquad.z), step(dquad.z, result.z));
    result = lerp(result, float3(yquad.w, xquad.w, dquad.w), step(dquad.w, result.z));
#endif // NSS_DEPTH_SCATTER_QUARTER_RES_INPUT

    float depth_dilated = result.z;
    float2 motion = result.xy * g.pc._InvScale;
    motion *= float(length(motion) > 0.1f);
    motion *= g.pc._InvOutputDims;

    float2 reproj_uv = uv - motion;
    ReconstructPrevDepth(idx.x, g.b._OutDepth, depth_dilated, reproj_uv, render_size);
}

#undef DEPTH_SCATTER_PARAMS
#undef DEPTH_SCATTER_ARGS
#undef DECL
