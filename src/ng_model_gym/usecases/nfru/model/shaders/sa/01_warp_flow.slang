// SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
// its affiliates <open-source-office@arm.com></text>
// SPDX-License-Identifier: Apache-2.0

#include "constants.slang"
#include "glsl_mappings.slang"

import "common";
import "../../../../common/model/shaders/tensor";
import "warp_flow_common";

// ─────────────────────────────────────────────────────────────────────────────
// Argument plumbing
// ─────────────────────────────────────────────────────────────────────────────
#define WARP_FLOW_PARAMS(X)                     \
    X(TensorView<float>, in_depth),           \
    X(TensorView<float>, in_motion),          \
    X(no_diff float,     in_timestep),        \
    X(TensorView<int>,   out_packed_mv)

#define DECL(type, name) type name
#define NAME_ONLY(_, name) name
#define WARP_FLOW_ARGS WARP_FLOW_PARAMS(NAME_ONLY)


struct Bindings {
    // inputs
    TensorView<float> _DepthTexture;
    TensorView<float> _DepthTextureM1;
    TensorView<float> _VectorTexture;
    TensorView<float> _DynamicMask;

    // outputs
    TensorView<int>    _OutputTexture;
    TensorView<float>  _OutputDynamicMask;
    TensorView<float>  _OutputHolesT;
    TensorView<float>  _OutputHolesTm1;
};


struct PushConsts {
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
Globals makeGlobals(WARP_FLOW_PARAMS(DECL), uint batch)
{
    Globals g;

    // resources
    g.b._DepthTexture      = in_depth;
    g.b._VectorTexture     = in_motion;
    g.b._OutputTexture     = out_packed_mv;

    // tensor sizes
    const int2 out_dims   = int2(tensor_size(out_packed_mv).zw);

    // push constants
    g.pc._OutputDims        = out_dims;
    g.pc._InvOutputDims     = rcp(float2(out_dims));
    g.pc._InvOutputArea     = rcp(float(out_dims.x * out_dims.y));
    g.pc._Timestep          = in_timestep;

    return g;
}


// ─────────────────────────────────────────────────────────────────────────────
// Kernels
// ─────────────────────────────────────────────────────────────────────────────
[AutoPyBindCUDA]
[CUDAKernel]
public void warp_flow(WARP_FLOW_PARAMS(DECL))
{
    uint tId = (cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim()).x;
    uint3 size = tensor_size(out_packed_mv).xzw;
    uint3 idx = tensor_idx(tId, size);

    if (any(idx >= size)) return;

    uint batch = idx.x;
    Globals g = makeGlobals(WARP_FLOW_ARGS, batch);

    int32_t2 output_pixel = int32_t2(idx.yz);
    int32_t output_pixel_cont = output_pixel.x + output_pixel.y * g.pc._OutputDims.x;

    float2 uv = (float2(output_pixel.xy) + 0.5f) * g.pc._InvOutputDims;

    int32_t2 iDepthCoord = int32_t2(floor(uv * float2(textureSize(g.b._DepthTexture, 0))));
    int32_t2 iFlowCoord = int32_t2(floor(uv * float2(textureSize(g.b._VectorTexture, 0))));

    // Load depth, normalise - by default, we assume 0 is closer to the camera
    float depth_m1 = LoadDepth(batch, iDepthCoord, g.b._DepthTexture);
    depth_m1 = InvertAndNormalizeDepth(depth_m1, output_pixel_cont, g.pc._InvOutputArea);

    // Load optical flow
    float2 flow_m1_f30_p1 = LoadMotion(batch, iFlowCoord, g.b._VectorTexture);

    // Perform depth-aware forward warp
    Warp(batch, g.b._OutputTexture, output_pixel, flow_m1_f30_p1, depth_m1, g.pc._OutputDims, g.pc._Timestep, 1.0HF);
}
