// SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
// its affiliates <open-source-office@arm.com></text>
// SPDX-License-Identifier: Apache-2.0

#include "common.slang"
#include "glsl_mappings.slang"

import "common";
import "../../../../common/model/shaders/tensor";

// ─────────────────────────────────────────────────────────────────────────────
// Argument plumbing
// ─────────────────────────────────────────────────────────────────────────────
#define POSTPROCESS_PARAMS(X)               \
    X(TensorView<float>, in_flow_t_f30_xx), \
    X(TensorView<float>, in_mv_t_f30_m1),   \
    X(TensorView<float>, in_rgb_m1),        \
    X(TensorView<float>, in_rgb_p1),        \
    X(DiffTensorView,    in_params),        \
    X(no_diff float,     in_timestep),      \
    X(DiffTensorView,    out_colour)       

#define DECL(type, name) type name
#define NAME_ONLY(_, name) name
#define POSTPROCESS_ARGS POSTPROCESS_PARAMS(NAME_ONLY)


struct Bindings {
    // inputs
    TensorView<float> _WarpedFlowTm1;
    TensorView<float> _WarpedMotionTp1;
    TensorView<float> _ColourTm1;
    TensorView<float> _ColourTp1;
    DiffTensorView    _ParamsTex;

    // outputs
    DiffTensorView _Output;
};


struct PushConsts {
    int32_t2 _OutputDims;
    float2   _InvOutputDims;
    float    _Timestep;
};


struct Globals {
    Bindings   b;
    PushConsts pc;
};


[Differentiable]
Globals makeGlobals(POSTPROCESS_PARAMS(DECL), uint batch)
{
    Globals g;

    // resources
    g.b._WarpedFlowTm1 = in_flow_t_f30_xx;
    g.b._WarpedMotionTp1 = in_mv_t_f30_m1;
    g.b._ColourTm1 = in_rgb_m1;
    g.b._ColourTp1 = in_rgb_p1;
    g.b._ParamsTex = in_params;
    g.b._Output = out_colour;

    // tensor sizes
    const int2 out_dims   = int2(tensor_size(out_colour).zw);

    // push constants
    g.pc._OutputDims        = out_dims;
    g.pc._InvOutputDims     = rcp(float2(out_dims));
    g.pc._Timestep          = in_timestep;

    return g;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

[Differentiable]
float max4(float4 x)
{
    return max(max(max(x.x, x.y), x.z), x.w);
}


[Differentiable]
float4 Stable_Softmax(float4 tensor)
{
    // Subtract the max value to help reduce overflow issues
    float4 stable_ten = tensor - max4(tensor);
    float4 e_ten = exp(stable_ten);
    float sum = dot(e_ten, float4(1.0));
    float4 smax_activated = e_ten / (sum + FLT_MIN);
    return smax_activated;
}


// ─────────────────────────────────────────────────────────────────────────────
// Kernels
// ─────────────────────────────────────────────────────────────────────────────
[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
public void postprocess(POSTPROCESS_PARAMS(DECL))
{
    uint tId = (cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim()).x;
    uint3 size = tensor_size(out_colour).xzw;
    uint3 idx = tensor_idx(tId, size);

    if (any(idx >= size)) return;

    int32_t2 output_pixel = int32_t2(idx.yz);

    uint batch = idx.x;
    Globals g = makeGlobals(POSTPROCESS_ARGS, batch);

    float2 uv = (float2(output_pixel.xy) + 0.5f) * g.pc._InvOutputDims;

    int32_t2 iMvCoord = int32_t2(floor(uv * float2(textureSize(g.b._WarpedMotionTp1, 0))));      
    int32_t2 iFlowCoord = int32_t2(floor(uv * float2(textureSize(g.b._WarpedFlowTm1, 0))));  

    // Read Output parameters - bilinear upsampling is done here
    float4 params = sample_tensor(g.b._ParamsTex, idx.x, uint4(0, 1, 2, 3), uv);

    // Read Vectors warped to "t" position - sampling nearest
    float2 flow_t_f30_p1 = sample_tensor_nearest(g.b._WarpedFlowTm1, idx.x, uint2(0, 1), iFlowCoord);
    float2 mv_t_f30_m1 = sample_tensor_nearest(g.b._WarpedMotionTp1, idx.x, uint2(0, 1), iMvCoord);

    // Calculate warped game mv positions
    float2 uv_m1_mv = uv + float2(mv_t_f30_m1 * g.pc._Timestep);
    float2 uv_p1_mv = uv - float2(mv_t_f30_m1 * (1.0f-g.pc._Timestep));

    // Calculate warped optical flow positions
    float2 uv_m1_flow = uv - float2(flow_t_f30_p1 * g.pc._Timestep);
    float2 uv_p1_flow = uv + float2(flow_t_f30_p1 * (1.0f-g.pc._Timestep));

    // Warp colour using gather approach with warped vectors
    float3 rgb_m1_warp_t_mv = float3(sample_tensor(g.b._ColourTm1, idx.x, uint3(0, 1, 2), uv_m1_mv).rgb);
    float3 rgb_p1_warp_t_mv = float3(sample_tensor(g.b._ColourTp1, idx.x, uint3(0, 1, 2), uv_p1_mv).rgb);
    float3 rgb_m1_warp_t_flow = float3(sample_tensor(g.b._ColourTm1, idx.x, uint3(0, 1, 2), uv_m1_flow).rgb);
    float3 rgb_p1_warp_t_flow = float3(sample_tensor(g.b._ColourTp1, idx.x, uint3(0, 1, 2), uv_p1_flow).rgb);

    // Softmax network output to get blending coeffs
    params = Stable_Softmax(params);

    // Composite Output
    float3 rgb_t = float3(0);
    rgb_t += rgb_m1_warp_t_mv * params.x;
    rgb_t += rgb_p1_warp_t_mv * params.y;
    rgb_t += rgb_m1_warp_t_flow * params.z;
    rgb_t += rgb_p1_warp_t_flow * params.w;

    // Write alpha composited output
    g.b._Output.storeOnce(uint4(idx.x, 0, output_pixel), rgb_t.r);
    g.b._Output.storeOnce(uint4(idx.x, 1, output_pixel), rgb_t.g);
    g.b._Output.storeOnce(uint4(idx.x, 2, output_pixel), rgb_t.b);
}
