// SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
// its affiliates <open-source-office@arm.com></text>
// SPDX-License-Identifier: Apache-2.0

#include "constants.slang"
#include "glsl_mappings.slang"

import "common";
import "../../../../common/model/shaders/tensor";
import "fill_holes_common";

// ─────────────────────────────────────────────────────────────────────────────
// Argument plumbing
// ─────────────────────────────────────────────────────────────────────────────
#define FILL_MV_PARAMS(X)                 \
    X(TensorView<int>,   in_packed_mv),   \
    X(TensorView<float>,   out_motion)

#define DECL(type, name) type name
#define NAME_ONLY(_, name) name
#define FILL_MV_ARGS FILL_MV_PARAMS(NAME_ONLY)


struct Bindings {
    // inputs
    TensorView<int> _WarpedDataTexture;
    // outputs
    TensorView<float> _OutputTexture;
};


struct PushConsts {
    int32_t2 _OutputDims;
    float2   _InvOutputDims;
};


struct Globals {
    Bindings   b;
    PushConsts pc;
};


[Differentiable]
Globals makeGlobals(FILL_MV_PARAMS(DECL), uint batch)
{
    Globals g;

    // resources
    g.b._WarpedDataTexture = in_packed_mv;
    g.b._OutputTexture     = out_motion;

    // tensor sizes
    const int2 out_dims   = int2(tensor_size(out_motion).zw);

    // push constants
    g.pc._OutputDims        = out_dims;
    g.pc._InvOutputDims     = rcp(float2(out_dims));

    return g;
}


// ─────────────────────────────────────────────────────────────────────────────
// Kernels
// ─────────────────────────────────────────────────────────────────────────────
[AutoPyBindCUDA]
[CUDAKernel]
public void fill_mv(FILL_MV_PARAMS(DECL))
{
    uint tId = (cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim()).x;
    uint3 size = tensor_size(out_motion).xzw;
    uint3 idx = tensor_idx(tId, size);

    if (any(idx >= size)) return;

    uint batch = idx.x;
    Globals g = makeGlobals(FILL_MV_ARGS, batch);
    int32_t2 output_pixel = int32_t2(idx.yz);
    if (any(greaterThanEqual(output_pixel, g.pc._OutputDims))) return;

    FillHoles(batch, output_pixel, g.pc._OutputDims, g.pc._InvOutputDims, g.b._WarpedDataTexture, g.b._OutputTexture);
}
