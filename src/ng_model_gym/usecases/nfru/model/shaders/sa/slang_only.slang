// SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
// its affiliates <open-source-office@arm.com></text>
// SPDX-License-Identifier: Apache-2.0

#include "glsl_mappings.slang"
#include "constants.slang"

import "common";
import "../../../../common/model/shaders/tensor";
import "../utils";

[AutoPyBindCUDA]
[CUDAKernel]
[NoDiffThis]
void calculate_previous_dynamic_mask(
    TensorView<float> tv_depth,
    TensorView<float> tv_mv_m1_f30_m3,
    TensorView<float> tv_motion_mat_tm1,
    TensorView<float> tv_dynamic_mask_p1,
)
{
    uint tId = (cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim()).x;

    // We Dispatch across `output`'s dimensions
    uint3 size = tensor_size(tv_depth).xzw; // N C H W -> NHW -> X Z W
    int H = size.y;
    int W = size.z;
    uint3 idx = tensor_idx(tId, size); // N H W
    if (any(idx >= size)) return;

    float2 uv = float2(idx.yz + 0.5f) * rcp(float2(size.yz));

    float2 mv_m1_f30_m3 = float2(
        tv_mv_m1_f30_m3[uint4(idx.x, 0, idx.yz)],
        tv_mv_m1_f30_m3[uint4(idx.x, 1, idx.yz)]
    );

    float depth_m1 = tv_depth[uint4(idx.x, 0, idx.yz)];
    float4x4 motion_mat_tm1 = tensor_to_float4x4(tv_motion_mat_tm1, idx.x);
    float mask;
    float2 cm_m1_f30_m3 = CalculateCameraMotion(uv, 1.0f-depth_m1, motion_mat_tm1, mask);
    float dynamic_mask_m1 = float(dynamic_mask_v4(cm_m1_f30_m3, mv_m1_f30_m3, MV_SIMILARITY_THRESHOLD, MV_SIMILARITY_NOISE_THRESHOLD) > 0.0);

    // Save dynamic mask to feedback for next set of frames
    tv_dynamic_mask_p1[uint4(idx.x, 0, idx.yz)] = dynamic_mask_m1;
}


[AutoPyBindCUDA]
[CUDAKernel]
[NoDiffThis]
void calculate_previous_dynamic_mask_v5(
    TensorView<float> tv_depth,
    TensorView<float> tv_mv_m1_f30_m3,
    TensorView<float> tv_motion_mat_tm1,
    TensorView<float> tv_dynamic_mask_p1,
)
{
    uint tId = (cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim()).x;

    // We Dispatch across `output`'s dimensions
    uint3 size = tensor_size(tv_depth).xzw; // N C H W -> NHW -> X Z W
    int H = size.y;
    int W = size.z;
    uint3 idx = tensor_idx(tId, size); // N H W
    if (any(idx >= size)) return;

    float2 uv = float2(idx.yz + 0.5f) * rcp(float2(size.yz));

    float2 mv_m1_f30_m3 = float2(
        tv_mv_m1_f30_m3[uint4(idx.x, 0, idx.yz)],
        tv_mv_m1_f30_m3[uint4(idx.x, 1, idx.yz)]
    )  * float2(size.yz);

    float depth_m1 = tv_depth[uint4(idx.x, 0, idx.yz)];
    float4x4 motion_mat_tm1 = tensor_to_float4x4(tv_motion_mat_tm1, idx.x);
    float mask;
    float2 cm_m1_f30_m3 = CalculateCameraMotion(uv, 1.0f-depth_m1, motion_mat_tm1, mask) * float2(size.yz);
    float dynamic_mask_m1 = float(dynamic_mask_v5(cm_m1_f30_m3, mv_m1_f30_m3, MV_SIMILARITY_THRESHOLD, MV_SIMILARITY_NOISE_THRESHOLD) > 0.0);

    // Save dynamic mask to feedback for next set of frames
    tv_dynamic_mask_p1[uint4(idx.x, 0, idx.yz)] = dynamic_mask_m1;
}
