// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0

import "../../../../common/model/shaders/depth_clip";
import "../../../../common/model/shaders/tensor";

#define LQ_DISOCCLUSION_MOTION_THRESHOLD 0.1f
#ifndef NSS_LQ_DISOCCLUSION_DEPTH_GRADIENT_TOLERANCE_SCALE
#define NSS_LQ_DISOCCLUSION_DEPTH_GRADIENT_TOLERANCE_SCALE 1.0f
#endif // !NSS_LQ_DISOCCLUSION_DEPTH_GRADIENT_TOLERANCE_SCALE

#define LQ_DISOCCLUSION_PARAMS(X)          \
    X(TensorView<float>, in_motion),       \
    X(TensorView<float>, in_depth),        \
    X(TensorView<int>,   in_depth_tm1),    \
    X(TensorView<float>, in_depth_params), \
    X(TensorView<float>, in_render_size),  \
    X(TensorView<float>, out_disocclusion_mask)

#define LQ_DECL(type, name) type name
#define LQ_NAME_ONLY(_, name) name
#define LQ_DISOCCLUSION_ARGS LQ_DISOCCLUSION_PARAMS(LQ_NAME_ONLY)


[NoDiffThis]
inline bool LqDisocclusionIsOnScreen(int2 pos, int2 size)
{
    return all(pos >= int2(0)) && all(pos < size);
}


[NoDiffThis]
inline int2 LqDisocclusionClampToSize(int2 pos, int2 size)
{
    return clamp(pos, int2(0), size - 1);
}


[NoDiffThis]
inline void LqDisocclusionFindDepthMotion4x4(
    TensorView<float> in_motion,
    TensorView<float> in_depth,
    uint batch,
    int2 dst_pos,
    int2 dst_size,
    float4 device_to_view,
    out float closest_depth,
    out float2 closest_motion,
    out float local_view_depth_range)
{
    int2 depth_size = int2(tensor_size(in_depth).zw);
    int2 src_base = int2(float2(dst_pos) * (float2(depth_size) / float2(dst_size)));
    int2 src_pos = LqDisocclusionClampToSize(src_base, depth_size);

    closest_depth = in_depth[uint4(batch, 0, uint2(src_pos))];
    float src_view_depth = GetViewSpaceDepth(closest_depth, device_to_view);
    float min_view_depth = src_view_depth;
    float max_view_depth = src_view_depth;
    closest_motion = float2(
        in_motion[uint4(batch, 0, uint2(src_pos))],
        in_motion[uint4(batch, 1, uint2(src_pos))]
    );

    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            int2 sample_pos = src_base + int2(x, y);
            if (LqDisocclusionIsOnScreen(sample_pos, depth_size)) {
                float sample_depth = in_depth[uint4(batch, 0, uint2(sample_pos))];
                float sample_view_depth = GetViewSpaceDepth(sample_depth, device_to_view);
                min_view_depth = min(min_view_depth, sample_view_depth);
                max_view_depth = max(max_view_depth, sample_view_depth);
                if (sample_depth < closest_depth) {
                    closest_depth = sample_depth;
                    closest_motion = float2(
                        in_motion[uint4(batch, 0, uint2(sample_pos))],
                        in_motion[uint4(batch, 1, uint2(sample_pos))]
                    );
                }
            }
        }
    }

    local_view_depth_range = max_view_depth - min_view_depth;
}


[NoDiffThis]
inline float2 LqDisocclusionMotionToDepthPixels(float2 motion_pixels, int2 input_size, int2 depth_size)
{
    float2 motion_depth_pixels = motion_pixels * (float2(depth_size) / float2(input_size));
    return motion_depth_pixels * float(length(motion_pixels) > LQ_DISOCCLUSION_MOTION_THRESHOLD);
}


[NoDiffThis]
inline float LqDisocclusionComputeDepthClip(
    TensorView<int> depth_tm1,
    uint batch,
    float2 uv,
    float2 render_size,
    float current_depth,
    float4 device_to_view,
    float local_view_depth_range,
    float bilinear_weight_threshold = 0.1f)
{
    float2 depth_tm1_size = tensor_size(depth_tm1).zw;
    float current_view_depth = GetViewSpaceDepth(current_depth, device_to_view);
    BilinearSamplingData bilinear_info = GetBilinearSamplingData(
        uv,
        uint2(depth_tm1_size)
    );
    float local_depth_tolerance =
        local_view_depth_range * NSS_LQ_DISOCCLUSION_DEPTH_GRADIENT_TOLERANCE_SCALE;

    float depth = 0.0f;
    float weight_sum = 0.0f;
    for (int sample_index = 0; sample_index < 4; sample_index++) {
        const int2 offset = bilinear_info.iOffsets[sample_index];
        const int2 sample_pos = bilinear_info.iBasePos + offset;
        const float weight = bilinear_info.fWeights[sample_index];
        const bool onscreen = IsOnScreen(sample_pos, uint2(depth_tm1_size));
        weight_sum += onscreen ? 0.0f : weight;

        if (onscreen && weight > bilinear_weight_threshold) {
            const float prev_depth =
                float(depth_tm1[uint4(batch, 0, sample_pos)]) * rcp(float(0x7FFFFFFF));
            const float prev_view_depth = GetViewSpaceDepth(prev_depth, device_to_view);
            const float depth_diff = current_view_depth - prev_view_depth;

            if (depth_diff > 0.0f) {
                const float plane_depth = max(prev_depth, current_depth);
                const float3 center = GetViewSpacePosition(
                    int2(render_size * 0.5f),
                    int2(render_size),
                    plane_depth,
                    device_to_view
                );
                const float3 corner = GetViewSpacePosition(
                    int2(0, 0),
                    int2(render_size),
                    plane_depth,
                    device_to_view
                );
                const float half_viewport_width = length(float2(render_size));
                const float depth_threshold = max(current_view_depth, prev_view_depth);
                const float required_sep =
                    (1.37e-05f * (length(corner) / length(center)) * half_viewport_width * depth_threshold)
                    + local_depth_tolerance;
                const float resolution_factor = saturate(
                    length(float2(render_size)) / length(float2(1080.0f, 1920.0f))
                );
                const float power = lerp(1.0f, 3.0f, resolution_factor);
                depth += pow(saturate(float(required_sep / depth_diff)), power) * weight;
                weight_sum += weight;
            }
        }
    }

    return (weight_sum > 0.0f) ? saturate(1.0f - depth / weight_sum) : 0.0f;
}


[AutoPyBindCUDA]
[CUDAKernel]
[NoDiffThis]
public void lq_disocclusion_mask(LQ_DISOCCLUSION_PARAMS(LQ_DECL))
{
    uint tId = (cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim()).x;
    uint3 size = tensor_size(out_disocclusion_mask).xzw;
    uint3 idx = tensor_idx(tId, size);
    if (any(idx >= size)) return;

    int2 depth_size = int2(tensor_size(out_disocclusion_mask).zw);
    int2 input_size = int2(tensor_size(in_depth).zw);
    float2 inv_depth_size = rcp(float2(depth_size));
    float2 uv = (float2(idx.yz) + 0.5f) * inv_depth_size;
    float4 device_to_view = float4(
        in_depth_params[uint4(idx.x, 0, 0, 0)],
        in_depth_params[uint4(idx.x, 1, 0, 0)],
        in_depth_params[uint4(idx.x, 2, 0, 0)],
        in_depth_params[uint4(idx.x, 3, 0, 0)]
    );
    float2 render_size = float2(
        in_render_size[uint4(idx.x, 0, 0, 0)],
        in_render_size[uint4(idx.x, 1, 0, 0)]
    );

    float current_depth;
    float2 motion;
    float local_view_depth_range;
    LqDisocclusionFindDepthMotion4x4(
        in_motion,
        in_depth,
        idx.x,
        int2(idx.yz),
        depth_size,
        device_to_view,
        current_depth,
        motion,
        local_view_depth_range
    );

    float2 motion_depth_pixels = LqDisocclusionMotionToDepthPixels(
        motion,
        input_size,
        depth_size
    );
    float2 reproj_uv = uv - (motion_depth_pixels * inv_depth_size);

    out_disocclusion_mask[uint4(idx.x, 0, idx.yz)] = LqDisocclusionComputeDepthClip(
        in_depth_tm1,
        idx.x,
        reproj_uv,
        render_size,
        current_depth,
        device_to_view,
        local_view_depth_range
    );
}

#undef LQ_DISOCCLUSION_ARGS
#undef LQ_NAME_ONLY
#undef LQ_DECL
#undef LQ_DISOCCLUSION_PARAMS
