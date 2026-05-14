// SPDX-FileCopyrightText: Copyright 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0

/* !NOTE: portions of this file adapt source code copied from FSR2, which has the following licence:

This file is part of the FidelityFX SDK.

Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Reference please see:
- https://github.com/GPUOpen-Effects/FidelityFX-FSR2/blob/HEAD/src/ffx-fsr2-api/shaders/ffx_fsr2_depth_clip.h
- https://github.com/GPUOpen-Effects/FidelityFX-FSR2/blob/HEAD/src/ffx-fsr2-api/shaders/ffx_fsr2_reconstruct_dilated_velocity_and_previous_depth.h
*/

import "../../../../common/model/shaders/tensor";


[NoDiffThis]
float EncodeNearestDepthCoord(int2 nearest_offset)
{
    int2 clamped = clamp(nearest_offset, -2, 2);
    int code = ((clamped.y + 2) << 3) | (clamped.x + 2);
    return float(code) / 255.0f;
}


[NoDiffThis]
uint EncodeNearestDepthCoordNibble(int2 nearest_offset)
{
    int2 clamped = clamp(nearest_offset, -1, 2);
    return uint(((clamped.y + 1) << 2) | (clamped.x + 1));
}


[NoDiffThis]
int2 DecodeNearestDepthCoordNibble(int code)
{
    int x = (code & 0x3) - 1;
    int y = ((code >> 2) & 0x3) - 1;
    return int2(x, y);
}


[NoDiffThis]
int2 DecodeNearestOffset(TensorView<float> nearest_tm1_offset, int2 pixel, int b)
{
    pixel = clamp(pixel, int2(0), int2(tensor_size(nearest_tm1_offset).zw) - 1);
    float norm_code = nearest_tm1_offset.load(uint4(b, 0, uint2(pixel)));
    int code = int(norm_code * 255.0f + 0.5f);
    int x = (code & 0x7) - 2;
    int y = ((code >> 3) & 0x7) - 2;
    return int2(x, y);
}


[NoDiffThis]
inline bool nss_v1_is_on_screen(int2 pos, uint2 size)
{
    return all(pos < int2(size)) && all(pos >= int2(0));
}


[NoDiffThis]
inline float nss_v1_gather_mask(int2 coord, int height, int width, bool clamp_to_edge)
{
    if (clamp_to_edge) {
        return 1.0;
    }

    bool inside = coord.x >= 0 && coord.x < height && coord.y >= 0 && coord.y < width;
    return inside ? 1.0 : 0.0;
}


[NoDiffThis]
float4 texture_gather(
    TensorView<float> tensor,
    uint b,
    uint c,
    float2 uv,
    bool clamp_to_edge = true)
{
    int height = tensor.size(2);
    int width = tensor.size(3);

    if (height == 0 || width == 0) {
        return float4(0.0);
    }

    int max_height_index = max(height - 1, 0);
    int max_width_index = max(width - 1, 0);

    float2 tex_size = float2(float(height), float(width));
    float2 idx = uv * tex_size;
    int2 g0 = int2(floor(idx - 0.5));
    int2 g1 = g0 + 1;

    int2 tl = int2(g0.x, g0.y);
    int2 tr = int2(g0.x, g1.y);
    int2 bl = int2(g1.x, g0.y);
    int2 br = int2(g1.x, g1.y);

    uint4 i_tl = uint4(b, c, clamp(tl.x, 0, max_height_index), clamp(tl.y, 0, max_width_index));
    uint4 i_tr = uint4(b, c, clamp(tr.x, 0, max_height_index), clamp(tr.y, 0, max_width_index));
    uint4 i_bl = uint4(b, c, clamp(bl.x, 0, max_height_index), clamp(bl.y, 0, max_width_index));
    uint4 i_br = uint4(b, c, clamp(br.x, 0, max_height_index), clamp(br.y, 0, max_width_index));

    float4 gathered;
    gathered.x = tensor[i_tl] * nss_v1_gather_mask(tl, height, width, clamp_to_edge);
    gathered.y = tensor[i_tr] * nss_v1_gather_mask(tr, height, width, clamp_to_edge);
    gathered.z = tensor[i_bl] * nss_v1_gather_mask(bl, height, width, clamp_to_edge);
    gathered.w = tensor[i_br] * nss_v1_gather_mask(br, height, width, clamp_to_edge);

    return gathered;
}


[NoDiffThis]
inline float nss_v1_get_view_space_depth(float depth, float4 device_to_view)
{
    return device_to_view[1] / (depth - device_to_view[0]);
}


[NoDiffThis]
inline float2 nss_v1_compute_ndc(float2 pix_pos, int2 size)
{
    return pix_pos / float2(size) * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f);
}


[NoDiffThis]
inline float3 nss_v1_get_view_space_position(
    int2 viewport_pos,
    int2 viewport_size,
    float device_depth,
    float4 device_to_view)
{
    const float z = nss_v1_get_view_space_depth(device_depth, device_to_view);
    const float2 ndc_pos = nss_v1_compute_ndc(viewport_pos, viewport_size);
    const float x = device_to_view[2] * ndc_pos.x * z;
    const float y = device_to_view[3] * ndc_pos.y * z;

    return float3(x, y, z);
}


struct NSSV1BilinearSamplingData
{
    int2 iOffsets[4];
    float fWeights[4];
    int2 iBasePos;
};


[NoDiffThis]
inline NSSV1BilinearSamplingData nss_v1_get_bilinear_sampling_data(float2 uv, uint2 size)
{
    NSSV1BilinearSamplingData data;

    float2 px_sample = uv * float2(size) - float2(0.5);
    data.iBasePos = int2(floor(px_sample));
    float2 px_frac = frac(px_sample);

    data.iOffsets[0] = int2(0, 0);
    data.iOffsets[1] = int2(1, 0);
    data.iOffsets[2] = int2(0, 1);
    data.iOffsets[3] = int2(1, 1);
    data.fWeights[0] = (1 - px_frac.x) * (1 - px_frac.y);
    data.fWeights[1] = px_frac.x * (1 - px_frac.y);
    data.fWeights[2] = (1 - px_frac.x) * px_frac.y;
    data.fWeights[3] = px_frac.x * px_frac.y;

    return data;
}


[NoDiffThis]
void FindNearestDepth_4x4(
    TensorView<float> depth_tensor,
    uint b,
    float2 uv,
    bool inverted,
    out float nearest_depth,
    out uint2 nearest_depth_coord,
    out int2 nearest_offset)
{
    const uint sample_count = 16;
    const int2 sample_offsets[sample_count] = {
        int2(+0, +0),
        int2(+1, +0),
        int2(+0, +1),
        int2(+0, -1),
        int2(-1, +0),
        int2(-1, +1),
        int2(+1, +1),
        int2(-1, -1),
        int2(+1, -1),
        int2(-1, +2),
        int2(+0, +2),
        int2(+1, +2),
        int2(+2, +2),
        int2(+2, +1),
        int2(+2, +0),
        int2(+2, -1),
    };

    uint2 px_size = tensor_size(depth_tensor).zw;
    uint2 px_pos = uint2(uv * float2(px_size));

    float depth[sample_count];
    uint sample_index = 0;
    [ForceUnroll]
    for (sample_index = 0; sample_index < sample_count; ++sample_index) {
        uint2 pos = clamp(px_pos + sample_offsets[sample_index], uint2(0), px_size - 1);
        depth[sample_index] = depth_tensor[uint4(b, 0, pos)];
    }

    nearest_depth = depth[0];
    nearest_depth_coord = px_pos;
    nearest_offset = int2(0);
    [ForceUnroll]
    for (sample_index = 1; sample_index < sample_count; ++sample_index) {
        int2 pos = int2(px_pos) + sample_offsets[sample_index];
        if (nss_v1_is_on_screen(pos, px_size)) {
            float depth_n = depth[sample_index];
            bool closer = inverted ? depth_n > nearest_depth : depth_n < nearest_depth;
            if (closer) {
                nearest_depth_coord = uint2(pos);
                nearest_depth = depth_n;
                nearest_offset = sample_offsets[sample_index];
            }
        }
    }
}


[NoDiffThis]
float ComputeDepthClip(
    TensorView<int> depth_tm1,
    uint b,
    float2 uv,
    float2 render_size,
    bool inverted,
    float current_depth,
    float4 device_to_view,
    float bilinear_weight_threshold = 0.1f)
{
    float2 depth_tm1_size = tensor_size(depth_tm1).zw;
    float current_depth_view_space = nss_v1_get_view_space_depth(current_depth, device_to_view);
    NSSV1BilinearSamplingData bilinear_info = nss_v1_get_bilinear_sampling_data(
        uv,
        uint2(depth_tm1_size)
    );

    float depth = 0.0f;
    float weight_sum = 0.0f;
    for (int sample_index = 0; sample_index < 4; sample_index++) {
        const int2 offset = bilinear_info.iOffsets[sample_index];
        const int2 sample_pos = bilinear_info.iBasePos + offset;

        const float weight = bilinear_info.fWeights[sample_index];
        const bool onscreen = nss_v1_is_on_screen(sample_pos, uint2(depth_tm1_size));
        weight_sum += onscreen ? 0.f : weight;

        if (onscreen && weight > bilinear_weight_threshold) {
            const float prev_depth_sample =
                float(depth_tm1[uint4(b, 0, sample_pos)]) * rcp(float(0x7FFFFFFF));
            const float prev_nearest_depth_view_space = nss_v1_get_view_space_depth(
                prev_depth_sample,
                device_to_view
            );

            const float depth_diff = current_depth_view_space - prev_nearest_depth_view_space;
            if (depth_diff > 0.0f) {
                const float plane_depth = inverted
                    ? min(prev_depth_sample, current_depth)
                    : max(prev_depth_sample, current_depth);

                const float3 center = nss_v1_get_view_space_position(
                    int2(render_size * 0.5f),
                    int2(render_size),
                    plane_depth,
                    device_to_view
                );
                const float3 corner = nss_v1_get_view_space_position(
                    int2(0, 0),
                    int2(render_size),
                    plane_depth,
                    device_to_view
                );

                const float half_viewport_width = length(float2(render_size));
                const float depth_threshold = max(
                    current_depth_view_space,
                    prev_nearest_depth_view_space
                );

                const float k_sep = 1.37e-05f;
                const float k_fov = length(corner) / length(center);
                const float required_depth_separation =
                    k_sep * k_fov * half_viewport_width * depth_threshold;

                const float resolution_factor = saturate(
                    length(float2(render_size)) / length(float2(1080.0f, 1920.0f))
                );
                const float power = lerp(1.0f, 3.0f, resolution_factor);
                depth += pow(saturate(float(required_depth_separation / depth_diff)), power)
                    * weight;
                weight_sum += weight;
            }
        }
    }

    return (weight_sum > 0) ? saturate(1.0f - depth / weight_sum) : 0.0f;
}


[NoDiffThis]
inline void nss_v1_store_reconstructed_depth(int4 px_pos, TensorView<int> out_depth, float depth)
{
    int old_val;
    int int_depth = int(depth * float(0x7FFFFFFF));
    out_depth.InterlockedMin(px_pos, int_depth, old_val);
}


[NoDiffThis]
void ReconstructPrevDepth(
    int b,
    TensorView<int> out_depth,
    float depth,
    float2 reprojected_uv,
    float2 render_size,
    float bilinear_weight_threshold = 0.1f)
{
    uint2 depth_size = tensor_size(out_depth).zw;
    NSSV1BilinearSamplingData bilinear_info = nss_v1_get_bilinear_sampling_data(
        reprojected_uv,
        uint2(depth_size)
    );

    for (int sample_index = 0; sample_index < 4; sample_index++) {
        const int2 offset = bilinear_info.iOffsets[sample_index];
        float weight = bilinear_info.fWeights[sample_index];

        if (weight > bilinear_weight_threshold) {
            int2 store_pos = bilinear_info.iBasePos + offset;
            if (nss_v1_is_on_screen(store_pos, depth_size)) {
                nss_v1_store_reconstructed_depth(int4(b, 0, store_pos), out_depth, depth);
            }
        }
    }
}
