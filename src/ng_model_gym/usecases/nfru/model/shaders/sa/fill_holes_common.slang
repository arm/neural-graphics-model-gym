// SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
// its affiliates <open-source-office@arm.com></text>
// SPDX-License-Identifier: Apache-2.0

#include "constants.slang"
#include "glsl_mappings.slang"

import "common";
import "quant";

#ifndef FILL_HOLES_COMMON
#define FILL_HOLES_COMMON

void FillHoles(
    uint batch,
    uint2 output_pixel,
    int2 output_dims,
    float2 inv_output_dims,
    TensorView<int> warped_data_texture,
    TensorView<float> output_texture
){
    /*
        Loosely based on:
        https://github.com/arm/accuracy-super-resolution-generic-library/blob/38697a58a6e7818ec9d28774bc073f537abb9178/
        include/gpu/fsr2/ffxm_fsr2_reconstruct_dilated_velocity_and_previous_depth.h#L59
    */
    const int32_t iSampleCount = HOLE_FILL_SAMPLE_COUNT;
    int depth_mvs_packed[iSampleCount];
    const int2 offsets[HOLE_FILL_SAMPLE_COUNT] = HOLE_FILL_OFFSETS_INT;

    // pull out the depth loads to allow SC to batch them
    depth_mvs_packed[0] = texelFetchOffset(batch, warped_data_texture, output_pixel, offsets[0]);
    depth_mvs_packed[1] = texelFetchOffset(batch, warped_data_texture, output_pixel, offsets[1]);
    depth_mvs_packed[2] = texelFetchOffset(batch, warped_data_texture, output_pixel, offsets[2]);
    depth_mvs_packed[3] = texelFetchOffset(batch, warped_data_texture, output_pixel, offsets[3]);
    depth_mvs_packed[4] = texelFetchOffset(batch, warped_data_texture, output_pixel, offsets[4]);
    depth_mvs_packed[5] = texelFetchOffset(batch, warped_data_texture, output_pixel, offsets[5]);
    depth_mvs_packed[6] = texelFetchOffset(batch, warped_data_texture, output_pixel, offsets[6]);
    depth_mvs_packed[7] = texelFetchOffset(batch, warped_data_texture, output_pixel, offsets[7]);
    depth_mvs_packed[8] = texelFetchOffset(batch, warped_data_texture, output_pixel, offsets[8]);

    // find closest depth
    int nearest_packed_depth_mv = depth_mvs_packed[0];
    [ForceUnroll]
    for (int32_t iSampleIndex = 1; iSampleIndex < iSampleCount; ++iSampleIndex) {
        int tap = depth_mvs_packed[iSampleIndex];
        if (tap > nearest_packed_depth_mv) {
            nearest_packed_depth_mv = tap;
        }
    }

    // Unpack / Dequantize motion
    float2 motion = dqFloatMv(nearest_packed_depth_mv);
    motion *= float2(inv_output_dims);

    // Write dilated vector
    imageStore(batch, output_texture, output_pixel, float4(motion, 0, 0));
}

#endif // FILL_HOLES_COMMON
