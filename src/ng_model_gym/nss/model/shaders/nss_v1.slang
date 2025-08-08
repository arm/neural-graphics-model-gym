// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
import "./tensor";
import "./tonemap";
import "./depth";
import "./image";
import "./depth_clip";

#define EPS 1e-7

[NoDiffThis]
public void ShaderAccurateFindNearestDepth(TensorView<float> depth_tensor, uint b, float2 uv, bool inverted, out float nearestDepth, out int2 nearestDepthCoordOffset)
{
    const uint iSampleCount = 9;
    // y, x - TODO: check accuracy when swizzled
    const int2 iSampleOffsets[iSampleCount] = {
        int2(+0, +0),
        int2(+1, +0),
        int2(+0, +1),
        int2(+0, -1),
        int2(-1, +0),
        int2(-1, +1),
        int2(+1, +1),
        int2(-1, -1),
        int2(+1, -1),
    };

    uint2 iPxSize = tensor_size(depth_tensor).zw;
    uint2 iPxPos = uint2(uv * float2(iPxSize));

    // pull out the depth loads to allow SC to batch them
    float depth[9];
    uint iSampleIndex = 0;
    [ForceUnroll]
    for (iSampleIndex = 0; iSampleIndex < iSampleCount; ++iSampleIndex) {
        uint2 iPos = clamp(iPxPos + iSampleOffsets[iSampleIndex], uint2(0), iPxSize - 1);
        depth[iSampleIndex] = depth_tensor[uint4(b, 0, iPos)];
    }

    // find closest depth
    nearestDepth = depth[0];
    nearestDepthCoordOffset = iSampleOffsets[0];
    [ForceUnroll]
    for (iSampleIndex = 1; iSampleIndex < iSampleCount; ++iSampleIndex) {
        int2 iPos = int2(iPxPos) + iSampleOffsets[iSampleIndex];
        if (IsOnScreen(iPos, iPxSize)) {
            float depthN = depth[iSampleIndex];
            bool closer = inverted ? depthN > nearestDepth : depthN < nearestDepth;
            if (closer) {
                nearestDepthCoordOffset = iSampleOffsets[iSampleIndex];
                nearestDepth = depthN;
            }
        }
    }
}

[NoDiffThis]
public float ShaderAccurateComputeDepthClip(
    TensorView<float> depth_tm1,
    TensorView<float> nearest_offset_tm1,
    uint b,
    float2 uv,
    float2 render_size,
    bool inverted,
    float current_depth,
    float4 device_to_view
) {
    const float fReconstructedDepthBilinearWeightThreshold = 0.1f;
    float2 depth_tm1_size = tensor_size(depth_tm1).zw;
    float fCurrentDepthViewSpace = GetViewSpaceDepth(current_depth, device_to_view);
    BilinearSamplingData bilinearInfo = GetBilinearSamplingData(uv, uint2(depth_tm1_size));

    float fDilatedSum = 0.0f;
    float fDepth = 0.0f;
    float fWeightSum = 0.0f;
    for (int iSampleIndex = 0; iSampleIndex < 4; iSampleIndex++) {

        const int2 iOffset = bilinearInfo.iOffsets[iSampleIndex];
        int2 dilation_offset = int2(0, 0);

        dilation_offset = DecodeNearestOffset(nearest_offset_tm1, bilinearInfo.iBasePos, b);
        const int2 iSamplePos = bilinearInfo.iBasePos + iOffset + dilation_offset;

        // Modification for off-screen regions to be flagged as disocclusions too
        const float fWeight = bilinearInfo.fWeights[iSampleIndex];
        const bool onscreen = IsOnScreen(iSamplePos, uint2(depth_tm1_size));
        fWeightSum += onscreen ? 0.f : fWeight;

        if (onscreen) {
            if (fWeight > fReconstructedDepthBilinearWeightThreshold) {
                // TODO: check if grabbing prev depth samples is being accurate
                const float fPrevDepthSample = depth_tm1[uint4(b, 0, iSamplePos)];
                const float fPrevNearestDepthViewSpace = GetViewSpaceDepth(fPrevDepthSample, device_to_view);

                const float fDepthDiff = fCurrentDepthViewSpace - fPrevNearestDepthViewSpace;

                if (fDepthDiff > 0.0f) {
                    const float fPlaneDepth = inverted ? min(fPrevDepthSample, current_depth) : max(fPrevDepthSample, current_depth);

                    const float3 fCenter = GetViewSpacePosition(int2(render_size * 0.5f), int2(render_size), fPlaneDepth, device_to_view);
                    const float3 fCorner = GetViewSpacePosition(int2(0, 0), int2(render_size), fPlaneDepth, device_to_view);

                    const float fHalfViewportWidth = length(float2(render_size));
                    const float fDepthThreshold = max(fCurrentDepthViewSpace, fPrevNearestDepthViewSpace);

                    const float Ksep = 1.37e-05f;
                    const float Kfov = length(fCorner) / length(fCenter);
                    const float fRequiredDepthSeparation = Ksep * Kfov * fHalfViewportWidth * fDepthThreshold;

                    const float fResolutionFactor = saturate(length(float2(render_size)) / length(float2(1080.0f, 1920.0f)));
                    const float fPower = lerp(1.0f, 3.0f, fResolutionFactor);
                    fDepth += pow(saturate(float(fRequiredDepthSeparation / fDepthDiff)), fPower) * fWeight;
                    fWeightSum += fWeight;
                }
            }
        }
    }

    return (fWeightSum > 0) ? saturate(1.0f - fDepth / fWeightSum) : 0.0f;
}

[NoDiffThis]
float EncodeNearestDepthCoord(int2 nearest_offset)
{
    #ifdef SCENARIO_ACCURATE
    return float(
        (nearest_offset.x + 1) << 2 | (nearest_offset.y + 1)
    ) / 255.0f;
    #else
    return float(
        (nearest_offset.y + 1) << 2 | (nearest_offset.x + 1)
    ) / 255.0f;
    #endif
}

[NoDiffThis]
int2 DecodeNearestOffset(TensorView<float> nearest_tm1_offset, int2 pixel, int b)
{
    pixel = clamp(pixel, uint2(0), tensor_size(nearest_tm1_offset).zw - 1);
    float norm_code = nearest_tm1_offset.load(uint4(b, 0, pixel));
    int code = int(norm_code * 255.0f + 0.5f);
    int x = int( code       & 0x3) - 1;  // bits 0-1
    int y = int((code >> 2) & 0x3) - 1;  // bits 2-3
    #ifdef SCENARIO_ACCURATE
    return int2(y, x);
    #else
    return int2(x, y);
    #endif
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void pre_process_v1(
    TensorView<float> colour,
    DiffTensorView history,
    TensorView<float> motion,
    TensorView<float> depth,
    TensorView<float> depth_tm1,
    TensorView<float> jitter,
    TensorView<float> jitter_tm1,
    DiffTensorView feedback_tm1,
    TensorView<float> derivative_tm1,
    TensorView<float> depth_params,
    TensorView<float> exposure,
    TensorView<float> render_size,
    DiffTensorView output_tensor,
    TensorView<float> out_luma_derivative,
    TensorView<float> out_depth_t,
    DiffTensorView dm_scale,
)
{
    uint tId = (cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim()).x;

    // We Dispatch across `output`'s dimensions
    uint3 size = tensor_size(output_tensor).xzw;
    uint3 idx = tensor_idx(tId, size);
    if (any(idx >= size)) return;

    float2 uv = (float2(idx.yz + 0.5f) / float2(size.yz));

    // Upscaling factor
    const float2 scale = float2(2.f);

    // Dilate Depth
    // ------------------------------------
    const bool inverted_depth = false;
    float depth_dilated = 0.f;
    uint2 nearest_coord = uint2(0);
    FindNearestDepth(depth, idx.x, uv, inverted_depth, depth_dilated, nearest_coord);
    float4 device_to_view = float4(
        no_diff(depth_params[uint4(idx.x, 0, 0, 0)]),
        no_diff(depth_params[uint4(idx.x, 1, 0, 0)]),
        no_diff(depth_params[uint4(idx.x, 2, 0, 0)]),
        no_diff(depth_params[uint4(idx.x, 3, 0, 0)]),
    );

    // Calculate UV's for tensor sampling
    // ------------------------------------
    float2 nearest_uv = (float2(nearest_coord) + 0.5f) / float2(size.yz);
    float2 vector = sample_tensor(motion, idx.x, uint2(0, 1), nearest_uv) * rcp(scale);
    vector *= float(length(vector) > 0.1f);

    float2 reproj_uv = uv - (vector / (float2(size.yz)));

    float2 jit_tm1 = float2(no_diff(jitter_tm1[uint4(idx.x, 0, 0, 0)]), no_diff(jitter_tm1[uint4(idx.x, 1, 0, 0)]));
    float2 unjitter_tm1_uv = reproj_uv - (jit_tm1 / size.yz);

    float2 jit_t = float2(no_diff(jitter[uint4(idx.x, 0, 0, 0)]), no_diff(jitter[uint4(idx.x, 1, 0, 0)]));

    float2 render_sz = float2(no_diff(render_size[uint4(idx.x, 0, 0, 0)]), no_diff(render_size[uint4(idx.x, 1, 0, 0)]));

    float img_exposure = no_diff(exposure[uint4(idx.x, 0, 0, 0)]);

    // Depth-based Disocclusion Mask
    // ------------------------------------
    float disocclusion_mask = no_diff(ComputeDepthClip(depth_tm1, idx.x, unjitter_tm1_uv, render_sz, inverted_depth, depth_dilated, device_to_view));

    // Small hack -> infinitely far depths are always flagged (sky box, etc.)
    const float furthest_depth = inverted_depth ? 0.f : 1.f;
    disocclusion_mask = depth_dilated == furthest_depth ? 1.f : disocclusion_mask;

    float dis_mask_scale = length(vector) > 0.5f ? 1.0 : dm_scale.load(0);
    disocclusion_mask = disocclusion_mask * dis_mask_scale;
    // Unjitter LR Colour
    // ------------------------------------
    float3 jittered_colour = float3(
        no_diff(colour[uint4(idx.x, 0, idx.yz)]),
        no_diff(colour[uint4(idx.x, 1, idx.yz)]),
        no_diff(colour[uint4(idx.x, 2, idx.yz)]),
    );
    jittered_colour = tonemap_forward(jittered_colour * img_exposure, TonemapMode.Karis);

    // Warp History Buffer @ input resolution to Current Frame `t_tm1` -> `t`
    // ------------------------------------
    float3 lr_warped_history = sample_tensor(history, idx.x, uint3(0, 1, 2), reproj_uv, false);
    lr_warped_history = tonemap_forward(lr_warped_history * img_exposure, TonemapMode.Karis);

    // Calculate Derivative of `luma` - helps identifying high-frequency flicker due to jitter
    // ------------------------------------
    const float derivative_dis_thresh = 0.01; // threshold, doesn't matter too much, but must be > 0.
    const float deriv_pow = 1.5; // increase importance of larger deviations
    const float deriv_min = 0.05; // threshold, removes residual ghosting
    const float deriv_max = 0.3; // threshold, doesn't matter too much, helps allocate precision
    const float deriv_denom = rcp(pow(deriv_max, deriv_pow));
    const float d_alpha = 0.1;

    // Unjittered colour in luma
    float luma_t = luminance(jittered_colour);

    // Warp Previous derivative and luma
    float2 deriv_tm1 = sample_tensor(derivative_tm1, idx.x, uint2(0, 1), reproj_uv, false);
    float luma_derivative_tm1 = deriv_tm1.x;
    float luma_tm1 = deriv_tm1.y;

    // Calculate new derivative
    float luma_derivative_t = abs(luma_t - luma_tm1);

    // Operations applied here:
    //   1) Discard values less than `deriv_min` to reduce ghosting
    //   2) Clip to `deriv_max` which is ~typical max value (allows for better precision allocation when normalized)
    //   3) Apply a power curve, puts more precision towards larger differences
    //   4) Normalize clipped derivative to [0, 1] range
    luma_derivative_t = luma_derivative_t > deriv_min ? min(luma_derivative_t, deriv_max) : 0.f;
    luma_derivative_t = pow(luma_derivative_t, deriv_pow) * deriv_denom;

    // Accumulate the new derivative into the history.
    // We apply an adaptive alpha scaling, to ensure that if a derivative converges to a high value
    // it becomes more difficult to reset that value, this provides temporally stable convergence
    float applied_d_alpha = lerp(d_alpha, d_alpha * 0.1, min(luma_derivative_tm1, deriv_max) * rcp(deriv_max));
    float luma_derivative = lerp(luma_derivative_tm1, luma_derivative_t, applied_d_alpha);

    // Remove disoccluded regions of the derivative
    float disocclusion_binary = float(disocclusion_mask > derivative_dis_thresh);
    luma_derivative = lerp(luma_derivative, 0.f, disocclusion_binary);

    // Warp Feedback and Previous Params to Current Frame `t_tm1` -> `t`
    // ------------------------------------
    float4 feedback = sample_tensor(feedback_tm1, idx.x, uint4(0,1,2,3), reproj_uv, false);

    // Write Outputs
    // ------------------------------------
    output_tensor.storeOnce(uint4(idx.x, 0, idx.yz), lr_warped_history.r);
    output_tensor.storeOnce(uint4(idx.x, 1, idx.yz), lr_warped_history.g);
    output_tensor.storeOnce(uint4(idx.x, 2, idx.yz), lr_warped_history.b);
    output_tensor.storeOnce(uint4(idx.x, 3, idx.yz), jittered_colour.r);
    output_tensor.storeOnce(uint4(idx.x, 4, idx.yz), jittered_colour.g);
    output_tensor.storeOnce(uint4(idx.x, 5, idx.yz), jittered_colour.b);
    output_tensor.storeOnce(uint4(idx.x, 6, idx.yz), disocclusion_mask);
    output_tensor.storeOnce(uint4(idx.x, 7, idx.yz), feedback.r);
    output_tensor.storeOnce(uint4(idx.x, 8, idx.yz), feedback.g);
    output_tensor.storeOnce(uint4(idx.x, 9, idx.yz), feedback.b);
    output_tensor.storeOnce(uint4(idx.x, 10, idx.yz), feedback.a);
    output_tensor.storeOnce(uint4(idx.x, 11, idx.yz), luma_derivative);

    out_luma_derivative.store(uint4(idx.x, 0, idx.yz), luma_derivative);
    out_luma_derivative.store(uint4(idx.x, 1, idx.yz), luma_t);

    out_depth_t.store(uint4(idx.x, 0, idx.yz), depth_dilated);
}

[Differentiable]
public float3 simulate_r11g11b10_fakeq(float3 rgb) {
    const float epsilon = 1e-12;
    const int r_m_bits = 6, g_m_bits = 6, b_m_bits = 5;
    const int exp_bits = 5;
    const float bias = no_diff(float((1 << (exp_bits - 1)) - 1)); // 15
    const float exp_max = no_diff(float((1 << exp_bits) - 1));

    float3 val = max(rgb, float3(epsilon));

    float3 exp_unclamped = no_diff(floor(log2(val)));
    float3 exp_clipped = clamp(exp_unclamped, -bias, bias + 1.0);

    float3 mant = val / exp2(exp_clipped) - 1.0;

    float r_exp = clamp(exp_clipped.x + bias, 0.0, exp_max) - bias;
    float g_exp = clamp(exp_clipped.y + bias, 0.0, exp_max) - bias;
    float b_exp = clamp(exp_clipped.z + bias, 0.0, exp_max) - bias;

    float r_mant = clamp(no_diff(round(mant.x * float(1 << r_m_bits))), 0.0, float((1 << r_m_bits) - 1)) / float(1 << r_m_bits);
    float g_mant = clamp(no_diff(round(mant.y * float(1 << g_m_bits))), 0.0, float((1 << g_m_bits) - 1)) / float(1 << g_m_bits);
    float b_mant = clamp(no_diff(round(mant.z * float(1 << b_m_bits))), 0.0, float((1 << b_m_bits) - 1)) / float(1 << b_m_bits);

    float r_out = (1.0 + r_mant) * exp2(r_exp);
    float g_out = (1.0 + g_mant) * exp2(g_exp);
    float b_out = (1.0 + b_mant) * exp2(b_exp);

    return float3(r_out, g_out, b_out);
}

[BackwardDerivativeOf(simulate_r11g11b10_fakeq)]
public void simulate_r11g11b10_fakeq_bwd(
    inout DifferentialPair<float3> rgb,
    float3 upstream)
{
    // Identity gradient — no gradient blockage from quantization
    rgb = rgb.dadd(rgb.dzero(), DifferentialPair<float3>(rgb.p, upstream));
}


[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void post_process_v1_shader_accurate(
    TensorView<float> colour,
    DiffTensorView history,
    DiffTensorView t_kpn_params,
    DiffTensorView temporal_params,
    TensorView<float> motion,
    TensorView<float> nearest_offset,
    TensorView<float> exposure,
    TensorView<float> jitter,
    TensorView<float> offset_lut,
    TensorView<float> scale,
    TensorView<int> idx_modulo,
    DiffTensorView output,
    DiffTensorView out_filtered,
)
{
    struct KernelTile {
        int4 dy;
        int4 dx;
        int4 k_idx;
    };
    const KernelTile kernelLut[4] = {
        {
            int4(-1, -1, +1, +1),
            int4(-1, +1, -1, +1),
            int4(0, 8, 2, 10),
        },
        {
            int4(-1, -1, +1, +1),
            int4(+0, +2, +0, +2),
            int4(4, 12, 6, 14),
        },
        {
            int4(+0, +0, +2, +2),
            int4(-1, +1, -1, +1),
            int4(1, 9, 3, 11),
        },
        {
            int4( 0, +0, +2, +2),
            int4( 0, +2, +0, +2),
            int4(5, 13, 7, 15),
        }
    };

    // Thread location
    uint tId = (cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim()).x;
    uint3 size = tensor_size(output).xzw;
    uint3 lr_size = tensor_size(colour).xzw;
    uint3 idx = tensor_idx(tId, size);
    if (any(idx >= size)) return;

    float2 inv_dims = rcp(size.yz);
    float2 uv = (float2(idx.yz + 0.5f) * inv_dims);

    //!NOTE: assumes same scale factor for height/width
    float scale = no_diff(scale[0]);
    int idx_mod = no_diff(idx_modulo[0]);

    // constants
    uint2 colour_size = tensor_size(colour).zw;
    uint2 output_size = size.yz;

    // Indexes
    int2 base_idx = idx.yz;
    int2 lr_coord = int2(floor((base_idx + 0.5f) * rcp(scale)));

    // Load Exposure
    float img_exposure = no_diff(exposure[uint4(idx.x, 0, 0, 0)]);

    // filtering
    float2 offset_idx = float2(
        no_diff(offset_lut[uint4(idx.x, 0, 0, 0)]),
        no_diff(offset_lut[uint4(idx.x, 1, 0, 0)]),
    );

    int2 tiled_idx = (idx.yz + int2(offset_idx)) % idx_mod;
    int lut_idx = tiled_idx.x * idx_mod + tiled_idx.y;
    KernelTile lut = kernelLut[lut_idx];

    float4 kpn_weights = max(sample_tensor(
        t_kpn_params,
        idx.x,
        uint4(lut.k_idx),
        uv
    ), EPS);

    uint2 tap_loc1 = clamp(
        uint2((float2(idx.yz + 0.5) + float2(lut.dy.x, lut.dx.x)) * 0.5f),
        uint2(0),
        lr_size.yz - 1
    );
    uint2 tap_loc2 = clamp(
        uint2((float2(idx.yz + 0.5) + float2(lut.dy.y, lut.dx.y)) * 0.5f),
        uint2(0),
        lr_size.yz - 1
    );
    uint2 tap_loc3 = clamp(
        uint2((float2(idx.yz + 0.5) + float2(lut.dy.z, lut.dx.z)) * 0.5f),
        uint2(0),
        lr_size.yz - 1
    );
    uint2 tap_loc4 = clamp(
        uint2((float2(idx.yz + 0.5) + float2(lut.dy.w, lut.dx.w)) * 0.5f),
        uint2(0),
        lr_size.yz - 1
    );

    float4 tap1 = float4(
        MaxHalf(no_diff(colour.load(uint4(idx.x, 0, tap_loc1))) * img_exposure),
        MaxHalf(no_diff(colour.load(uint4(idx.x, 1, tap_loc1))) * img_exposure),
        MaxHalf(no_diff(colour.load(uint4(idx.x, 2, tap_loc1))) * img_exposure),
        1.0f
    );
    float4 tap2 = float4(
        MaxHalf(no_diff(colour.load(uint4(idx.x, 0, tap_loc2))) * img_exposure),
        MaxHalf(no_diff(colour.load(uint4(idx.x, 1, tap_loc2))) * img_exposure),
        MaxHalf(no_diff(colour.load(uint4(idx.x, 2, tap_loc2))) * img_exposure),
        1.0f
    );
    float4 tap3 = float4(
        MaxHalf(no_diff(colour.load(uint4(idx.x, 0, tap_loc3))) * img_exposure),
        MaxHalf(no_diff(colour.load(uint4(idx.x, 1, tap_loc3))) * img_exposure),
        MaxHalf(no_diff(colour.load(uint4(idx.x, 2, tap_loc3))) * img_exposure),
        1.0f
    );
    float4 tap4 = float4(
        MaxHalf(no_diff(colour.load(uint4(idx.x, 0, tap_loc4))) * img_exposure),
        MaxHalf(no_diff(colour.load(uint4(idx.x, 1, tap_loc4))) * img_exposure),
        MaxHalf(no_diff(colour.load(uint4(idx.x, 2, tap_loc4))) * img_exposure),
        1.0f
    );

    float mask = float(lut.dx.x == 0) * float(lut.dy.x == 0);
    float3 colour_to_accum = tap1.rgb * mask;

    float4 accum = (
        (tap1 * kpn_weights.x) +
        (tap2 * kpn_weights.y) +
        (tap3 * kpn_weights.z) +
        (tap4 * kpn_weights.w)
    );

    float3 filtered_colour = accum.rgb * rcp(accum.w);


    // Upsample Learnt Parameters
    float2 params = sample_tensor(temporal_params, idx.x, uint2(0, 1), uv);
    float theta = params.x; // {0 <= x <= 1}
    float alpha = params.y * 0.35 + 0.05; // { 0.05 <= x <= 0.4}

    // Warp history
    int2 input_loc = int2(uv * float2(544.0f, 960.f));
    int2 dilation_offset = DecodeNearestOffset(nearest_offset, input_loc, idx.x);
    uint2 dilated_coord = clamp(
        uint2(input_loc + dilation_offset),
        uint2(0),
        lr_size.yz - 1
    );
    float2 vector = float2(
        no_diff(motion.load(uint4(idx.x, 0, dilated_coord))),
        no_diff(motion.load(uint4(idx.x, 1, dilated_coord)))
    );
    float2 reproj_uv = uv - vector;

    float onscreen = float(all(reproj_uv >= 0.f) && all(reproj_uv < 1.f));
    float3 warped_history = MaxHalf(sample_tensor_catmull(history, idx.x, uint3(0, 1, 2), reproj_uv, true) * img_exposure);

    // Rectify History
    float3 rectified = lerp(filtered_colour, warped_history, theta * onscreen);

    // Accumulate in TM domain
    float3 rectified_tm = tonemap_forward(rectified, TonemapMode.Karis);

    // Accumulate the low resolution sample, if valid, for the current output pixel
    float learnt_masked_alpha = alpha * mask;
    float3 colour_to_accum_tm = tonemap_forward(colour_to_accum, TonemapMode.Karis);
    float3 accumulated = lerp(rectified_tm, colour_to_accum_tm, learnt_masked_alpha);

    // Clamp Output to range that is accepted by invertible tonemapper i.e. `max_val < 1.0`
    const float max_rgb = 1.f - EPS;
    float3 clamped_output = clamp(accumulated, 0.f, max_rgb);
    float3 out_linear = tonemap_inverse(clamped_output, TonemapMode.Karis) * rcp(img_exposure);
    out_linear = simulate_r11g11b10_fakeq(out_linear);

    // Write Output
    output.storeOnce(uint4(idx.x, 0, idx.yz), out_linear.r);
    output.storeOnce(uint4(idx.x, 1, idx.yz), out_linear.g);
    output.storeOnce(uint4(idx.x, 2, idx.yz), out_linear.b);

    // Only used for computing loss -> in inference we don't require this output
    float3 filtered_to_write = filtered_colour * rcp(img_exposure);
    out_filtered.storeOnce(uint4(idx.x, 0, idx.yz), filtered_to_write.r);
    out_filtered.storeOnce(uint4(idx.x, 1, idx.yz), filtered_to_write.g);
    out_filtered.storeOnce(uint4(idx.x, 2, idx.yz), filtered_to_write.b);
}

[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void post_process_v1(
    TensorView<float> colour,
    DiffTensorView history,
    DiffTensorView t_kpn_params,
    DiffTensorView temporal_params,
    TensorView<float> motion,
    TensorView<float> exposure,
    TensorView<float> jitter,
    TensorView<float> offset_lut,
    TensorView<float> scale,
    TensorView<int> idx_modulo,
    DiffTensorView output,
    DiffTensorView out_filtered,
)
{
    // Kernel Window
    const uint kernel_sz = 16;
    const int2 tap_offset[kernel_sz] = {
        int2(-1,-1), int2( 0,-1), int2( 1,-1), int2( 2,-1),
        int2(-1, 0), int2( 0, 0), int2( 1, 0), int2( 2, 0),
        int2(-1, 1), int2( 0, 1), int2( 1, 1), int2( 2, 1),
        int2(-1, 2), int2( 0, 2), int2( 1, 2), int2( 2, 2),
    };

    // Thread location
    uint tId = (cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim()).x;
    uint3 size = tensor_size(output).xzw;
    uint3 idx = tensor_idx(tId, size);
    if (any(idx >= size)) return;

    float2 inv_dims = rcp(size.yz);
    float2 uv = (float2(idx.yz + 0.5f) * inv_dims);

    //!NOTE: assumes same scale factor for height/width
    float scale = no_diff(scale[0]);
    int idx_mod = no_diff(idx_modulo[0]);

    // constants
    uint2 colour_size = tensor_size(colour).zw;
    uint2 output_size = size.yz;

    // Indexes
    int2 base_idx = idx.yz;
    int2 lr_coord = int2(floor((base_idx + 0.5f) * rcp(scale)));

    // Load Exposure
    float img_exposure = no_diff(exposure[uint4(idx.x, 0, 0, 0)]);

    // filtering
    float weight_sum = 0.f;
    float3 filtered_colour = float3(0.f);
    [ForceUnroll]
    for (int i = 0; i < kernel_sz; i++) {
        // Tap position within the 4x4 kernel window
        int2 tap = base_idx + tap_offset[i];

        // Find the relevant jitter details for this tap
        int2 tiled_idx = abs(tap % int2(idx_mod));
        float3 tile = float3(
            no_diff(offset_lut[uint4(idx.x, 0, tiled_idx)]),
            no_diff(offset_lut[uint4(idx.x, 1, tiled_idx)]),
            no_diff(offset_lut[uint4(idx.x, 2, tiled_idx)])
        );

        // Given the jittered coordinate load the corresponding low-res input
        int2 lr_tap = clamp(int2(floor((tap + 0.5f) * rcp(scale))) + int2(tile.xy), 0, colour_size - 1);
        float3 colour_tap = MaxHalf(float3(
            no_diff(colour[uint4(idx.x, 0, lr_tap)]),
            no_diff(colour[uint4(idx.x, 1, lr_tap)]),
            no_diff(colour[uint4(idx.x, 2, lr_tap)]),
        ) * img_exposure);

        // Load KPN Weight
        float weight = max(sample_tensor(t_kpn_params, idx.x, i, uv), EPS);

        // Nullify weights for invalid samples
        weight *= tile.z;

        // accumulate
        filtered_colour += colour_tap * weight;
        weight_sum += weight;
    }
    filtered_colour /= max(weight_sum, EPS);

    // Upsample Learnt Parameters
    float2 params = sample_tensor(temporal_params, idx.x, uint2(0, 1), uv);
    float theta = params.x; // {0 <= x <= 1}
    float alpha = params.y * 0.35 + 0.05; // { 0.05 <= x <= 0.4}

    // Warp history
    float2 vector = float2(no_diff motion[uint4(idx.x, 0, idx.yz)], no_diff motion[uint4(idx.x, 1, idx.yz)]);
    vector *= float(length(vector) > 0.1f);
    float2 reproj_uv = uv - (vector * inv_dims);
    float onscreen = float(all(reproj_uv >= 0.f) && all(reproj_uv <= 1.f));
    float3 warped_history = MaxHalf(sample_tensor_catmull(history, idx.x, uint3(0, 1, 2), reproj_uv) * img_exposure);

    // Rectify History
    float3 rectified = lerp(filtered_colour, warped_history, theta * onscreen);

    // Accumulate in TM domain
    float3 rectified_tm = tonemap_forward(rectified, TonemapMode.Karis);

    // Accumulate the low resolution sample, if valid, for the current output pixel
    int2 tiled_idx = abs(base_idx % int2(idx_mod));
    float3 tile = float3(
        no_diff(offset_lut[uint4(idx.x, 0, tiled_idx)]),
        no_diff(offset_lut[uint4(idx.x, 1, tiled_idx)]),
        no_diff(offset_lut[uint4(idx.x, 2, tiled_idx)])
    );
    int2 lr_tap = clamp(lr_coord + int2(tile.xy), 0, colour_size - 1);
    float3 colour_to_accum = MaxHalf(float3(
        no_diff(colour[uint4(idx.x, 0, lr_tap)]),
        no_diff(colour[uint4(idx.x, 1, lr_tap)]),
        no_diff(colour[uint4(idx.x, 2, lr_tap)]),
    ) * img_exposure);
    float learnt_masked_alpha = alpha * tile.z;
    float3 colour_to_accum_tm = tonemap_forward(colour_to_accum, TonemapMode.Karis);
    float3 accumulated = lerp(rectified_tm, colour_to_accum_tm, learnt_masked_alpha);

    // Clamp Output to range that is accepted by invertible tonemapper i.e. `max_val < 1.0`
    const float max_rgb = 1.f - EPS;
    float3 clamped_output = clamp(accumulated, 0.f, max_rgb);
    float3 out_linear = tonemap_inverse(clamped_output, TonemapMode.Karis) * rcp(img_exposure);

    // Write Output
    output.storeOnce(uint4(idx.x, 0, idx.yz), out_linear.r);
    output.storeOnce(uint4(idx.x, 1, idx.yz), out_linear.g);
    output.storeOnce(uint4(idx.x, 2, idx.yz), out_linear.b);

    // Only used for computing loss -> in inference we don't require this output
    float3 filtered_to_write = filtered_colour * rcp(img_exposure);
    out_filtered.storeOnce(uint4(idx.x, 0, idx.yz), filtered_to_write.r);
    out_filtered.storeOnce(uint4(idx.x, 1, idx.yz), filtered_to_write.g);
    out_filtered.storeOnce(uint4(idx.x, 2, idx.yz), filtered_to_write.b);
}



// TODO: use halfs for precision drop
[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
void pre_process_v1_shader_accurate(
    TensorView<float> colour,
    DiffTensorView history,
    TensorView<float> motion,
    TensorView<float> depth,
    TensorView<float> depth_tm1,
    TensorView<float> nearest_offset_tm1,
    TensorView<float> jitter,
    TensorView<float> jitter_tm1,
    DiffTensorView feedback_tm1,
    TensorView<float> derivative_tm1,
    TensorView<float> depth_params,
    TensorView<float> exposure,
    TensorView<float> render_size,
    DiffTensorView output_tensor,
    TensorView<float> out_luma_derivative,
    TensorView<float> out_depth_t,
    DiffTensorView dm_scale,
)
{
    uint tId = (cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim()).x;

    // We Dispatch across `output`'s dimensions
    uint3 size = tensor_size(output_tensor).xzw;
    uint3 idx = tensor_idx(tId, size);
    if (any(idx >= size)) return;

    float2 uv = (float2(idx.yz + 0.5f) / float2(size.yz));

    // Upscaling factor
    const float2 scale = float2(2.f);
    float2 render_sz = float2(no_diff(render_size[uint4(idx.x, 0, 0, 0)]), no_diff(render_size[uint4(idx.x, 1, 0, 0)]));

    // Dilate Depth
    // ------------------------------------
    float4 device_to_view = float4(
        no_diff(depth_params[uint4(idx.x, 0, 0, 0)]),
        no_diff(depth_params[uint4(idx.x, 1, 0, 0)]),
        no_diff(depth_params[uint4(idx.x, 2, 0, 0)]),
        no_diff(depth_params[uint4(idx.x, 3, 0, 0)]),
    );

    /// Shader Accurate change
    const bool inverted_depth = false;
    float depth_dilated = 0.f;
    int2 nearest_coord_offset = int2(0);
    ShaderAccurateFindNearestDepth(depth, idx.x, uv, inverted_depth, depth_dilated, nearest_coord_offset);
    uint2 nearest_coord = clamp(
        idx.yz + nearest_coord_offset,
        uint2(0),
        uint2(size.yz)-1
    );

    // Calculate UV's for tensor sampling
    // ------------------------------------
    float2 nearest_uv = (float2(nearest_coord) + 0.5f) / float2(size.yz);
    float2 vector = float2(
        no_diff(motion.load(uint4(idx.x, 0, nearest_coord))),
        no_diff(motion.load(uint4(idx.x, 1, nearest_coord)))
    );
    float2 vector_pix = vector * render_sz;
    vector *= float(dot(vector_pix, vector_pix) > (0.1f * 0.1f));
    float2 reproj_uv = uv - vector;

    float2 jit_tm1 = float2(no_diff(jitter_tm1[uint4(idx.x, 0, 0, 0)]), no_diff(jitter_tm1[uint4(idx.x, 1, 0, 0)]));
    float2 unjitter_tm1_uv = reproj_uv - (jit_tm1 / size.yz);

    float2 jit_t = float2(no_diff(jitter[uint4(idx.x, 0, 0, 0)]), no_diff(jitter[uint4(idx.x, 1, 0, 0)]));
    float img_exposure = no_diff(exposure[uint4(idx.x, 0, 0, 0)]);

    // Depth-based Disocclusion Mask
    // ------------------------------------
    float disocclusion_mask = no_diff(ShaderAccurateComputeDepthClip(depth_tm1, nearest_offset_tm1, idx.x, unjitter_tm1_uv, render_sz, inverted_depth, depth_dilated, device_to_view));
    // Small hack -> infinitely far depths are always flagged (sky box, etc.)
    const float furthest_depth = inverted_depth ? 0.f : 1.f;
    disocclusion_mask = depth_dilated == furthest_depth ? 1.f : disocclusion_mask;

    float dis_mask_scale = dot(vector_pix, vector_pix) > (0.5f * 0.5f) ? 1.0 : dm_scale[0];
    disocclusion_mask = disocclusion_mask * dis_mask_scale;

    // Unjitter LR Colour
    // ------------------------------------
    float3 jittered_colour = float3(
        no_diff(colour[uint4(idx.x, 0, idx.yz)]),
        no_diff(colour[uint4(idx.x, 1, idx.yz)]),
        no_diff(colour[uint4(idx.x, 2, idx.yz)]),
    );
    jittered_colour = tonemap_forward(jittered_colour * img_exposure, TonemapMode.Karis);

    // Warp History Buffer @ input resolution to Current Frame `t_tm1` -> `t`
    // ------------------------------------
    float3 lr_warped_history = sample_tensor(history, idx.x, uint3(0, 1, 2), reproj_uv, true);
    lr_warped_history = tonemap_forward(lr_warped_history * img_exposure, TonemapMode.Karis);

    // Calculate Derivative of `luma` - helps identifying high-frequency flicker due to jitter
    // ------------------------------------
    const float derivative_dis_thresh = 0.01; // threshold, doesn't matter too much, but must be > 0.
    const float deriv_pow = 1.5; // increase importance of larger deviations
    const float deriv_min = 0.05; // threshold, removes residual ghosting
    const float deriv_max = 0.3; // threshold, doesn't matter too much, helps allocate precision
    const float deriv_denom = rcp(pow(deriv_max, deriv_pow));
    const float d_alpha = 0.1;

    // Unjittered colour in luma
    float luma_t = luminance(jittered_colour);

    // Warp Previous derivative and luma
    float2 deriv_tm1 = sample_tensor(derivative_tm1, idx.x, uint2(0, 1), reproj_uv, true);
    float luma_derivative_tm1 = deriv_tm1.x;
    float luma_tm1 = deriv_tm1.y;

    // Calculate new derivative
    float luma_derivative_t = abs(luma_t - luma_tm1);

    // Operations applied here:
    //   1) Discard values less than `deriv_min` to reduce ghosting
    //   2) Clip to `deriv_max` which is ~typical max value (allows for better precision allocation when normalized)
    //   3) Apply a power curve, puts more precision towards larger differences
    //   4) Normalize clipped derivative to [0, 1] range
    luma_derivative_t = luma_derivative_t > deriv_min ? min(luma_derivative_t, deriv_max) : 0.f;
    luma_derivative_t = pow(luma_derivative_t, deriv_pow) * deriv_denom;

    // Accumulate the new derivative into the history.
    // We apply an adaptive alpha scaling, to ensure that if a derivative converges to a high value
    // it becomes more difficult to reset that value, this provides temporally stable convergence
    float applied_d_alpha = lerp(d_alpha, d_alpha * 0.1, min(luma_derivative_tm1, deriv_max) * rcp(deriv_max));
    float luma_derivative = lerp(luma_derivative_tm1, luma_derivative_t, applied_d_alpha);

    // Remove disoccluded regions of the derivative
    float disocclusion_binary = float(disocclusion_mask > derivative_dis_thresh);
    luma_derivative = lerp(luma_derivative, 0.f, disocclusion_binary);

    // Warp Feedback and Previous Params to Current Frame `t_tm1` -> `t`
    // ------------------------------------
    float4 feedback = sample_tensor(feedback_tm1, idx.x, uint4(0,1,2,3), reproj_uv, false);

    float encoded_depth_offset = EncodeNearestDepthCoord(nearest_coord_offset);

    // Write Outputs
    // ------------------------------------
    output_tensor.storeOnce(uint4(idx.x, 0, idx.yz), lr_warped_history.r);
    output_tensor.storeOnce(uint4(idx.x, 1, idx.yz), lr_warped_history.g);
    output_tensor.storeOnce(uint4(idx.x, 2, idx.yz), lr_warped_history.b);
    output_tensor.storeOnce(uint4(idx.x, 3, idx.yz), jittered_colour.r);
    output_tensor.storeOnce(uint4(idx.x, 4, idx.yz), jittered_colour.g);
    output_tensor.storeOnce(uint4(idx.x, 5, idx.yz), jittered_colour.b);
    output_tensor.storeOnce(uint4(idx.x, 6, idx.yz), disocclusion_mask);
    output_tensor.storeOnce(uint4(idx.x, 7, idx.yz), feedback.r);
    output_tensor.storeOnce(uint4(idx.x, 8, idx.yz), feedback.g);
    output_tensor.storeOnce(uint4(idx.x, 9, idx.yz), feedback.b);
    output_tensor.storeOnce(uint4(idx.x, 10, idx.yz), feedback.a);
    output_tensor.storeOnce(uint4(idx.x, 11, idx.yz), luma_derivative);

    out_luma_derivative.store(uint4(idx.x, 0, idx.yz), luma_derivative);
    out_luma_derivative.store(uint4(idx.x, 1, idx.yz), luma_t);

    out_depth_t.store(uint4(idx.x, 0, idx.yz), encoded_depth_offset);
}
