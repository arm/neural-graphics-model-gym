// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0

import "../../../../common/model/shaders/image";
import "../../../../common/model/shaders/tensor";
import "../../../../common/model/shaders/tonemap";

import "nss_common";

#ifndef NSS_USE_HISTORY_CATMULL
#ifdef USE_BILINEAR_HISTORY_SAMPLING
#define NSS_USE_HISTORY_CATMULL 0
#else
#define NSS_USE_HISTORY_CATMULL 1
#endif // USE_BILINEAR_HISTORY_SAMPLING
#endif // !NSS_USE_HISTORY_CATMULL
#ifndef NSS_USE_SPARSE_2X2_FILTER
#define NSS_USE_SPARSE_2X2_FILTER 0
#endif // !NSS_USE_SPARSE_2X2_FILTER
#ifndef NSS_PREPROCESS_HALF_RES_INPUT
#define NSS_PREPROCESS_HALF_RES_INPUT 0
#endif // !NSS_PREPROCESS_HALF_RES_INPUT
#ifndef NSS_PACKED_NEAREST_OFFSET_QUAD
#define NSS_PACKED_NEAREST_OFFSET_QUAD 0
#endif // !NSS_PACKED_NEAREST_OFFSET_QUAD
#ifndef NSS_DISABLE_HISTORY_CLAMP
#ifdef DISABLE_HISTORY_CLAMP
#define NSS_DISABLE_HISTORY_CLAMP 1
#else
#define NSS_DISABLE_HISTORY_CLAMP 0
#endif // DISABLE_HISTORY_CLAMP
#endif // !NSS_DISABLE_HISTORY_CLAMP
#ifndef FILTER_COLOUR_KERNEL_SZ
#define FILTER_COLOUR_KERNEL_SZ 9
#endif // !FILTER_COLOUR_KERNEL_SZ

#define EPS 1e-7

// ─────────────────────────────────────────────────────────────────────────────
// Argument plumbing
// ─────────────────────────────────────────────────────────────────────────────
#define POSTPROCESS_PARAMS(X)                   \
    X(TensorView<float>, in_colour),            \
    X(DiffTensorView,   in_history),            \
    X(DiffTensorView,   in_kpn_params),         \
    X(DiffTensorView,   in_temporal_params),    \
    X(TensorView<float>, in_motion),            \
    X(TensorView<float>, in_nearest_depth_off), \
    X(TensorView<float>, in_exposure),          \
    X(TensorView<float>, in_jitter),            \
    X(TensorView<float>, in_offset_lut),        \
    X(TensorView<float>, in_idx_modulo),        \
    X(TensorView<float>, in_reset),             \
    X(DiffTensorView,   out_colour),            \
    X(DiffTensorView,   out_colour_filtered)

#define DECL(type, name) type name
#define NAME_ONLY(_, name) name
#define POSTPROCESS_ARGS POSTPROCESS_PARAMS(NAME_ONLY)


struct Bindings {
    TensorView<float> _ColourTex;
    DiffTensorView _HistoryTex;
    DiffTensorView _KpnParamsTensor;
    DiffTensorView _TemporalParamsTex;
    TensorView<float> _MotionVectorTex;
    TensorView<float> _NearestDepthOffsetTex;
    TensorView<float> _OffsetLutTex;

    DiffTensorView _ColourOut;
    DiffTensorView _ColourFilteredOut;
};


struct PushConsts {
    // High-res output extent divided by low-res input extent.
    float2 _Scale;
    // Inverse of `_Scale`.
    float2 _InvScale;
    // High-resolution output dimensions.
    int32_t2 _OutputDims;
    // Low-resolution input dimensions.
    int32_t2 _InputDims;
    // KPN tensor dimensions.
    int32_t2 _KpnDims;
    // Logical preprocess dimensions used to address temporal feedback.
    int32_t2 _PreprocessDims;
    // Inverse of `_OutputDims`.
    float2 _InvOutputDims;
    // Inverse of `_InputDims`.
    float2 _InvInputDims;
    // Inverse of `_KpnDims`.
    float2 _InvKpnDims;
    // UV scale from preprocess space into temporal-feedback space.
    float2 _PaddedUvScale;
    // UV scale from temporal-feedback space into KPN space.
    float2 _KpnScale;
    // Tile modulo used for offset-LUT pattern selection.
    int32_t2 _IdxModulo;
    // Exposure packed as (exposure, 1/exposure).
    float2 _Exposure;
    // Temporal reset / history blending gate.
    float _Reset;
};


struct Globals {
    Bindings b;
    PushConsts pc;
};


[Differentiable]
Globals makeGlobals(POSTPROCESS_PARAMS(DECL), uint batch)
{
    Globals g;

    g.b._ColourTex = in_colour;
    g.b._HistoryTex = in_history;
    g.b._KpnParamsTensor = in_kpn_params;
    g.b._TemporalParamsTex = in_temporal_params;
    g.b._MotionVectorTex = in_motion;
    g.b._NearestDepthOffsetTex = in_nearest_depth_off;
    g.b._OffsetLutTex = in_offset_lut;
    g.b._ColourOut = out_colour;
    g.b._ColourFilteredOut = out_colour_filtered;

    const int2 in_dims = int2(tensor_size(in_colour).zw);
    const int2 out_dims = int2(tensor_size(out_colour).zw);
    const int2 kpn_dims = int2(tensor_size(in_kpn_params).zw);
    const int2 temporal_dims = int2(tensor_size(in_temporal_params).zw);
    const float2 inv_temporal_dims = rcp(float2(temporal_dims));

    g.pc._InputDims = in_dims;
    g.pc._OutputDims = out_dims;
    g.pc._KpnDims = kpn_dims;
    g.pc._Scale = float2(out_dims) / float2(in_dims);
    g.pc._InvScale = rcp(g.pc._Scale);
    g.pc._InvOutputDims = rcp(float2(out_dims));
    g.pc._InvInputDims = rcp(float2(in_dims));
    g.pc._InvKpnDims = rcp(float2(kpn_dims));
#if NSS_PREPROCESS_HALF_RES_INPUT
    g.pc._IdxModulo = int2(
        int(no_diff(in_idx_modulo[int4(batch, 0, 0, 0)])),
        int(no_diff(in_idx_modulo[int4(batch, 1, 0, 0)]))
    );
    g.pc._PreprocessDims = int2(
        int(no_diff(in_idx_modulo[int4(batch, 2, 0, 0)])),
        int(no_diff(in_idx_modulo[int4(batch, 3, 0, 0)]))
    );
#else
    g.pc._IdxModulo = int2(
        int(no_diff(in_idx_modulo[int4(0, 0, 0, 0)])),
        int(no_diff(in_idx_modulo[int4(0, 0, 0, 1)]))
    );
    g.pc._PreprocessDims = in_dims;
#endif // NSS_PREPROCESS_HALF_RES_INPUT
    // Legacy `_PaddedUvScale` name: this is preprocess->temporal UV scaling,
    // not physical padded extent scaling.
    g.pc._PaddedUvScale = float2(g.pc._PreprocessDims) * inv_temporal_dims;
    g.pc._KpnScale = float2(kpn_dims) / float2(temporal_dims);

    const float exposure = no_diff(in_exposure[uint4(batch, 0, 0, 0)]);
    g.pc._Exposure = float2(exposure, rcp(exposure));
    g.pc._Reset = no_diff(in_reset[uint4(batch, 0, 0, 0)]);

    return g;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
#ifdef SHADER_ACCURATE

[Differentiable]
inline float2 LoadMotion(Globals g, uint b, uint2 output_coord)
{
    int2 input_coord = int2(float2(output_coord) * g.pc._InvScale);
    int2 dilation_offset = int2(0);
#if NSS_PREPROCESS_HALF_RES_INPUT
    int2 texel = clamp(
        int2(floor((float2(input_coord) + 0.5f) * float2(g.pc._PreprocessDims) * g.pc._InvInputDims)),
        int2(0),
        g.pc._PreprocessDims - 1
    );
    #if NSS_PACKED_NEAREST_OFFSET_QUAD
    float2 norm_code_pair = float2(
        no_diff(g.b._NearestDepthOffsetTex[uint4(b, 0, uint2(texel))]),
        no_diff(g.b._NearestDepthOffsetTex[uint4(b, 1, uint2(texel))])
    );
    int lane_idx = (input_coord.x & 1) * 2 + (input_coord.y & 1);
    int packed_byte = lane_idx < 2 ? int(norm_code_pair.x * 255.0f + 0.5f) : int(norm_code_pair.y * 255.0f + 0.5f);
    int packed_nibble = (lane_idx & 1) == 0 ? (packed_byte & 0xF) : ((packed_byte >> 4) & 0xF);
    dilation_offset = DecodeNearestDepthCoordNibble(packed_nibble);
    #else
    int code = int(no_diff(g.b._NearestDepthOffsetTex[uint4(b, 0, uint2(texel))]) * 255.0f + 0.5f);
    dilation_offset = int2((code & 0x7) - 2, ((code >> 3) & 0x7) - 2);
    #endif // NSS_PACKED_NEAREST_OFFSET_QUAD
#else
    dilation_offset = DecodeNearestOffset(g.b._NearestDepthOffsetTex, input_coord, b);
#endif // NSS_PREPROCESS_HALF_RES_INPUT
    uint2 dilated_coord = uint2(clamp(input_coord + dilation_offset, int2(0), g.pc._InputDims - 1));
    float2 v = float2(
        no_diff(g.b._MotionVectorTex[uint4(b, 0, dilated_coord)]),
        no_diff(g.b._MotionVectorTex[uint4(b, 1, dilated_coord)])
    );
    v *= g.pc._Scale;
    v *= float(length(v) > 0.1f);
    return v;
}

#else

[Differentiable]
inline float2 LoadMotion(Globals g, uint b, uint2 coord)
{
    float2 v = float2(
        no_diff(g.b._MotionVectorTex[uint4(b, 0, coord)]),
        no_diff(g.b._MotionVectorTex[uint4(b, 1, coord)])
    );
    v *= float(length(v) > 0.1f);
    return v;
}

#endif // SHADER_ACCURATE


[Differentiable]
inline float3 LoadColourTap(Globals g, uint b, int2 lr_tap, float exposure)
{
    return MaxHalf(float3(
        no_diff(g.b._ColourTex[uint4(b, 0, lr_tap)]),
        no_diff(g.b._ColourTex[uint4(b, 1, lr_tap)]),
        no_diff(g.b._ColourTex[uint4(b, 2, lr_tap)])
    ) * exposure);
}


[Differentiable]
inline float SampleKpnWeight(Globals g, uint b, int tap_z, int2 kpn_tap)
{
    int channel = clamp(tap_z, 0, int(tensor_size(g.b._KpnParamsTensor).y) - 1);
    return max(g.b._KpnParamsTensor.load(uint4(b, channel, uint2(kpn_tap))), EPS);
}


[NoDiffThis]
inline int KpnCoordFromInputCoord(int input_coord, int input_dim, int kpn_dim)
{
    int safe_input_dim = max(input_dim, 1);
    return min((input_coord * kpn_dim) / safe_input_dim, kpn_dim - 1);
}


[Differentiable]
inline float3 LoadWarpedHistory(Globals g, uint b, float2 uv, float exposure)
{
#if NSS_USE_HISTORY_CATMULL
    float3 h = sample_tensor_catmull(g.b._HistoryTex, b, uint3(0, 1, 2), uv);
#else
    float3 h = sample_tensor(g.b._HistoryTex, b, uint3(0, 1, 2), uv, false);
#endif // NSS_USE_HISTORY_CATMULL
    return MaxHalf(h * exposure);
}


static const int kSparsePrunedChannelCount = 36;
static const int kSparsePrunedChannelMap[kSparsePrunedChannelCount] = {
    -1, -1, -1, -1, -1, -1,
    -1, 0, 1, 2, 3, -1,
    -1, 4, 5, 6, 7, -1,
    -1, 8, 9, 10, 11, -1,
    -1, 12, 13, 14, 15, -1,
    -1, -1, -1, -1, -1, -1,
};


[NoDiffThis]
inline int MapSparse2x2KpnChannel(int tap_z)
{
    if (tap_z < 0 || tap_z >= kSparsePrunedChannelCount) {
        return -1;
    }
    return kSparsePrunedChannelMap[tap_z];
}


[Differentiable]
inline void FilterColour(
    Globals g,
    uint b,
    int2 output_px,
    float2 uv,
    float exposure,
    out float3 m1,
    out float3 m2,
    out float4 center_sample
)
{
    const int2 colour_size = g.pc._InputDims;
    const int2 kpn_size = g.pc._KpnDims;
    const float2 inv_scale = g.pc._InvScale;
    const float2 kpn_scale = g.pc._KpnScale;

    const int2 idx_mod = g.pc._IdxModulo;
    int2 tile_idx = abs(output_px % idx_mod);
    int lut_idx = tile_idx.x * idx_mod.y + tile_idx.y;

    float weight_sum = 0.f;
    m1 = float3(0.f);
    m2 = float3(0.f);
    center_sample = float4(0.f);

    [ForceUnroll]
    for (int i = 0; i < FILTER_COLOUR_KERNEL_SZ; i++) {
        float3 tile = float3(
            no_diff(g.b._OffsetLutTex[uint4(b, 0, lut_idx, i)]),
            no_diff(g.b._OffsetLutTex[uint4(b, 1, lut_idx, i)]),
            no_diff(g.b._OffsetLutTex[uint4(b, 2, lut_idx, i)])
        );
        float3 tap_pos = float3(
            no_diff(g.b._OffsetLutTex[uint4(b, 3, lut_idx, i)]),
            no_diff(g.b._OffsetLutTex[uint4(b, 4, lut_idx, i)]),
            no_diff(g.b._OffsetLutTex[uint4(b, 5, lut_idx, i)])
        );

        int2 tap = output_px + int2(tap_pos.xy);
        int2 lr_tap = clamp(int2(floor((tap + 0.5f) * inv_scale + 1e-3)) + int2(tile.xy), 0, colour_size - 1);
        float is_center = float(tap_pos.x == 0.f && tap_pos.y == 0.f);
        float3 colour_tap = LoadColourTap(g, b, lr_tap, exposure);
        center_sample = lerp(center_sample, float4(colour_tap, 1.f), is_center);

#if NSS_USE_SPARSE_2X2_FILTER || NSS_PREPROCESS_HALF_RES_INPUT
        int2 kpn_tap = clamp(
            int2(
                KpnCoordFromInputCoord(lr_tap.x, colour_size.x, kpn_size.x),
                KpnCoordFromInputCoord(lr_tap.y, colour_size.y, kpn_size.y)
            ),
            int2(0),
            kpn_size - 1
        );
#else
        int2 kpn_tap = clamp(int2(floor((float2(lr_tap) + 0.5f + 1e-3) * kpn_scale)), 0, kpn_size - 1);
#endif // NSS_USE_SPARSE_2X2_FILTER || NSS_PREPROCESS_HALF_RES_INPUT

#if NSS_USE_SPARSE_2X2_FILTER
        int pruned_channel = MapSparse2x2KpnChannel(int(tap_pos.z));
        float weight = pruned_channel >= 0 ? SampleKpnWeight(g, b, pruned_channel, kpn_tap) : 0.f;
#else
        float weight = SampleKpnWeight(g, b, int(tap_pos.z), kpn_tap);
#endif // NSS_USE_SPARSE_2X2_FILTER
        weight *= tile.z;

        m1 += colour_tap * weight;
#if !NSS_DISABLE_HISTORY_CLAMP
        m2 += colour_tap * colour_tap * weight;
#endif // !NSS_DISABLE_HISTORY_CLAMP
        weight_sum += weight;
    }

    m1 /= max(weight_sum, EPS);
#if !NSS_DISABLE_HISTORY_CLAMP
    m2 /= max(weight_sum, EPS);
#endif // !NSS_DISABLE_HISTORY_CLAMP
}


[Differentiable]
inline void SampleTemporalParams(
    Globals g,
    uint b,
    float2 uv,
    out float theta,
    out float alpha,
    out float gamma
)
{
    float2 uv_temporal = uv * g.pc._PaddedUvScale;
#if NSS_DISABLE_HISTORY_CLAMP
    float2 params = sample_tensor(g.b._TemporalParamsTex, b, uint2(0, 1), uv_temporal);
    gamma = 0.f;
#else
    float3 params = sample_tensor(g.b._TemporalParamsTex, b, uint3(0, 1, 2), uv_temporal);
    gamma = params.z * 2.f;
#endif // NSS_DISABLE_HISTORY_CLAMP
    theta = params.x;
    alpha = params.y * 0.35 + 0.05;
}


[Differentiable]
inline void WarpHistory(
    Globals g,
    uint b,
    int2 output_px,
    float2 uv,
    float exposure,
    out float3 warped_colour,
    out float onscreen
)
{
    const float2 inv_out_size = g.pc._InvOutputDims;
    float2 motion = LoadMotion(g, b, uint2(output_px));
    float2 reproj_uv = uv - (motion * inv_out_size);
    onscreen = float(all(reproj_uv >= 0.f) && all(reproj_uv <= 1.f));
    warped_colour = LoadWarpedHistory(g, b, reproj_uv, exposure);
}


[Differentiable]
inline void ClampHistoryToStats(
    float3 m1,
    float3 m2,
    float3 warped_history,
    float theta,
    float gamma,
    float reset,
    float onscreen,
    out float3 rectified
)
{
#if NSS_DISABLE_HISTORY_CLAMP
    rectified = lerp(m1, warped_history, theta * onscreen * reset);
#else
    float3 sigma = sqrt(max(abs(m2 - m1 * m1), EPS)) * gamma;
    float3 aabb_min = m1 - sigma;
    float3 aabb_max = m1 + sigma;
    float3 history_clamped = lerp(m1, clamp(warped_history, aabb_min, aabb_max), reset);
    rectified = lerp(history_clamped, warped_history, theta * onscreen * reset);
#endif // NSS_DISABLE_HISTORY_CLAMP
}


[Differentiable]
inline float3 ClampToInvertibleRange(Globals g, float3 accumulated)
{
    const float max_rgb = 1.f - EPS;
    float3 clamped_output = clamp(accumulated, 0.f, max_rgb);
    return tonemap_inverse(clamped_output, TonemapMode.Karis) * g.pc._Exposure.y;
}


[Differentiable]
inline void WriteColourOut(Globals g, uint3 idx, float3 out_linear)
{
    g.b._ColourOut.storeOnce(uint4(idx.x, 0, idx.yz), out_linear.r);
    g.b._ColourOut.storeOnce(uint4(idx.x, 1, idx.yz), out_linear.g);
    g.b._ColourOut.storeOnce(uint4(idx.x, 2, idx.yz), out_linear.b);
}


[Differentiable]
inline void WriteColourFilteredOut(Globals g, uint3 idx, float3 filtered)
{
    g.b._ColourFilteredOut.storeOnce(uint4(idx.x, 0, idx.yz), filtered.r);
    g.b._ColourFilteredOut.storeOnce(uint4(idx.x, 1, idx.yz), filtered.g);
    g.b._ColourFilteredOut.storeOnce(uint4(idx.x, 2, idx.yz), filtered.b);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernels
// ─────────────────────────────────────────────────────────────────────────────
[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
public void post_process(POSTPROCESS_PARAMS(DECL))
{
    uint tId = (cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim()).x;
    uint3 size = tensor_size(out_colour).xzw;
    uint3 idx = tensor_idx(tId, size);
    if (any(idx >= size)) return;

    Globals g = makeGlobals(POSTPROCESS_ARGS, idx.x);

    const float2 inv_out_size = g.pc._InvOutputDims;
    const float exposure = g.pc._Exposure.x;
    const float reset = g.pc._Reset;

    float2 uv = (float2(idx.yz) + 0.5f) * inv_out_size;
    int2 output_px = idx.yz;

    // Filtering
    float3 m1;
    float3 m2;
    float4 center_sample;
    FilterColour(g, idx.x, output_px, uv, exposure, m1, m2, center_sample);

    // Upsample temporal params (may be padded)
    float theta;
    float alpha;
    float gamma;
    SampleTemporalParams(g, idx.x, uv, theta, alpha, gamma);

    // Warp history
    float3 warped_colour;
    float onscreen;
    WarpHistory(g, idx.x, output_px, uv, exposure, warped_colour, onscreen);

    // Clamp history to neighbourhood stats
    float3 rectified;
    ClampHistoryToStats(m1, m2, warped_colour, theta, gamma, reset, onscreen, rectified);

    // Accumulate in TM domain
    float3 rectified_tm = tonemap_forward(rectified, TonemapMode.Karis);
    float learnt_masked_alpha = alpha * center_sample.a * reset;
    float3 colour_to_accum_tm = tonemap_forward(center_sample.rgb, TonemapMode.Karis);
    float3 accumulated = lerp(rectified_tm, colour_to_accum_tm, learnt_masked_alpha);

    // Clamp to invertible range and write
    float3 out_linear = ClampToInvertibleRange(g, accumulated);

    // Write Output(s)
    WriteColourOut(g, idx, out_linear);

    float3 filtered_to_write = m1 * g.pc._Exposure.y;
    WriteColourFilteredOut(g, idx, filtered_to_write);
}

#undef POSTPROCESS_PARAMS
#undef POSTPROCESS_ARGS
#undef DECL
