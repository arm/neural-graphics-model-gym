// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0

import "../../../../common/model/shaders/image";
import "../../../../common/model/shaders/tensor";
import "../../../../common/model/shaders/tonemap";

import "nss_common";

#ifndef NSS_PREPROCESS_HALF_RES_INPUT
#define NSS_PREPROCESS_HALF_RES_INPUT 0
#endif // !NSS_PREPROCESS_HALF_RES_INPUT
#ifndef NSS_PACKED_NEAREST_OFFSET_QUAD
#define NSS_PACKED_NEAREST_OFFSET_QUAD 0
#endif // !NSS_PACKED_NEAREST_OFFSET_QUAD
#ifndef NSS_V1_LUMA_DERIVATIVE
#define NSS_V1_LUMA_DERIVATIVE 1
#endif // !NSS_V1_LUMA_DERIVATIVE

#define EPS 1e-7
#define DEPTH_SCALE 2147483647.0
#define INV_DEPTH_SCALE (1.0 / DEPTH_SCALE)
#if !NSS_V1_LUMA_DERIVATIVE
#define DERIVATIVE_DIS_THRESH 0.01
#define DERIV_MIN 0.05
#define DERIV_MAX 0.3
#define DERIV_ALPHA 0.1
#endif // !NSS_V1_LUMA_DERIVATIVE


[NoDiffThis]
inline bool IsOnScreen(int2 pos, int2 size)
{
    return all(pos < size) && all(pos >= int2(0));
}

// ─────────────────────────────────────────────────────────────────────────────
// Argument plumbing
// ─────────────────────────────────────────────────────────────────────────────
#define PREPROCESS_PARAMS(X)                    \
    X(TensorView<float>, in_colour),            \
    X(DiffTensorView,   in_history),            \
    X(TensorView<float>, in_motion),            \
    X(TensorView<float>, in_depth),             \
    X(TensorView<int>,  in_depth_tm1),          \
    X(TensorView<float>, in_jitter),            \
    X(TensorView<float>, in_jitter_tm1),        \
    X(DiffTensorView,   in_feedback_tm1),       \
    X(TensorView<float>, in_derivative_tm1),    \
    X(TensorView<float>, in_depth_params),      \
    X(TensorView<float>, in_exposure),          \
    X(TensorView<float>, in_render_size),       \
    X(DiffTensorView,   out_tensor),            \
    X(TensorView<float>, out_luma_derivative),  \
    X(TensorView<float>, out_disocclusion_mask),\
    X(TensorView<float>, out_nearest_depth_off)

#define DECL(type, name) type name
#define NAME_ONLY(_, name) name
#define PREPROCESS_ARGS PREPROCESS_PARAMS(NAME_ONLY)


struct Bindings {
    TensorView<float> _ColourTex;
    TensorView<float> _DepthTex;
    TensorView<float> _MotionVectorTex;
    DiffTensorView _HistoryTex;
    DiffTensorView _FeedbackTensor;
    TensorView<int> _DepthTm1Tex;
    TensorView<float> _LumaDerivTm1Tex;

    DiffTensorView _InputTensor;
    TensorView<float> _LumaDeriv;
    TensorView<float> _DisocclusionMask;
    TensorView<float> _NearestDepthOffset;
};


struct PushConsts {
    // Projection params used to reconstruct view-space depth/position.
    float4 _DeviceToViewDepth;
    // Current-frame jitter as (x, y, x/input_w, y/input_h).
    float4 _JitterOffset;
    // Previous-frame jitter in the same packed layout as `_JitterOffset`.
    float4 _JitterOffsetTm1;
    // Logical preprocess extent divided by low-res input extent.
    float2 _Scale;
    // Inverse of `_Scale`.
    float2 _InvScale;
    // Logical pre-process output dimensions.
    int32_t2 _OutputDims;
    // Low-resolution input dimensions.
    int32_t2 _InputDims;
    // Padded graph/input-tensor dimensions.
    int32_t2 _PaddedDims;
    // Inverse of `_OutputDims`.
    float2 _InvOutputDims;
    // Inverse of `_InputDims`.
    float2 _InvInputDims;
    // Inverse of `_PaddedDims`.
    float2 _InvPaddedDims;
    // Inverse dimensions of the previous-depth feedback surface.
    float2 _InvDepthTm1Dims;
    // Render-resolution dimensions used by depth-clip shaping.
    float2 _RenderSize;
    // Exposure packed as (exposure, 1/exposure).
    float2 _Exposure;
};


struct Globals {
    Bindings b;
    PushConsts pc;
};

[Differentiable]
Globals makeGlobals(PREPROCESS_PARAMS(DECL), uint batch)
{
    Globals g;

    g.b._ColourTex = in_colour;
    g.b._DepthTex = in_depth;
    g.b._MotionVectorTex = in_motion;
    g.b._HistoryTex = in_history;
    g.b._FeedbackTensor = in_feedback_tm1;
    g.b._DepthTm1Tex = in_depth_tm1;
    g.b._LumaDerivTm1Tex = in_derivative_tm1;
    g.b._InputTensor = out_tensor;
    g.b._LumaDeriv = out_luma_derivative;
    g.b._DisocclusionMask = out_disocclusion_mask;
    g.b._NearestDepthOffset = out_nearest_depth_off;

    const int2 low_res_size = int2(tensor_size(in_colour).zw);
    const int2 process_size = int2(tensor_size(out_disocclusion_mask).zw);
    const int2 pad_res_size = int2(tensor_size(out_tensor).zw);
    const int2 depth_tm1_size = int2(tensor_size(in_depth_tm1).zw);

    const float2 jitter_tm1 = float2(
        no_diff(in_jitter_tm1[uint4(batch, 0, 0, 0)]),
        no_diff(in_jitter_tm1[uint4(batch, 1, 0, 0)])
    );
    const float2 jitter_t = float2(
        no_diff(in_jitter[uint4(batch, 0, 0, 0)]),
        no_diff(in_jitter[uint4(batch, 1, 0, 0)])
    );
    const float exposure = no_diff(in_exposure[uint4(batch, 0, 0, 0)]);

    g.pc._DeviceToViewDepth = float4(
        no_diff(in_depth_params[uint4(batch, 0, 0, 0)]),
        no_diff(in_depth_params[uint4(batch, 1, 0, 0)]),
        no_diff(in_depth_params[uint4(batch, 2, 0, 0)]),
        no_diff(in_depth_params[uint4(batch, 3, 0, 0)])
    );
#if NSS_PREPROCESS_HALF_RES_INPUT
    g.pc._Scale = float2(process_size) / float2(low_res_size);
    g.pc._OutputDims = process_size;
#else
    const int2 high_res_size = int2(tensor_size(in_history).zw);
    g.pc._Scale = float2(high_res_size) / float2(low_res_size);
    g.pc._OutputDims = high_res_size;
#endif // NSS_PREPROCESS_HALF_RES_INPUT
    g.pc._InvScale = rcp(g.pc._Scale);
    g.pc._InputDims = low_res_size;
    g.pc._PaddedDims = pad_res_size;
    g.pc._InvOutputDims = rcp(float2(g.pc._OutputDims));
    g.pc._InvInputDims = rcp(float2(low_res_size));
    g.pc._InvPaddedDims = rcp(float2(pad_res_size));
    g.pc._InvDepthTm1Dims = rcp(float2(depth_tm1_size));
    g.pc._RenderSize = float2(
        no_diff(in_render_size[uint4(batch, 0, 0, 0)]),
        no_diff(in_render_size[uint4(batch, 1, 0, 0)])
    );
    g.pc._Exposure = float2(exposure, rcp(exposure));
    g.pc._JitterOffset = float4(
        jitter_t,
        jitter_t * g.pc._InvInputDims
    );
    g.pc._JitterOffsetTm1 = float4(
        jitter_tm1,
        jitter_tm1 * g.pc._InvInputDims
    );

    return g;
}


// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
[NoDiffThis]
int ReflectIndex(int coord, int size)
{
    return coord < 0 ? -coord - 1 : (coord >= size ? (2 * size - coord - 1) : coord);
}


[NoDiffThis]
int2 ReflectIndex(int2 coord, int2 size)
{
    return int2(ReflectIndex(coord.x, size.x), ReflectIndex(coord.y, size.y));
}


[NoDiffThis]
inline float GetViewSpaceDepth(float depth, float4 device_to_view)
{
    return device_to_view.y / (depth - device_to_view.x);
}


[NoDiffThis]
inline float ComputeDepthClipRequiredSepScale(Globals g)
{
    float half_viewport = length(g.pc._RenderSize);
    float resolution_factor = clamp(
        half_viewport / length(float2(1080.0f, 1920.0f)),
        0.0f,
        1.0f
    );
    float3 corner_view_dir = float3(-g.pc._DeviceToViewDepth.z, g.pc._DeviceToViewDepth.w, 1.0f);
    return 1.37e-05f * length(corner_view_dir) * half_viewport;
}


[NoDiffThis]
inline float ComputeDepthClipPower(Globals g)
{
    float half_viewport = length(g.pc._RenderSize);
    float resolution_factor = clamp(
        half_viewport / length(float2(1080.0f, 1920.0f)),
        0.0f,
        1.0f
    );
    return 1.0f + 2.0f * resolution_factor;
}


[NoDiffThis]
inline void UpdateNearestDepth(
    int2 sample_pos,
    int2 px,
    int2 size,
    float sample_depth,
    inout float nearest_depth,
    inout float2 nearest_offset)
{
    float take = float(IsOnScreen(sample_pos, size)) * step(sample_depth, nearest_depth);
    nearest_depth = lerp(nearest_depth, sample_depth, take);
    nearest_offset = lerp(nearest_offset, float2(sample_pos - px), take);
}


[NoDiffThis]
inline void FindNearestDepth_4x4_FromPixel(
    Globals g,
    uint b,
    int2 px,
    out float nearest_depth,
    out uint2 nearest_coord,
    out int2 nearest_offset)
{
    int2 size = g.pc._InputDims;
    int2 clamp_max = size - 1;

    float nearest_depth_local = 0.0f;
    float2 nearest_offset_f = float2(0.0f);

    #define NSS_LOAD_NEAREST_DEPTH(POS) g.b._DepthTex[uint4(b, 0, uint2(clamp((POS), int2(0), clamp_max)))]

    nearest_depth_local = NSS_LOAD_NEAREST_DEPTH(px);
    UpdateNearestDepth(px + int2(0, -1), px, size, NSS_LOAD_NEAREST_DEPTH(px + int2(0, -1)), nearest_depth_local, nearest_offset_f);
    UpdateNearestDepth(px + int2(-1, 0), px, size, NSS_LOAD_NEAREST_DEPTH(px + int2(-1, 0)), nearest_depth_local, nearest_offset_f);
    UpdateNearestDepth(px + int2(-1, -1), px, size, NSS_LOAD_NEAREST_DEPTH(px + int2(-1, -1)), nearest_depth_local, nearest_offset_f);

    UpdateNearestDepth(px + int2(0, 1), px, size, NSS_LOAD_NEAREST_DEPTH(px + int2(0, 1)), nearest_depth_local, nearest_offset_f);
    UpdateNearestDepth(px + int2(-1, 1), px, size, NSS_LOAD_NEAREST_DEPTH(px + int2(-1, 1)), nearest_depth_local, nearest_offset_f);
    UpdateNearestDepth(px + int2(-1, 2), px, size, NSS_LOAD_NEAREST_DEPTH(px + int2(-1, 2)), nearest_depth_local, nearest_offset_f);
    UpdateNearestDepth(px + int2(0, 2), px, size, NSS_LOAD_NEAREST_DEPTH(px + int2(0, 2)), nearest_depth_local, nearest_offset_f);

    UpdateNearestDepth(px + int2(1, 0), px, size, NSS_LOAD_NEAREST_DEPTH(px + int2(1, 0)), nearest_depth_local, nearest_offset_f);
    UpdateNearestDepth(px + int2(1, -1), px, size, NSS_LOAD_NEAREST_DEPTH(px + int2(1, -1)), nearest_depth_local, nearest_offset_f);
    UpdateNearestDepth(px + int2(2, 0), px, size, NSS_LOAD_NEAREST_DEPTH(px + int2(2, 0)), nearest_depth_local, nearest_offset_f);
    UpdateNearestDepth(px + int2(2, -1), px, size, NSS_LOAD_NEAREST_DEPTH(px + int2(2, -1)), nearest_depth_local, nearest_offset_f);

    UpdateNearestDepth(px + int2(1, 1), px, size, NSS_LOAD_NEAREST_DEPTH(px + int2(1, 1)), nearest_depth_local, nearest_offset_f);
    UpdateNearestDepth(px + int2(1, 2), px, size, NSS_LOAD_NEAREST_DEPTH(px + int2(1, 2)), nearest_depth_local, nearest_offset_f);
    UpdateNearestDepth(px + int2(2, 1), px, size, NSS_LOAD_NEAREST_DEPTH(px + int2(2, 1)), nearest_depth_local, nearest_offset_f);
    UpdateNearestDepth(px + int2(2, 2), px, size, NSS_LOAD_NEAREST_DEPTH(px + int2(2, 2)), nearest_depth_local, nearest_offset_f);

    #undef NSS_LOAD_NEAREST_DEPTH

    nearest_depth = nearest_depth_local;
    nearest_offset = int2(nearest_offset_f);
    nearest_coord = uint2(clamp(px + nearest_offset, int2(0), size - 1));
}


[NoDiffThis]
inline void FindNearestDepthLQ(
    Globals g,
    uint b,
    float2 uv,
    out float nearest_depth,
    out uint2 nearest_coord,
    out int2 nearest_offset)
{
    int2 px = clamp(int2(uv * float2(g.pc._InputDims)), int2(0), g.pc._InputDims - 1);
    FindNearestDepth_4x4_FromPixel(g, b, px, nearest_depth, nearest_coord, nearest_offset);
}


[Differentiable]
inline float2 LoadMotion(Globals g, uint b, float2 uv)
{
    float2 v = sample_tensor(g.b._MotionVectorTex, b, uint2(0, 1), uv);
#ifndef SHADER_ACCURATE
    v *= g.pc._InvScale;
#endif // !SHADER_ACCURATE
    v *= float(length(v) > 0.1f);
    return v;
}


[Differentiable]
inline float2 LoadMotionAtPixel(Globals g, uint b, int2 pixel)
{
    pixel = clamp(pixel, int2(0), g.pc._InputDims - 1);
    float2 v = float2(
        no_diff(g.b._MotionVectorTex[uint4(b, 0, uint2(pixel))]),
        no_diff(g.b._MotionVectorTex[uint4(b, 1, uint2(pixel))])
    );
    v *= float(length(v) > 0.1f);
    return v;
}


[Differentiable]
inline float3 LoadColourUnjittered(Globals g, uint b, float2 uv)
{
    float3 c = sample_tensor(g.b._ColourTex, b, uint3(0, 1, 2), uv);
    return tonemap_forward(c * g.pc._Exposure.x, TonemapMode.Karis);
}


[Differentiable]
inline float3 LoadColourForDerivativeAtPixel(Globals g, uint b, int2 pixel)
{
    int2 sample_coord = ReflectIndex(pixel, g.pc._InputDims);
    float3 c = float3(
        no_diff(g.b._ColourTex[uint4(b, 0, uint2(sample_coord))]),
        no_diff(g.b._ColourTex[uint4(b, 1, uint2(sample_coord))]),
        no_diff(g.b._ColourTex[uint4(b, 2, uint2(sample_coord))])
    );
    c = max(c * g.pc._Exposure.x, 0.0f);
    return sqrt(c);
}


[Differentiable]
inline float3 WarpHistory(Globals g, uint b, float2 uv)
{
    float3 h = sample_tensor(g.b._HistoryTex, b, uint3(0, 1, 2), uv, false);
    return tonemap_forward(h * g.pc._Exposure.x, TonemapMode.Karis);
}


[Differentiable]
inline float4 WarpFeedback(Globals g, uint b, float2 uv, float disocclusion_mask)
{
    float4 feedback = sample_tensor(g.b._FeedbackTensor, b, uint4(0, 1, 2, 3), uv, false);
    float disocclusion_binary = float(disocclusion_mask > 0.01);
    feedback = lerp(feedback, 0.f, disocclusion_binary);
    return feedback;
}


[Differentiable]
inline float4 LoadDerivativeTm1(Globals g, uint b, float2 uv)
{
    return sample_tensor(g.b._LumaDerivTm1Tex, b, uint4(0, 1, 2, 3), uv, false);
}


[NoDiffThis]
inline int2 ProcessCoordToInputCoord(Globals g, int2 process_coord)
{
#if NSS_PREPROCESS_HALF_RES_INPUT
    float2 scaled = (float2(process_coord) + 0.5f) * g.pc._InvScale;
    return clamp(int2(floor(scaled)), int2(0), g.pc._InputDims - 1);
#else
    return clamp(process_coord, int2(0), g.pc._InputDims - 1);
#endif // NSS_PREPROCESS_HALF_RES_INPUT
}


[NoDiffThis]
inline int2 InputCoordToDepthCoord(Globals g, int2 input_coord)
{
    int2 depth_size = int2(round(rcp(max(g.pc._InvDepthTm1Dims, float2(EPS)))));
    float2 scaled = (float2(input_coord) + 0.5f) * rcp(float2(g.pc._InputDims)) * float2(depth_size);
    return clamp(int2(floor(scaled)), int2(0), depth_size - 1);
}


[NoDiffThis]
inline float2 MotionToPaddedUvDelta(Globals g, float2 motion)
{
#if NSS_PREPROCESS_HALF_RES_INPUT
    return motion * g.pc._Scale * g.pc._InvPaddedDims;
#else
    return motion * g.pc._InvPaddedDims;
#endif // NSS_PREPROCESS_HALF_RES_INPUT
}


[NoDiffThis]
inline float2 DerivativeSampleUv(Globals g, float2 uv)
{
    return uv * (float2(g.pc._OutputDims) * g.pc._InvPaddedDims);
}


[Differentiable]
inline float ComputeDisocclusion(Globals g, uint b, float2 reproj_270p_uv, float depth_dilated, bool inverted_depth)
{
#if NSS_PREPROCESS_HALF_RES_INPUT
    const float bilinear_weight_threshold = 0.1f;
    int2 depth_tm1_size = int2(round(rcp(max(g.pc._InvDepthTm1Dims, float2(EPS)))));
    float current_view_depth = no_diff(GetViewSpaceDepth(depth_dilated, g.pc._DeviceToViewDepth));
    float2 sample_px = (reproj_270p_uv * float2(depth_tm1_size)) - 0.5f;
    int2 sample_base = int2(floor(sample_px));
    float2 sample_frac = frac(sample_px);

    float w00 = (1.0f - sample_frac.x) * (1.0f - sample_frac.y);
    float w10 = sample_frac.x * (1.0f - sample_frac.y);
    float w01 = (1.0f - sample_frac.x) * sample_frac.y;
    float w11 = sample_frac.x * sample_frac.y;

    float required_sep_scale = no_diff(ComputeDepthClipRequiredSepScale(g));
    float depth_clip_power = no_diff(ComputeDepthClipPower(g));

    float fDepth = 0.0f;
    float fWeightSum = 0.0f;

    #define NSS_DEPTH_CLIP_SAMPLE_BLOCK(OFF_X, OFF_Y, WEIGHT)                                                  \
        {                                                                                                       \
            int2 sample_pos = sample_base + int2((OFF_X), (OFF_Y));                                            \
            float weight = (WEIGHT);                                                                            \
            bool onscreen = IsOnScreen(sample_pos, depth_tm1_size);                                             \
            fWeightSum += onscreen ? 0.0f : weight;                                                             \
            if (onscreen && weight > bilinear_weight_threshold) {                                               \
                float prev_depth = float(no_diff(g.b._DepthTm1Tex[uint4(b, 0, uint2(sample_pos))])) * INV_DEPTH_SCALE; \
                float prev_view_depth = no_diff(GetViewSpaceDepth(prev_depth, g.pc._DeviceToViewDepth));        \
                float depth_diff = current_view_depth - prev_view_depth;                                        \
                if (depth_diff > 0.0f) {                                                                        \
                    float depth_threshold = max(current_view_depth, prev_view_depth);                           \
                    float required_sep = required_sep_scale * depth_threshold;                                  \
                    float sep_ratio = saturate(required_sep / max(depth_diff, EPS));                            \
                    fDepth += pow(sep_ratio, depth_clip_power) * weight;                                        \
                    fWeightSum += weight;                                                                       \
                }                                                                                               \
            }                                                                                                   \
        }

    NSS_DEPTH_CLIP_SAMPLE_BLOCK(0, 0, w00);
    NSS_DEPTH_CLIP_SAMPLE_BLOCK(1, 0, w10);
    NSS_DEPTH_CLIP_SAMPLE_BLOCK(0, 1, w01);
    NSS_DEPTH_CLIP_SAMPLE_BLOCK(1, 1, w11);

    #undef NSS_DEPTH_CLIP_SAMPLE_BLOCK

    return fWeightSum > 0.0f ? saturate(1.0f - fDepth / fWeightSum) : 0.0f;
#else
    return no_diff(
        ComputeDepthClip(
            g.b._DepthTm1Tex,
            b,
            reproj_270p_uv,
            float2(g.pc._RenderSize),
            inverted_depth,
            depth_dilated,
            g.pc._DeviceToViewDepth
        )
    );
#endif // NSS_PREPROCESS_HALF_RES_INPUT
}


[Differentiable]
inline float CalculateMotionDetector(float2 vector, float2 render_size)
{
    const float k_pix_min = length(float2(1.f) / render_size);
    const float k_pix_max = length(float2(200.f) / render_size);
    const float k_pix_denom = rcp(k_pix_max - k_pix_min);

    float2 motion_norm = vector / render_size;
    float motion_length = length(motion_norm);
    return sqrt((clamp(motion_length, k_pix_min, k_pix_max) - k_pix_min) * k_pix_denom);
}


[NoDiffThis]
inline float3 RGBToYCoCg(float3 rgb)
{
    float Co = rgb.r - rgb.b;
    float t = rgb.b + Co * 0.5f;
    float Cg = rgb.g - t;
    float Y = t + Cg * 0.5f;
    return float3(Y, Co, Cg);
}


#if NSS_V1_LUMA_DERIVATIVE

[Differentiable]
inline float ComputeDerivativeDelta(
    float3 ycocg_a,
    float3 ycocg_b,
    float chroma_weight)
{
    // Measure color difference in YCoCg space, but intentionally weight chroma
    // more than luma because the moire/flicker this detector is chasing often
    // shows up more clearly in color than in plain luminance.
    return length(float3(
        ycocg_a.x - ycocg_b.x,
        (ycocg_a.y - ycocg_b.y) * chroma_weight,
        (ycocg_a.z - ycocg_b.z) * chroma_weight));
}


[Differentiable]
inline float4 CalculateLumaDerivative(
    Globals g,
    uint b,
    int2 ref_coord,
    float4 deriv_tm1,
    float disocclusion_mask,
    out float instability)
{
    // Thresholds for the derivative detector and its temporal filter.
    //
    // The persisted state layout is:
    //   deriv_tm1.xyz = previous frame color at this reprojected location in YCoCg
    //   deriv_tm1.w   = previous filtered instability scalar
    //
    // The detector has two jobs:
    //   1. decide whether the current pixel looks temporally unstable
    //   2. turn that into a calmer guidance signal for the network
    const float derivative_dis_thresh = 0.01;       // Binary cutoff for treating the derivative history as disoccluded/unreliable.
    const float chroma_weight = 1.25f;              // Extra weight applied to Co/Cg differences when computing YCoCg delta magnitude.
    const float recall_floor = 0.065f;              // Supported-instability level where entry begins to rise above zero.
    const float recall_ceil = 0.420f;               // Supported-instability level where the recall term saturates to fully on.
    const float excursion_floor = 0.025f;           // Minimum increase over `prev_instability` needed before new activation starts.
    const float excursion_ceil = 0.160f;            // Increase over `prev_instability` where the excursion term saturates.
    const float mean_gate_floor = 0.070f;           // Minimum absolute supported-instability needed to allow entry at all.
    const float mean_gate_ceil = 0.230f;            // Absolute supported-instability where the mean gate reaches full strength.
    const float sustain_cold_floor = 0.177f;        // Sustain threshold start for pixels that were previously cold.
    const float sustain_cold_ceil = 0.330f;         // Sustain threshold saturation point for previously cold pixels.
    const float sustain_hot_floor = 0.157f;         // Easier sustain threshold start for pixels that were already hot.
    const float sustain_hot_ceil = 0.305f;          // Easier sustain threshold saturation point for already-hot pixels.
    const float sustain_hysteresis_floor = 0.110f;  // `prev_instability` level where a pixel starts being treated as hot for sustain tuning.
    const float sustain_hysteresis_ceil = 0.210f;   // `prev_instability` level where the hot/cold sustain interpolation fully switches to hot.
    const float sustain_support_alpha = 0.30f;      // Blend factor that low-pass filters sustain support toward current support.
    const float hot_hold_floor = 0.180f;            // `prev_instability` level where the extra hot-only hold path starts to appear.
    const float hot_hold_ceil = 0.280f;             // `prev_instability` level where the extra hot-only hold path reaches full strength.
    const float sustain_min_hot_hold = 0.12f;       // Small minimum carry reserved for genuinely hot pixels even if sustain weakens briefly.
    const float decay_min_hot_gate = 0.50f;         // Minimum decay gate for hot pixels so their fall alpha behaves like a leaky EMA.
    const float sustain_strength = 0.80f;           // Multiplier applied when carrying previous instability forward through the sustain path.
    const float instability_rise_alpha_min = 0.08f; // Slowest rise alpha used for weakly supported new activations.
    const float instability_rise_alpha_max = 0.22f; // Fastest rise alpha used for strongly supported new activations.
    const float instability_fast_fall_alpha = 0.24f; // Fast decay alpha used when sustain support collapses.
    const float instability_fall_alpha = 0.05f;     // Slow decay alpha used when a hot pixel still has meaningful carry support.
    const float spatial_support_scale = 0.75f;      // Overall gain applied after blending temporal center delta with spatial support.
    const float spatial_support_blend = 0.30f;      // Weight of current-frame spatial support vs. center temporal delta in the blended score.

    // Load current-frame color for this pixel and its axial neighbors in the
    // derivative domain. `LoadColourForDerivativeAtPixel(...)` already applies
    // exposure and sqrt shaping, so the detector works on a compressed version
    // of the raw linear color rather than the main Karis-tonemapped path.
    float3 ycocg_c = no_diff(RGBToYCoCg(LoadColourForDerivativeAtPixel(g, b, ref_coord)));
    float3 ycocg_n = no_diff(RGBToYCoCg(LoadColourForDerivativeAtPixel(g, b, ref_coord + int2(0, -1))));
    float3 ycocg_s = no_diff(RGBToYCoCg(LoadColourForDerivativeAtPixel(g, b, ref_coord + int2(0, 1))));
    float3 ycocg_e = no_diff(RGBToYCoCg(LoadColourForDerivativeAtPixel(g, b, ref_coord + int2(1, 0))));
    float3 ycocg_w = no_diff(RGBToYCoCg(LoadColourForDerivativeAtPixel(g, b, ref_coord + int2(-1, 0))));

    // `d_center` is the actual temporal change: current center color against the
    // reprojected previous center color stored in history.
    //
    // The axial deltas are not temporal; they measure how much the current
    // center disagrees with its N/S/E/W neighbors. They are used as a cheap
    // "is this instability spatially supported?" signal.
    float d_center = ComputeDerivativeDelta(ycocg_c, deriv_tm1.xyz, chroma_weight);
    float d_n = ComputeDerivativeDelta(ycocg_c, ycocg_n, chroma_weight);
    float d_s = ComputeDerivativeDelta(ycocg_c, ycocg_s, chroma_weight);
    float d_e = ComputeDerivativeDelta(ycocg_c, ycocg_e, chroma_weight);
    float d_w = ComputeDerivativeDelta(ycocg_c, ycocg_w, chroma_weight);
    float spatial_delta_sum = d_n + d_s + d_e + d_w;
    float spatial_delta_max = max(max(d_n, d_s), max(d_e, d_w));
    float prev_instability = deriv_tm1.w;

    // Build a robust spatial-support estimate from the current frame only:
    // average the 3 weakest axial deltas and throw away the single strongest.
    // This keeps one bad neighbor from dominating the support score.
    //
    // Then blend that support with the true temporal delta:
    //   - `d_center` says "did this pixel change over time?"
    //   - `spatial_support` says "does the current neighborhood also look
    //      structured/unstable instead of being a single isolated outlier?"
    //
    // The result is a branchless, low-bandwidth instability estimate that keeps
    // only one reprojected history tap.
    float spatial_support = max(spatial_delta_sum - spatial_delta_max, 0.0f) * (1.0f / 3.0f);
    float supported_instability = lerp(d_center, spatial_support, spatial_support_blend) * spatial_support_scale;

    // Entry logic for new instability.
    //
    // A pixel should heat up only when:
    //   - the supported signal itself is large enough (`recall_score`)
    //   - it is meaningfully above the previous filtered value (`excursion_score`)
    //   - the overall mean level is not trivial (`mean_gate`)
    //
    // `raw_entry` is therefore an ignition term for genuinely new instability,
    // not a general-purpose "whatever is nonzero should turn on" score.
    float recall_excursion = max(supported_instability - prev_instability, 0.0f);
    float recall_score = saturate((supported_instability - recall_floor) * rcp(recall_ceil - recall_floor));
    float excursion_score = saturate((recall_excursion - excursion_floor) * rcp(excursion_ceil - excursion_floor));
    float mean_gate = saturate((supported_instability - mean_gate_floor) * rcp(mean_gate_ceil - mean_gate_floor));
    float raw_entry = sqrt(recall_score) * sqrt(excursion_score) * mean_gate;

    // Sustain / carry logic for pixels that were already hot.
    //
    // `sustain_heat` converts the previous instability into a hotness measure.
    // Hot pixels get slightly easier sustain thresholds than cold ones, which
    // reduces on/off chatter near the activation boundary.
    //
    // `sustain_support` intentionally smooths the support input a bit so a hot
    // pixel does not immediately collapse when one frame dips.
    //
    // `hot_hold_gate` is a smaller emergency carry path: once a pixel is
    // clearly hot, preserve a minimum amount of persistence even if the normal
    // sustain gate softens for a frame.
    float sustain_heat = saturate((prev_instability - sustain_hysteresis_floor) * rcp(sustain_hysteresis_ceil - sustain_hysteresis_floor));
    float sustain_support = lerp(prev_instability, supported_instability, sustain_support_alpha);
    float sustain_floor = lerp(sustain_cold_floor, sustain_hot_floor, sustain_heat);
    float sustain_ceil = lerp(sustain_cold_ceil, sustain_hot_ceil, sustain_heat);
    float sustain_gate = saturate((sustain_support - sustain_floor) * rcp(sustain_ceil - sustain_floor));
    sustain_gate *= sustain_gate;
    float hot_hold = saturate((prev_instability - hot_hold_floor) * rcp(hot_hold_ceil - hot_hold_floor));
    hot_hold *= hot_hold;
    float hot_hold_gate = hot_hold * sustain_min_hot_hold;
    float carry_gate = max(sustain_gate, hot_hold_gate);
    float raw_sustain = prev_instability * carry_gate * sustain_strength;

    // The detector target is whichever path is stronger:
    // a new ignition (`raw_entry`) or a carried-forward hot state
    // (`raw_sustain`).
    float raw_instability = max(raw_entry, raw_sustain);

    // Convert that target into the persisted instability state.
    //
    // Rising pixels use a support-dependent alpha: strong, well-supported
    // activations rise faster than weak ones.
    //
    // Falling pixels use a separate decay path keyed by `decay_gate`, so
    // established hot pixels cool more like a leaky EMA than a hard threshold.
    float decay_gate = max(sustain_gate, sustain_heat * sustain_heat * decay_min_hot_gate);
    float fall_alpha = lerp(instability_fast_fall_alpha, instability_fall_alpha, decay_gate);
    float rise_support = sqrt(recall_score * mean_gate);
    float rise_alpha = lerp(instability_rise_alpha_min, instability_rise_alpha_max, rise_support);
    float instability_alpha = raw_instability > prev_instability ? rise_alpha : fall_alpha;
    float filtered_instability = lerp(prev_instability, raw_instability, instability_alpha);

    // Under normal valid-history operation, the persisted scalar state is
    // `filtered_instability`, but the value exposed to the network gets one
    // extra mild EMA pass. This is output-only smoothing: it calms visible
    // channel-11 flicker without changing the underlying persisted state model.
    float output_rise_alpha = 0.75f;
    float output_fall_alpha = 0.80f;
    float output_alpha = filtered_instability > prev_instability ? output_rise_alpha : output_fall_alpha;
    float visible_instability = lerp(prev_instability, filtered_instability, output_alpha);

    // This is the candidate next history state for the normal valid-history
    // path: current YCoCg plus the newly filtered scalar instability.
    float4 derivative_state = float4(ycocg_c, filtered_instability);
    instability = visible_instability;

    // Disocclusion means the reprojected history should not be trusted.
    // Uninitialized history means we are effectively on the first valid frame.
    //
    // Visible instability is suppressed immediately on invalid history.
    //
    // The persisted derivative state must also be hard-reset here:
    //   - keep the current-frame YCoCg color as the new history seed
    //   - zero the scalar instability channel
    //
    // Softly blending old derivative history through a disocclusion lets stale
    // color/instability survive into the next frame, which then shows up as a
    // one-frame-late ghost trail once the disocclusion mask relaxes.
    float disocclusion_binary = float(disocclusion_mask > derivative_dis_thresh);
    float uninitialized_state = float((abs(deriv_tm1.x) + abs(deriv_tm1.y) + abs(deriv_tm1.z) + abs(deriv_tm1.w)) < 1e-4f);
    float history_valid = 1.0f - disocclusion_binary;
    instability *= history_valid;

    float4 reset_state = float4(ycocg_c, 0.0f);
    float4 seeded_state = lerp(derivative_state, reset_state, disocclusion_binary);
    instability = lerp(instability, 0.f, uninitialized_state);
    return lerp(seeded_state, reset_state, uninitialized_state);
}

#else

[Differentiable]
inline float4 CalculateLumaDerivative(
    float3 unjittered_colour,
    float4 deriv_tm1,
    float disocclusion_mask,
    out float instability)
{
    float deriv_max_pow_r = rcp(DERIV_MAX * sqrt(DERIV_MAX));
    float luma_tm1 = deriv_tm1.y;
    float luma_derivative_tm1 = deriv_tm1.x;
    float luma_t = luminance(unjittered_colour);

    float luma_derivative_t = abs(luma_t - luma_tm1);
    float clipped = min(luma_derivative_t, DERIV_MAX);
    clipped *= no_diff(step(DERIV_MIN, luma_derivative_t));
    float curved = clipped * sqrt(clipped) * deriv_max_pow_r;

    float applied_d_alpha = lerp(
        DERIV_ALPHA,
        DERIV_ALPHA * 0.1f,
        clamp(luma_derivative_tm1, 0.f, DERIV_MAX) * rcp(DERIV_MAX)
    );
    float luma_derivative = lerp(luma_derivative_tm1, curved, applied_d_alpha);

    luma_derivative *= no_diff(step(disocclusion_mask, DERIVATIVE_DIS_THRESH));
    instability = luma_derivative;
    return float4(luma_derivative, luma_t, 0.f, 0.f);
}

#endif // NSS_V1_LUMA_DERIVATIVE


[Differentiable]
inline void WriteToTensor(
    Globals g,
    uint batch,
    uint2 coord,
    float3 history,
    float3 colour,
    float motion_detector,
    float4 feedback,
    float luma_derivative)
{
    g.b._InputTensor.storeOnce(uint4(batch, 0, coord), history.r);
    g.b._InputTensor.storeOnce(uint4(batch, 1, coord), history.g);
    g.b._InputTensor.storeOnce(uint4(batch, 2, coord), history.b);
    g.b._InputTensor.storeOnce(uint4(batch, 3, coord), colour.r);
    g.b._InputTensor.storeOnce(uint4(batch, 4, coord), colour.g);
    g.b._InputTensor.storeOnce(uint4(batch, 5, coord), colour.b);
    g.b._InputTensor.storeOnce(uint4(batch, 6, coord), motion_detector);
    g.b._InputTensor.storeOnce(uint4(batch, 7, coord), feedback.r);
    g.b._InputTensor.storeOnce(uint4(batch, 8, coord), feedback.g);
    g.b._InputTensor.storeOnce(uint4(batch, 9, coord), feedback.b);
    g.b._InputTensor.storeOnce(uint4(batch, 10, coord), feedback.a);
    g.b._InputTensor.storeOnce(uint4(batch, 11, coord), luma_derivative);
}


[Differentiable]
inline void WriteLumaDerivative(Globals g, uint batch, uint2 coord, float4 luma)
{
    g.b._LumaDeriv.store(uint4(batch, 0, coord), luma.x);
    g.b._LumaDeriv.store(uint4(batch, 1, coord), luma.y);
    g.b._LumaDeriv.store(uint4(batch, 2, coord), luma.z);
    g.b._LumaDeriv.store(uint4(batch, 3, coord), luma.w);
}


[Differentiable]
inline void WriteDisocclusionMask(Globals g, uint batch, uint2 coord, float disocclusion_mask)
{
    g.b._DisocclusionMask.store(uint4(batch, 0, coord), disocclusion_mask);
}


[Differentiable]
inline void WriteNearestOffset(Globals g, uint batch, uint2 coord, int2 nearest_offset)
{
    float encoded = EncodeNearestDepthCoord(nearest_offset);
    g.b._NearestDepthOffset.store(uint4(batch, 0, coord), encoded);
}


[Differentiable]
inline void WriteNearestOffsetPackedQuad(Globals g, uint batch, uint2 coord)
{
    int2 base_input_coord = clamp(int2(coord) * int2(2), int2(0), g.pc._InputDims - 1);
    int2 lane_input_00 = base_input_coord;
    int2 lane_input_10 = min(base_input_coord + int2(0, 1), g.pc._InputDims - 1);
    int2 lane_input_01 = min(base_input_coord + int2(1, 0), g.pc._InputDims - 1);
    int2 lane_input_11 = min(base_input_coord + int2(1, 1), g.pc._InputDims - 1);

    float quad_depth_unused = 0.f;
    uint2 quad_nearest_coord = uint2(0);
    int2 quad_offset_00 = int2(0);
    int2 quad_offset_10 = int2(0);
    int2 quad_offset_01 = int2(0);
    int2 quad_offset_11 = int2(0);

    FindNearestDepth_4x4_FromPixel(g, batch, lane_input_00, quad_depth_unused, quad_nearest_coord, quad_offset_00);
    FindNearestDepth_4x4_FromPixel(g, batch, lane_input_10, quad_depth_unused, quad_nearest_coord, quad_offset_10);
    FindNearestDepth_4x4_FromPixel(g, batch, lane_input_01, quad_depth_unused, quad_nearest_coord, quad_offset_01);
    FindNearestDepth_4x4_FromPixel(g, batch, lane_input_11, quad_depth_unused, quad_nearest_coord, quad_offset_11);

    uint byte_r = EncodeNearestDepthCoordNibble(quad_offset_00) | (EncodeNearestDepthCoordNibble(quad_offset_10) << 4);
    uint byte_g = EncodeNearestDepthCoordNibble(quad_offset_01) | (EncodeNearestDepthCoordNibble(quad_offset_11) << 4);
    g.b._NearestDepthOffset.store(uint4(batch, 0, coord), float(byte_r) / 255.0f);
    g.b._NearestDepthOffset.store(uint4(batch, 1, coord), float(byte_g) / 255.0f);
}


// ─────────────────────────────────────────────────────────────────────────────
// Kernel
// ─────────────────────────────────────────────────────────────────────────────
[AutoPyBindCUDA]
[CUDAKernel]
[Differentiable]
public void pre_process(PREPROCESS_PARAMS(DECL))
{
    uint tId = (cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim()).x;
    uint3 size = tensor_size(out_tensor).xzw;
    uint3 idx = tensor_idx(tId, size);
    if (any(idx >= size)) return;

    Globals g = makeGlobals(PREPROCESS_ARGS, idx.x);

    const int2 low_res_size = g.pc._InputDims;
    const int2 process_size = int2(tensor_size(out_disocclusion_mask).zw);
    const float2 inv_low_res_size = g.pc._InvInputDims;
    const float2 inv_depth_tm1_size = g.pc._InvDepthTm1Dims;
    const float2 inv_size = g.pc._InvPaddedDims;

    int2 padded_coord = int2(idx.yz);
#if NSS_PREPROCESS_HALF_RES_INPUT
    int2 process_coord = ReflectIndex(padded_coord, process_size);
    int2 ref_coord = ProcessCoordToInputCoord(g, process_coord);
#else
    int2 ref_coord = ReflectIndex(padded_coord, low_res_size);
#endif // NSS_PREPROCESS_HALF_RES_INPUT
    float2 uv = (float2(ref_coord) + 0.5f) * inv_low_res_size;
    float2 uv_pad = (float2(padded_coord) + 0.5f) * inv_size;

    const float2 jitter_t = g.pc._JitterOffset.xy;
    const float2 render_size = g.pc._RenderSize;

    const bool inverted_depth = false;
    float depth_dilated = 0.f;
    uint2 nearest_coord = uint2(0);
    int2 nearest_offset = int2(0);
#if NSS_PREPROCESS_HALF_RES_INPUT
    FindNearestDepthLQ(g, idx.x, uv, depth_dilated, nearest_coord, nearest_offset);
#else
    FindNearestDepth_4x4(g.b._DepthTex, idx.x, uv, inverted_depth, depth_dilated, nearest_coord, nearest_offset);
#endif // NSS_PREPROCESS_HALF_RES_INPUT

#if NSS_PREPROCESS_HALF_RES_INPUT
    float2 motion = LoadMotionAtPixel(g, idx.x, int2(nearest_coord));
#else
    float2 nearest_uv = (float2(nearest_coord) + 0.5f) / float2(low_res_size);
    float2 motion = LoadMotion(g, idx.x, nearest_uv);
#endif // NSS_PREPROCESS_HALF_RES_INPUT

    float2 reproj_uv = uv - (motion * inv_low_res_size);
    float2 unjitter_uv = uv - (jitter_t * inv_low_res_size);

#if NSS_PREPROCESS_HALF_RES_INPUT
    int2 depth_coord = InputCoordToDepthCoord(g, ref_coord);
    float2 reproj_270p_uv = ((float2(depth_coord) + 0.5f) * inv_depth_tm1_size) - (motion * inv_low_res_size);
    float2 reproj_pad_uv = uv_pad - no_diff(MotionToPaddedUvDelta(g, motion));
#else
    float2 reproj_270p_uv = ((float2(ref_coord / 2) + 0.5f) * inv_depth_tm1_size) - (motion * inv_low_res_size);
    float2 reproj_pad_uv = uv_pad - (motion * inv_size);
#endif // NSS_PREPROCESS_HALF_RES_INPUT

    float disocclusion_mask = ComputeDisocclusion(g, idx.x, reproj_270p_uv, depth_dilated, inverted_depth);
    float3 unjittered_colour = LoadColourUnjittered(g, idx.x, unjitter_uv);
    float3 lr_warped_history = WarpHistory(g, idx.x, reproj_uv);

#if NSS_PREPROCESS_HALF_RES_INPUT
    float4 deriv_tm1 = LoadDerivativeTm1(g, idx.x, reproj_pad_uv);
#else
    float4 deriv_tm1 = LoadDerivativeTm1(g, idx.x, reproj_uv);
#endif // NSS_PREPROCESS_HALF_RES_INPUT
    float instability = 0.0f;
#if NSS_V1_LUMA_DERIVATIVE
    float4 luma = CalculateLumaDerivative(g, idx.x, ref_coord, deriv_tm1, disocclusion_mask, instability);
#else
    float4 luma = CalculateLumaDerivative(unjittered_colour, deriv_tm1, disocclusion_mask, instability);
#endif // NSS_V1_LUMA_DERIVATIVE
    float4 feedback = WarpFeedback(g, idx.x, reproj_pad_uv, disocclusion_mask);
    float motion_detector = CalculateMotionDetector(motion, render_size);

    WriteToTensor(g, idx.x, idx.yz, lr_warped_history, unjittered_colour, motion_detector, feedback, instability);

    if (any(idx.yz >= process_size)) return;
    WriteLumaDerivative(g, idx.x, idx.yz, luma);
    WriteDisocclusionMask(g, idx.x, idx.yz, disocclusion_mask);
#if NSS_PACKED_NEAREST_OFFSET_QUAD
    WriteNearestOffsetPackedQuad(g, idx.x, idx.yz);
#else
    WriteNearestOffset(g, idx.x, idx.yz, nearest_offset);
#endif // NSS_PACKED_NEAREST_OFFSET_QUAD
}

#undef NAME_ONLY
#undef DECL
#undef PREPROCESS_ARGS
#undef PREPROCESS_PARAMS
