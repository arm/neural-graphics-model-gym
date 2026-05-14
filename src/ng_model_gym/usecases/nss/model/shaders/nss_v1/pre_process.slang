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

#define EPS 1e-7
#define DEPTH_SCALE 2147483647.0
#define INV_DEPTH_SCALE (1.0 / DEPTH_SCALE)


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
inline float2 LoadDerivativeTm1(Globals g, uint b, float2 uv)
{
    return sample_tensor(g.b._LumaDerivTm1Tex, b, uint2(0, 1), uv, false);
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


[Differentiable]
inline float2 CalculateLumaDerivative(
    float3 unjittered_colour,
    float2 deriv_tm1,
    float disocclusion_mask)
{
    const float derivative_dis_thresh = 0.01;
    const float deriv_pow = 1.5;
    const float deriv_min = 0.05;
    const float deriv_max = 0.3;
    const float deriv_denom = rcp(pow(deriv_max, deriv_pow));
    const float d_alpha = 0.1;

    float luma_tm1 = deriv_tm1.y;
    float luma_derivative_tm1 = deriv_tm1.x;
    float luma_t = luminance(unjittered_colour);

    float luma_derivative_t = abs(luma_t - luma_tm1);
    luma_derivative_t = luma_derivative_t > deriv_min ? min(luma_derivative_t, deriv_max) : 0.f;
    luma_derivative_t = pow(luma_derivative_t, deriv_pow) * deriv_denom;

    float applied_d_alpha = lerp(d_alpha, d_alpha * 0.1, min(luma_derivative_tm1, deriv_max) * rcp(deriv_max));
    float luma_derivative = lerp(luma_derivative_tm1, luma_derivative_t, applied_d_alpha);

    float disocclusion_binary = float(disocclusion_mask > derivative_dis_thresh);
    float derivative = lerp(luma_derivative, 0.f, disocclusion_binary);
    return float2(derivative, luma_t);
}


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
inline void WriteLumaDerivative(Globals g, uint batch, uint2 coord, float2 luma)
{
    g.b._LumaDeriv.store(uint4(batch, 0, coord), luma.x);
    g.b._LumaDeriv.store(uint4(batch, 1, coord), luma.y);
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
    float2 deriv_tm1 = LoadDerivativeTm1(g, idx.x, no_diff(DerivativeSampleUv(g, reproj_uv)));
#else
    float2 deriv_tm1 = LoadDerivativeTm1(g, idx.x, reproj_uv);
#endif // NSS_PREPROCESS_HALF_RES_INPUT
    float2 luma = CalculateLumaDerivative(unjittered_colour, deriv_tm1, disocclusion_mask);
    float4 feedback = WarpFeedback(g, idx.x, reproj_pad_uv, disocclusion_mask);
    float motion_detector = CalculateMotionDetector(motion, render_size);

    WriteToTensor(g, idx.x, idx.yz, lr_warped_history, unjittered_colour, motion_detector, feedback, luma.x);

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
