// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
import custom_ops;

#define EPS 1e-7


[NoDiffThis]
[PreferRecompute]
public uint4 tensor_size(DiffTensorView tensor)
{
    return uint4(
        tensor.size(0),
        tensor.size(1),
        tensor.size(2),
        tensor.size(3)
    );
}


[NoDiffThis]
[PreferRecompute]
public uint4 tensor_size<T>(TensorView<T> tensor)
{
    return uint4(
        tensor.size(0),
        tensor.size(1),
        tensor.size(2),
        tensor.size(3)
    );
}


[NoDiffThis]
[PreferRecompute]
public uint4 tensor_size(TensorView<uint> tensor)
{
    return uint4(
        tensor.size(0),
        tensor.size(1),
        tensor.size(2),
        tensor.size(3)
    );
}


[NoDiffThis]
[PreferRecompute]
public uint4 tensor_size(TensorView<int> tensor)
{
    return uint4(
        tensor.size(0),
        tensor.size(1),
        tensor.size(2),
        tensor.size(3)
    );
}



[NoDiffThis]
public uint4 tensor_idx(uint tId, uint4 size)
{
    return uint4(
        (tId / (size.w * size.z * size.y)) % size.x,
        (tId / (size.w * size.z)) % size.y,
        (tId / size.w) % size.z,
        tId % size.w
    );
}


[NoDiffThis]
public uint3 tensor_idx(uint tId, uint3 size)
{
    return uint3(
        (tId / (size.z * size.y)) % size.x,
        (tId / size.z) % size.y,
        tId % size.z,
    );
}


[Differentiable]
public float sample_tensor(TensorView<float> tensor, uint b, uint c, float2 uv, bool clamp_to_edge = true)
{
    uint2 size = tensor_size(tensor).zw;
    float2 idx = uv * float2(size);
    int2 g0 = int2(floor(idx - 0.5));
    int2 g1 = g0 + 1;

    float2 w0 = lerp_weight(float2(g0) + 0.5, idx);
    float2 w1 = lerp_weight(float2(g1) + 0.5, idx);

    if (!clamp_to_edge) {
        w0 *= float2(g0 >= 0.f) * float2(g0 < size);
        w1 *= float2(g1 >= 0.f) * float2(g1 < size);
    }

    int height = tensor.size(2);
    int width = tensor.size(3);

    uint4 i00 = uint4(b, c, clamp(g0.x, 0, height-1 ), clamp(g0.y, 0, width-1));
    uint4 i01 = uint4(b, c, clamp(g0.x, 0, height-1 ), clamp(g1.y, 0, width-1));
    uint4 i10 = uint4(b, c, clamp(g1.x, 0, height-1 ), clamp(g0.y, 0, width-1));
    uint4 i11 = uint4(b, c, clamp(g1.x, 0, height-1 ), clamp(g1.y, 0, width-1));

    float t00 = no_diff(tensor[i00]) * w0.x * w0.y;
    float t01 = no_diff(tensor[i01]) * w0.x * w1.y;
    float t10 = no_diff(tensor[i10]) * w1.x * w0.y;
    float t11 = no_diff(tensor[i11]) * w1.x * w1.y;

    return t00 + t01 + t10 + t11;
}


[Differentiable]
public float2 sample_tensor(TensorView<float> tensor, uint b, uint2 c, float2 uv, bool clamp_to_edge = true)
{
    return float2(
        sample_tensor(tensor, b, c.x, uv, clamp_to_edge),
        sample_tensor(tensor, b, c.y, uv, clamp_to_edge)
    );
}


[Differentiable]
public float3 sample_tensor(TensorView<float> tensor, uint b, uint3 c, float2 uv, bool clamp_to_edge = true)
{
    return float3(
        sample_tensor(tensor, b, c.x, uv, clamp_to_edge),
        sample_tensor(tensor, b, c.y, uv, clamp_to_edge),
        sample_tensor(tensor, b, c.z, uv, clamp_to_edge)
    );
}


[Differentiable]
public float4 sample_tensor(TensorView<float> tensor, uint b, uint4 c, float2 uv, bool clamp_to_edge = true)
{
    return float4(
        sample_tensor(tensor, b, c.x, uv, clamp_to_edge),
        sample_tensor(tensor, b, c.y, uv, clamp_to_edge),
        sample_tensor(tensor, b, c.z, uv, clamp_to_edge),
        sample_tensor(tensor, b, c.w, uv, clamp_to_edge),
    );
}


[Differentiable]
public float sample_tensor(DiffTensorView tensor, uint b, uint c, float2 uv, bool clamp_to_edge = true)
{
    uint2 size = tensor_size(tensor).zw;
    float2 idx = uv * float2(size);
    float2 g0 = (idx - 0.5) - frac(idx - 0.5);
    float2 g1 = g0 + 1;

    float2 w0 = lerp_weight(float2(g0) + 0.5, idx);
    float2 w1 = lerp_weight(float2(g1) + 0.5, idx);

    if (!clamp_to_edge) {
        w0 *= float2(g0 >= 0.f) * float2(g0 < size);
        w1 *= float2(g1 >= 0.f) * float2(g1 < size);
    }

    int height = tensor.size(2);
    int width = tensor.size(3);

    uint4 i00 = uint4(b, c, int(custom_clamp(g0.x, 0.0, float(height-1))), int(custom_clamp(g0.y, 0, float(width-1))));
    uint4 i01 = uint4(b, c, int(custom_clamp(g0.x, 0.0, float(height-1))), int(custom_clamp(g1.y, 0, float(width-1))));
    uint4 i10 = uint4(b, c, int(custom_clamp(g1.x, 0.0, float(height-1))), int(custom_clamp(g0.y, 0, float(width-1))));
    uint4 i11 = uint4(b, c, int(custom_clamp(g1.x, 0.0, float(height-1))), int(custom_clamp(g1.y, 0, float(width-1))));

    float t00 = tensor.load(i00) * w0.x * w0.y;
    float t01 = tensor.load(i01) * w0.x * w1.y;
    float t10 = tensor.load(i10) * w1.x * w0.y;
    float t11 = tensor.load(i11) * w1.x * w1.y;

    return t00 + t01 + t10 + t11;
}


[Differentiable]
public float2 sample_tensor(DiffTensorView tensor, uint b, uint2 c, float2 uv, bool clamp_to_edge = true)
{
    return float2(
        sample_tensor(tensor, b, c.x, uv, clamp_to_edge),
        sample_tensor(tensor, b, c.y, uv, clamp_to_edge)
    );
}


[Differentiable]
public float3 sample_tensor(DiffTensorView tensor, uint b, uint3 c, float2 uv, bool clamp_to_edge = true)
{
    return float3(
        sample_tensor(tensor, b, c.x, uv, clamp_to_edge),
        sample_tensor(tensor, b, c.y, uv, clamp_to_edge),
        sample_tensor(tensor, b, c.z, uv, clamp_to_edge)
    );
}


[Differentiable]
public float4 sample_tensor(DiffTensorView tensor, uint b, uint4 c, float2 uv, bool clamp_to_edge = true)
{
    return float4(
        sample_tensor(tensor, b, c.x, uv, clamp_to_edge),
        sample_tensor(tensor, b, c.y, uv, clamp_to_edge),
        sample_tensor(tensor, b, c.z, uv, clamp_to_edge),
        sample_tensor(tensor, b, c.w, uv, clamp_to_edge),
    );
}


public float getInputElement(DiffTensorView input, uint4 loc)
{
    return input[loc];
}


[BackwardDerivativeOf(getInputElement)]
public void getInputElement_bwd(
    DiffTensorView input,
    uint4 loc,
    float derivative)
{
    float oldVal;
    input.diff.diff.InterlockedAdd(loc, derivative, oldVal);
}


[Differentiable]
[PreferRecompute]
public float2 lerp_weight(float2 x, float2 xs)
{
    float2 dx = x - xs;
    float2 abs_dx = custom_abs(dx);
    return custom_max(float2(1.f) - abs_dx, float2(0.f));
}



#define BICUBIC_CATMULL_ROM_SAMPLES 5U


// optimised bicubic catmull rom implementation
public void Bicubic2DCatmullRom(in float2 uv, in float2 size, in float2 invSize, out float2 samples[3], out float2 weights[3])
{
    uv *= size;

    float2 tc = floor(uv - 0.5) + 0.5;
    float2 f  = uv - tc;
    float2 f2 = f * f;
    float2 f3 = f2 * f;

    float2 w0 = f2 - 0.5 * (f3 + f);
    float2 w1 = 1.5 * f3 - 2.5 * f2 + 1.f;
    float2 w3 = 0.5 * (f3 - f2);
    float2 w2 = 1.f - w0 - w1 - w3;

    weights[0] = w0;
    weights[1] = w1 + w2;
    weights[2] = w3;

    samples[0] = tc - 1.f;
    samples[1] = tc + w2 / weights[1];
    samples[2] = tc + 2.f;

    samples[0] *= invSize;
    samples[1] *= invSize;
    samples[2] *= invSize;
}


public struct CatmullRomSamples
{
    // bilinear sampling UV coordinates of the samples
    public float2 UV[BICUBIC_CATMULL_ROM_SAMPLES];

    // weights of the samples
    public float Weight[BICUBIC_CATMULL_ROM_SAMPLES];

    // final multiplier (it is faster to multiply 3 RGB values than reweights the 5 weights)
    public float FinalMultiplier;
};


public CatmullRomSamples GetBicubic2DCatmullRomSamples(float2 uv, float2 size, in float2 invSize)
{
    float2 weights[3];
    float2 samples[3];
    Bicubic2DCatmullRom(uv, size, invSize, samples, weights);

    CatmullRomSamples crSamples;

    // optimized by removing corner samples
    crSamples.UV[0] = float2(samples[1].x, samples[0].y);
    crSamples.UV[1] = float2(samples[0].x, samples[1].y);
    crSamples.UV[2] = float2(samples[1].x, samples[1].y);
    crSamples.UV[3] = float2(samples[2].x, samples[1].y);
    crSamples.UV[4] = float2(samples[1].x, samples[2].y);

    crSamples.Weight[0] = weights[1].x * weights[0].y;
    crSamples.Weight[1] = weights[0].x * weights[1].y;
    crSamples.Weight[2] = weights[1].x * weights[1].y;
    crSamples.Weight[3] = weights[2].x * weights[1].y;
    crSamples.Weight[4] = weights[1].x * weights[2].y;

    // reweight after removing the corners
    float cornerWeights;
    cornerWeights = crSamples.Weight[0];
    cornerWeights += crSamples.Weight[1];
    cornerWeights += crSamples.Weight[2];
    cornerWeights += crSamples.Weight[3];
    cornerWeights += crSamples.Weight[4];
    crSamples.FinalMultiplier = 1.f / cornerWeights;

    return crSamples;
}

[Differentiable]
public float sample_tensor_nearest(TensorView<float> tensor, uint b, uint c, float2 uv, bool clamp_to_edge = true)
{
    uint2 size = tensor_size(tensor).zw;
    float2 idx = uv * float2(size);
    uint2 uvi = uint2(floor(idx));
    uint4 i = uint4(b, c, uvi);
    return no_diff(tensor.load(i));
}

[Differentiable]
public float2 sample_tensor_nearest(TensorView<float> tensor, uint b, uint2 c, float2 uv, bool clamp_to_edge = true)
{
    return float2(
        sample_tensor_nearest(tensor, b, c.x, uv, clamp_to_edge),
        sample_tensor_nearest(tensor, b, c.y, uv, clamp_to_edge)
    );
}

[Differentiable]
public float3 sample_tensor_nearest(TensorView<float> tensor, uint b, uint3 c, float2 uv, bool clamp_to_edge = true)
{
    return float3(
        sample_tensor_nearest(tensor, b, c.x, uv, clamp_to_edge),
        sample_tensor_nearest(tensor, b, c.y, uv, clamp_to_edge),
        sample_tensor_nearest(tensor, b, c.z, uv, clamp_to_edge)
    );
}


[Differentiable]
public float4 sample_tensor_nearest(TensorView<float> tensor, uint b, uint4 c, float2 uv, bool clamp_to_edge = true)
{
    return float4(
        sample_tensor_nearest(tensor, b, c.x, uv, clamp_to_edge),
        sample_tensor_nearest(tensor, b, c.y, uv, clamp_to_edge),
        sample_tensor_nearest(tensor, b, c.z, uv, clamp_to_edge),
        sample_tensor_nearest(tensor, b, c.w, uv, clamp_to_edge),
    );
}

[Differentiable]
public float sample_tensor_nearest(DiffTensorView<float> tensor, uint b, uint c, float2 uv, bool clamp_to_edge = true)
{
    uint2 size = tensor_size(tensor).zw;
    float2 idx = uv * float2(size);
    uint2 uvi = uint2(floor(idx));
    uint4 i = uint4(b, c, uvi);
    return tensor.load(i);
}

[Differentiable]
public float2 sample_tensor_nearest(DiffTensorView<float> tensor, uint b, uint2 c, float2 uv, bool clamp_to_edge = true)
{
    return float2(
        sample_tensor_nearest(tensor, b, c.x, uv, clamp_to_edge),
        sample_tensor_nearest(tensor, b, c.y, uv, clamp_to_edge)
    );
}


[Differentiable]
public float3 sample_tensor_nearest(DiffTensorView<float> tensor, uint b, uint3 c, float2 uv, bool clamp_to_edge = true)
{
    return float3(
        sample_tensor_nearest(tensor, b, c.x, uv, clamp_to_edge),
        sample_tensor_nearest(tensor, b, c.y, uv, clamp_to_edge),
        sample_tensor_nearest(tensor, b, c.z, uv, clamp_to_edge)
    );
}


[Differentiable]
public float4 sample_tensor_nearest(DiffTensorView<float> tensor, uint b, uint4 c, float2 uv, bool clamp_to_edge = true)
{
    return float4(
        sample_tensor_nearest(tensor, b, c.x, uv, clamp_to_edge),
        sample_tensor_nearest(tensor, b, c.y, uv, clamp_to_edge),
        sample_tensor_nearest(tensor, b, c.z, uv, clamp_to_edge),
        sample_tensor_nearest(tensor, b, c.w, uv, clamp_to_edge),
    );
}

[Differentiable]
public float3 sample_tensor_catmull(DiffTensorView tensor, uint b, uint3 c, float2 uv, bool clamp_to_edge = true, bool dering = true)
{
    float2 size = float2(tensor_size(tensor).zw);

    CatmullRomSamples samples = GetBicubic2DCatmullRomSamples(uv, size, rcp(size));

    // assumes fp16 colour precision
    const float max_half = 65504.f;
    float3 colour = float3(0.f);
    float3 aabb_min = float(max_half);
    float3 aabb_max = float(-max_half);
    [ForceUnroll]
    for (uint i = 0U; i < BICUBIC_CATMULL_ROM_SAMPLES; i++)
    {
        float2 sampleUV = samples.UV[i];
        float3 sample = sample_tensor(tensor, b, c, sampleUV, clamp_to_edge);
        aabb_min = min(aabb_min, sample);
        aabb_max = max(aabb_max, sample);
        colour += sample * samples.Weight[i];
    }
    colour *= samples.FinalMultiplier;

    // defend against negative values
    if (dering)
        return any(colour < 0.f) ? clamp(colour, aabb_min, aabb_max) : colour;
    else
        return max(float3(0.f), colour * samples.FinalMultiplier);
}

[Differentiable]
public float4 sample_tensor_catmull(DiffTensorView tensor, uint b, uint4 c, float2 uv, bool clamp_to_edge = true, bool dering = true)
{
    float2 size = float2(tensor_size(tensor).zw);

    CatmullRomSamples samples = GetBicubic2DCatmullRomSamples(uv, size, rcp(size));

    // assumes fp16 colour precision
    const float max_half = 65504.f;
    float4 colour = float4(0.f);
    float4 aabb_min = float4(max_half);
    float4 aabb_max = float4(-max_half);
    [ForceUnroll]
    for (uint i = 0U; i < BICUBIC_CATMULL_ROM_SAMPLES; i++)
    {
        float2 sampleUV = samples.UV[i];
        float4 sample = sample_tensor(tensor, b, c, sampleUV, clamp_to_edge);
        aabb_min = min(aabb_min, sample);
        aabb_max = max(aabb_max, sample);
        colour += sample * samples.Weight[i];
    }
    colour *= samples.FinalMultiplier;

    // defend against negative values
    if (dering)
        return any(colour < 0.f) ? clamp(colour, aabb_min, aabb_max) : colour;
    else
        return max(float4(0.f), colour * samples.FinalMultiplier);
}

[NoDiffThis]
public float4x4 tensor_to_float4x4(TensorView<float> tensor, uint b)
{
    float4x4 tensor_float4x4 = float4x4(0.0);
    tensor_float4x4[0][0] = (tensor[uint3(b, 0, 0)]);
    tensor_float4x4[0][1] = (tensor[uint3(b, 0, 1)]);
    tensor_float4x4[0][2] = (tensor[uint3(b, 0, 2)]);
    tensor_float4x4[0][3] = (tensor[uint3(b, 0, 3)]);
    tensor_float4x4[1][0] = (tensor[uint3(b, 1, 0)]);
    tensor_float4x4[1][1] = (tensor[uint3(b, 1, 1)]);
    tensor_float4x4[1][2] = (tensor[uint3(b, 1, 2)]);
    tensor_float4x4[1][3] = (tensor[uint3(b, 1, 3)]);
    tensor_float4x4[2][0] = (tensor[uint3(b, 2, 0)]);
    tensor_float4x4[2][1] = (tensor[uint3(b, 2, 1)]);
    tensor_float4x4[2][2] = (tensor[uint3(b, 2, 2)]);
    tensor_float4x4[2][3] = (tensor[uint3(b, 2, 3)]);
    tensor_float4x4[3][0] = (tensor[uint3(b, 3, 0)]);
    tensor_float4x4[3][1] = (tensor[uint3(b, 3, 1)]);
    tensor_float4x4[3][2] = (tensor[uint3(b, 3, 2)]);
    tensor_float4x4[3][3] = (tensor[uint3(b, 3, 3)]);
    return tensor_float4x4;
}

[Differentiable]
public float3 avg_pool2d(DiffTensorView tensor, uint b, uint2 xy, uint2 scale)
{
    uint2 scaled_idx = xy * scale;

    float3 total = float3(0.f);
    [MaxIters(100)]
    for (uint i = 0; i < scale.x; i++) {

        [MaxIters(100)]
        for (uint j = 0; j < scale.y; j++) {
            total += float3(
                tensor.load(uint4(b, 0, xy + i)),
                tensor.load(uint4(b, 1, xy + i)),
                tensor.load(uint4(b, 2, xy + i))
            );
        }
    }
    return total / float(scale.x * scale.y);
}
