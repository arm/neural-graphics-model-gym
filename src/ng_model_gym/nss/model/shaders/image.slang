// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
module image;


[Differentiable]
public float luminance(float3 rgb)
{
    // ITU-R BT.709: `0.2126 * R + 0.7152 * G + 0.0722 * B`
    return dot(rgb, float3(0.2126, 0.7152, 0.0722));
}


[Differentiable]
public float3 RGBToYCoCg(float3 x)
{
    // https://en.wikipedia.org/wiki/YCoCg
    return float3(
         0.25f * x.r + 0.5f * x.g + 0.25f * x.b,
         0.50f * x.r - 0.5f * x.b,
        -0.25f * x.r + 0.5f * x.g - 0.25f * x.b,
    );
}


[Differentiable]
public float3 YCoCgToRGB(float3 x)
{
    // https://en.wikipedia.org/wiki/YCoCg
    float Y = x.r;
    float Co = x.g;
    float Cg = x.b;
    float tmp = Y - Cg;
    return float3(
        tmp + Co,
        Y + Cg,
        tmp - Co,
    );
}


[Differentiable]
public float3 MaxHalf(float3 x)
{
    // clamp to max half float
    const float max_float = 65504.0f;
    return min(x, max_float);
}

[Differentiable]
public float MaxHalf(float x)
{
    // clamp to max half float
    const float max_float = 65504.0f;
    return min(x, max_float);
}
