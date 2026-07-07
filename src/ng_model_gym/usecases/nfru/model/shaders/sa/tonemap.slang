// SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
// its affiliates <open-source-office@arm.com></text>
// SPDX-License-Identifier: Apache-2.0

#ifndef TONEMAP
#define TONEMAP
//#define TONEMAP_KARIS
#define TONEMAP_REINHARD

#if defined(TONEMAP_KARIS)

// see https ://graphicrants.blogspot.com/2013/12/tone-mapping.html
float3 Tonemap(float3 color)
{
    // 1 / (1 + max(rgb))
    return color / (1.f + max(max(color.r, color.g), color.b));
}

float3 InverseTonemap(float3 color)
{
    // 1 / (1 - max(rgb))
    return color / (1.f - max(max(color.r, color.g), color.b));
}

#elif defined(TONEMAP_REINHARD)

float3 Tonemap(float3 color)
{
    // 1 / (1 + color)
    return color / (float3(1.f) + color);
}

float3 TonemapExp(float3 color, float exponent)
{
    // 1 / (1 + color)
    color = color * exp(exponent);
    return color / (float3(1.f) + color);
}

float3 InverseTonemap(float3 color)
{
    // 1 / (1 - color)
    return color / (float3(1.f) - color);
}

float4 Tonemap(float4 color)
{
    // 1 / (1 + color)
    return color / (float4(1.f) + color);
}

float4 InverseTonemap(float4 color)
{
    // 1 / (1 - color)
    return color / (float4(1.f) - color);
}
#endif

#endif // TONEMAP
