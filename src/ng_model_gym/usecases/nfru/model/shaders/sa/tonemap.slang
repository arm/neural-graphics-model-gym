// SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
// its affiliates <open-source-office@arm.com></text>
// SPDX-License-Identifier: Apache-2.0

#ifndef TONEMAP
#define TONEMAP
//#define TONEMAP_KARIS
#define TONEMAP_REINHARD

#if defined(TONEMAP_KARIS)

// see https ://graphicrants.blogspot.com/2013/12/tone-mapping.html
float3 Tonemap(float3 colour)
{
    // 1 / (1 + max(rgb))
    return colour / (1.f + max(max(colour.r, colour.g), colour.b));
}

float3 InverseTonemap(float3 colour)
{
    // 1 / (1 - max(rgb))
    return colour / (1.f - max(max(colour.r, colour.g), colour.b));
}

#elif defined(TONEMAP_REINHARD)

float3 Tonemap(float3 colour)
{
    // 1 / (1 + colour)
    return colour / (float3(1.f) + colour);
}

float3 TonemapExp(float3 colour, float exponent)
{
    // 1 / (1 + colour)
    colour = colour * exp(exponent);
    return colour / (float3(1.f) + colour);
}

float3 InverseTonemap(float3 colour)
{
    // 1 / (1 - colour)
    return colour / (float3(1.f) - colour);
}

float4 Tonemap(float4 colour)
{
    // 1 / (1 + colour)
    return colour / (float4(1.f) + colour);
}

float4 InverseTonemap(float4 colour)
{
    // 1 / (1 - colour)
    return colour / (float4(1.f) - colour);
}
#endif

#endif // TONEMAP
