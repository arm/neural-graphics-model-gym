// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
module tonemap;

#define HDR_MAX 65504.0f


public enum TonemapMode {
    Reinhard,
    Karis,
};


[Differentiable]
public float3 tonemap_forward(float3 x, TonemapMode mode, float max_val = 1.f)
{
    float3 tmp_x = max(x, float3(0.0f));
    float3 colour;

    switch (mode) {
        case TonemapMode.Reinhard:
            colour = tmp_x * rcp(1.f + tmp_x);
            break;
        case TonemapMode.Karis:
            colour = tmp_x * rcp(1.f + max(max(tmp_x.r, tmp_x.g), tmp_x.b));
            break;
    }

    return clamp(colour, 0.0, max_val);
}


[Differentiable]
public float3 tonemap_inverse(float3 x, TonemapMode mode, float max_val = 1.f)
{
    float3 tmp_x = max(x, float3(0.0f));
    float3 colour;
    switch (mode) {
        case TonemapMode.Reinhard:
            tmp_x = clamp(
                tmp_x,
                float3(0.0f),
                tonemap_forward(float3(HDR_MAX), TonemapMode.Reinhard)
            );
            colour = tmp_x * rcp(1.f - tmp_x);
            break;
        case TonemapMode.Karis:
            tmp_x = clamp(
                tmp_x,
                float3(0.0f),
                tonemap_forward(float3(HDR_MAX), TonemapMode.Karis)
            );
            colour = tmp_x * rcp(1.f - max(max(tmp_x.r, tmp_x.g), tmp_x.b));
            break;
    }

    return colour;
}
