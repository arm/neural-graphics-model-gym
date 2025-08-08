// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
module custom_ops;


[Differentiable]
public float custom_max(float x, float y){
    if (x > y) {
        return x;
    } else {
        return y;
    }
}


[BackwardDerivativeOf(custom_max)]
public void custom_max_bwd(inout DifferentialPair<float> x, inout DifferentialPair<float> y, float derivative){
    float x_d = 0.0;
    float y_d = 0.0;
    if (x.p > y.p) {
        x_d = 1.0 * derivative;
        y_d = 0.0 * derivative;
    } else if (x.p == y.p) {
        x_d = 0.5 * derivative;
        y_d = 0.5 * derivative;
    } else {
        x_d = 0.0 * derivative;
        y_d = 1.0 * derivative;
    }

    x = x.dadd(x.dzero(), DifferentialPair<float>(x.p, x_d));
    y = y.dadd(y.dzero(), DifferentialPair<float>(y.p, y_d));

    return;
}


[Differentiable]
public float2 custom_max(float2 x, float2 y) {
    return float2(
        custom_max(x.x, y.x),
        custom_max(x.y, y.y)
    );
}


[Differentiable]
public float3 custom_max(float3 x, float3 y) {
    return float3(
        custom_max(x.x, y.x),
        custom_max(x.y, y.y),
        custom_max(x.z, y.z),
    );
}


[Differentiable]
public float4 custom_max(float4 x, float4 y) {
    return float4(
        custom_max(x.x, y.x),
        custom_max(x.y, y.y),
        custom_max(x.z, y.z),
        custom_max(x.w, y.w),
    );
}


[Differentiable]
public float custom_min(float x, float y){
    if (x < y) {
        return x;
    } else {
        return y;
    }
}


[BackwardDerivativeOf(custom_min)]
public void custom_min_bwd(inout DifferentialPair<float> x, inout DifferentialPair<float> y, float derivative){
    float x_d = 0.0;
    float y_d = 0.0;
    if (x.p < y.p) {
        x_d = 1.0 * derivative;
        y_d = 0.0 * derivative;
    } else if (x.p == y.p) {
        x_d = 0.5 * derivative;
        y_d = 0.5 * derivative;
    } else {
        x_d = 0.0 * derivative;
        y_d = 1.0 * derivative;
    }

    x = x.dadd(x.dzero(), DifferentialPair<float>(x.p, x_d));
    y = y.dadd(y.dzero(), DifferentialPair<float>(y.p, y_d));

    return;
}


[Differentiable]
public float2 custom_min(float2 x, float2 y) {
    return float2(
        custom_min(x.x, y.x),
        custom_min(x.y, y.y),
    );
}


[Differentiable]
public float3 custom_min(float3 x, float3 y) {
    return float3(
        custom_min(x.x, y.x),
        custom_min(x.y, y.y),
        custom_min(x.z, y.z),
    );
}


[Differentiable]
public float4 custom_min(float4 x, float4 y) {
    return float4(
        custom_min(x.x, y.x),
        custom_min(x.y, y.y),
        custom_min(x.z, y.z),
        custom_min(x.w, y.w),
    );
}


[Differentiable]
public float custom_abs(float x){
    if (x < 0) {
        return -x;
    } else {
        return x;
    }
}


[BackwardDerivativeOf(custom_abs)]
public void custom_abs_bwd(inout DifferentialPair<float> x, float derivative){
    float x_d = 0.0;
    if (x.p > 0.0) {
        x_d = 1.0 * derivative;
    } else if (x.p == 0.0) {
        x_d = 0.0 * derivative;
    } else {
        x_d = -1.0 * derivative;
    }

    x = x.dadd(x.dzero(), DifferentialPair<float>(x.p, x_d));

    return;
}


[Differentiable]
public float2 custom_abs(float2 x){
    return float2(
        custom_abs(x.x),
        custom_abs(x.y)
    );
}


[Differentiable]
public float3 custom_abs(float3 x){
    return float3(
        custom_abs(x.x),
        custom_abs(x.y),
        custom_abs(x.z)
    );
}


[Differentiable]
public float4 custom_abs(float4 x){
    return float4(
        custom_abs(x.x),
        custom_abs(x.y),
        custom_abs(x.z),
        custom_abs(x.w)
    );
}


[Differentiable]
public float custom_clamp(float x, float min, float max) {
    float min_val = custom_min(x, max);
    return custom_max(min_val, min);
}


[Differentiable]
public float2 custom_clamp(float2 x, float2 min, float2 max) {
    return float2(
        custom_clamp(x.x, min.x, max.x),
        custom_clamp(x.y, min.y, max.y),
    );
}


[Differentiable]
public float3 custom_clamp(float3 x, float3 min, float3 max) {
    return float3(
        custom_clamp(x.x, min.x, max.x),
        custom_clamp(x.y, min.y, max.y),
        custom_clamp(x.z, min.z, max.z)
    );
}


[Differentiable]
public float4 custom_clamp(float4 x, float4 min, float4 max) {
    return float4(
        custom_clamp(x.x, min.x, max.x),
        custom_clamp(x.y, min.y, max.y),
        custom_clamp(x.z, min.z, max.z),
        custom_clamp(x.w, min.w, max.w),
    );
}
