// SPDX-FileCopyrightText: Copyright 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0

/* !NOTE: source code copied from FSR2, which has the following licence:

This file is part of the FidelityFX SDK.

Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Reference please see:
- https://github.com/GPUOpen-Effects/FidelityFX-FSR2/blob/master/src/ffx-fsr2-api/shaders/ffx_fsr2_depth_clip.h
- https://github.com/GPUOpen-Effects/FidelityFX-FSR2/blob/master/src/ffx-fsr2-api/shaders/ffx_fsr2_reconstruct_dilated_velocity_and_previous_depth.h
*/

module depth_clip;

import tensor;


[NoDiffThis]
public bool IsOnScreen(int2 pos, uint2 size)
{
    return all(pos < int2(size)) && all(pos >= int2(0));
}


[NoDiffThis]
public float2 ComputeNdc(float2 pixPos, int2 size)
{
    return pixPos / float2(size) * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f);
}


[NoDiffThis]
public float GetViewSpaceDepth(float depth, float4 device_to_view)
{
    // device_to_view details found in ffx_fsr2.cpp
    return (device_to_view[1] / (depth - device_to_view[0]));
}


[NoDiffThis]
public float3 GetViewSpacePosition(int2 iViewportPos, int2 iViewportSize, float fDeviceDepth, float4 device_to_view)
{
    const float Z = GetViewSpaceDepth(fDeviceDepth, device_to_view);

    const float2 fNdcPos = ComputeNdc(iViewportPos, iViewportSize);
    const float X = device_to_view[2] * fNdcPos.x * Z;
    const float Y = device_to_view[3] * fNdcPos.y * Z;

    return float3(X, Y, Z);
}


public struct BilinearSamplingData
{
    public int2 iOffsets[4];
    public float fWeights[4];
    public int2 iBasePos;
};


[NoDiffThis]
public BilinearSamplingData GetBilinearSamplingData(float2 fUv, uint2 iSize)
{
    BilinearSamplingData data;

    float2 fPxSample = (fUv * iSize) - float2(0.5f, 0.5f);
    data.iBasePos = int2(floor(fPxSample));
    float2 fPxFrac = fract(fPxSample);

    data.iOffsets[0] = int2(0, 0);
    data.iOffsets[1] = int2(1, 0);
    data.iOffsets[2] = int2(0, 1);
    data.iOffsets[3] = int2(1, 1);

    data.fWeights[0] = (1 - fPxFrac.x) * (1 - fPxFrac.y);
    data.fWeights[1] = (fPxFrac.x) * (1 - fPxFrac.y);
    data.fWeights[2] = (1 - fPxFrac.x) * (fPxFrac.y);
    data.fWeights[3] = (fPxFrac.x) * (fPxFrac.y);

    return data;
}


[NoDiffThis]
public void FindNearestDepth(TensorView<float> depth_tensor, uint b, float2 uv, bool inverted, out float nearestDepth, out uint2 nearestDepthCoord)
{
    const uint iSampleCount = 9;
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
    nearestDepthCoord = iPxPos;
    [ForceUnroll]
    for (iSampleIndex = 1; iSampleIndex < iSampleCount; ++iSampleIndex) {
        int2 iPos = int2(iPxPos) + iSampleOffsets[iSampleIndex];
        if (IsOnScreen(iPos, iPxSize)) {
            float depthN = depth[iSampleIndex];
            bool closer = inverted ? depthN > nearestDepth : depthN < nearestDepth;
            if (closer) {
                nearestDepthCoord = iPos;
                nearestDepth = depthN;
            }
        }
    }
}


[NoDiffThis]
public float ComputeDepthClip(TensorView<float> depth_tm1, uint b, float2 uv, float2 render_size, bool inverted, float current_depth, float4 device_to_view, float bilinear_weight_threshold = 0.1f)
{
    const float fReconstructedDepthBilinearWeightThreshold = bilinear_weight_threshold;
    float2 depth_tm1_size = tensor_size(depth_tm1).zw;
    float fCurrentDepthViewSpace = GetViewSpaceDepth(current_depth, device_to_view);
    BilinearSamplingData bilinearInfo = GetBilinearSamplingData(uv, uint2(depth_tm1_size));

    float fDilatedSum = 0.0f;
    float fDepth = 0.0f;
    float fWeightSum = 0.0f;
    for (int iSampleIndex = 0; iSampleIndex < 4; iSampleIndex++) {

        const int2 iOffset = bilinearInfo.iOffsets[iSampleIndex];
        const int2 iSamplePos = bilinearInfo.iBasePos + iOffset;

        //ANET Modification for off-screen regions to be flagged as disocclusions too
        const float fWeight = bilinearInfo.fWeights[iSampleIndex];
        const bool onscreen = IsOnScreen(iSamplePos, uint2(depth_tm1_size));
        fWeightSum += onscreen ? 0.f : fWeight;

        if (onscreen) {
            if (fWeight > fReconstructedDepthBilinearWeightThreshold) {

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
