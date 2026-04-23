// SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
// its affiliates <open-source-office@arm.com></text>
// SPDX-License-Identifier: Apache-2.0

import "../../../../common/model/shaders/tensor";

#ifndef GLSL_MAPPING
#define GLSL_MAPPING

/*
    Mappings of common GLSL functions. These are not 1:1, but are syntactically very similar. (e.g. might have
    an extra batch input)
*/

#define lessThan(x, y) ((x) < (y))
#define greaterThanEqual(x, y) ((x) >= (y))
#define findMSB(x) (((x == 0u) ? 0 : (int(floor(log2(float(x)))) + 1)) - 1)
#define textureSize(x, y) (tensor_size(x).zw)

T texelFetchOffset<T : IArithmetic>(uint b, TensorView<T> tensor, uint2 pixel, int2 offset)
{
    int2 p   = int2(pixel);
    int2 ofs = int2(offset);
    int2 maxc = int2(tensor_size(tensor).zw) - int2(1, 1);
    int2 c   = clamp(p + ofs, int2(0, 0), maxc);
    return tensor[uint4(uint(b), 0u, uint2(c))];
}

void imageStore(uint b, TensorView<float> tensor, uint2 pixel, float4 value)
{
    uint4 size = tensor_size(tensor);
    uint channels = size.y;
    // Don't write pixels out of frame
    if (any(pixel < 0) || any(pixel > (size.zw - 1))) return;
    // Write values to all existing channels
    tensor[b, 0, pixel.x, pixel.y] = value.x;
    if (channels > 1) tensor[b, 1, pixel.x, pixel.y] = value.y;
    if (channels > 2) tensor[b, 2, pixel.x, pixel.y] = value.z;
    if (channels > 3) tensor[b, 3, pixel.x, pixel.y] = value.w;
}


#endif // GLSL_MAPPING
