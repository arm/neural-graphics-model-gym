// SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
// its affiliates <open-source-office@arm.com></text>
// SPDX-License-Identifier: Apache-2.0

#include "constants.slang"

import "../../../common/model/shaders/tensor";
import "../../../common/model/shaders/depth";
import "../../../common/model/shaders/image";

// Multiply a 4x4 matrix by a 4D vector.
[NoDiffThis]
float VectorDot(float4 vector1, float4 vector2)
{
    float result = vector1[0]*vector2[0] + vector1[1]*vector2[1] + vector1[2]*vector2[2] + vector1[3]*vector2[3];
    return result;
}

// Multiply a 4x4 matrix by a 4D vector.
[NoDiffThis]
float4 MatrixVectorMultiplication(float4x4 matrix, float4 vector)
{
    float4 result;
    // Each row of the matrix is dotted with the vector.
    result.x = VectorDot(matrix[0], vector);
    result.y = VectorDot(matrix[1], vector);
    result.z = VectorDot(matrix[2], vector);
    result.w = VectorDot(matrix[3], vector);
    return result;
}

// Multiply a 4x4 matrix by a 4x4 matrix.
[NoDiffThis]
float4x4 MatrixMatrixMultiplication(float4x4 matrix1, float4x4 matrix2)
{
    float4x4 result = float4x4(0.0);
    result[0][0] = VectorDot(matrix1[0], float4(matrix2[0][0], matrix2[1][0], matrix2[2][0], matrix2[3][0]));
    result[0][1] = VectorDot(matrix1[0], float4(matrix2[0][1], matrix2[1][1], matrix2[2][1], matrix2[3][1]));
    result[0][2] = VectorDot(matrix1[0], float4(matrix2[0][2], matrix2[1][2], matrix2[2][2], matrix2[3][2]));
    result[0][3] = VectorDot(matrix1[0], float4(matrix2[0][3], matrix2[1][3], matrix2[2][3], matrix2[3][3]));

    result[1][0] = VectorDot(matrix1[1], float4(matrix2[0][0], matrix2[1][0], matrix2[2][0], matrix2[3][0]));
    result[1][1] = VectorDot(matrix1[1], float4(matrix2[0][1], matrix2[1][1], matrix2[2][1], matrix2[3][1]));
    result[1][2] = VectorDot(matrix1[1], float4(matrix2[0][2], matrix2[1][2], matrix2[2][2], matrix2[3][2]));
    result[1][3] = VectorDot(matrix1[1], float4(matrix2[0][3], matrix2[1][3], matrix2[2][3], matrix2[3][3]));

    result[2][0] = VectorDot(matrix1[2], float4(matrix2[0][0], matrix2[1][0], matrix2[2][0], matrix2[3][0]));
    result[2][1] = VectorDot(matrix1[2], float4(matrix2[0][1], matrix2[1][1], matrix2[2][1], matrix2[3][1]));
    result[2][2] = VectorDot(matrix1[2], float4(matrix2[0][2], matrix2[1][2], matrix2[2][2], matrix2[3][2]));
    result[2][3] = VectorDot(matrix1[2], float4(matrix2[0][3], matrix2[1][3], matrix2[2][3], matrix2[3][3]));

    result[3][0] = VectorDot(matrix1[3], float4(matrix2[0][0], matrix2[1][0], matrix2[2][0], matrix2[3][0]));
    result[3][1] = VectorDot(matrix1[3], float4(matrix2[0][1], matrix2[1][1], matrix2[2][1], matrix2[3][1]));
    result[3][2] = VectorDot(matrix1[3], float4(matrix2[0][2], matrix2[1][2], matrix2[2][2], matrix2[3][2]));
    result[3][3] = VectorDot(matrix1[3], float4(matrix2[0][3], matrix2[1][3], matrix2[2][3], matrix2[3][3]));

    return result;
}


// Helper function: Compute the determinant of a 3x3 matrix.
[NoDiffThis]
float det3(float3x3 m)
{
    return m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
        - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
        + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
}

// Compute the determinant of a 4x4 matrix.
[NoDiffThis]
float det4(float4x4 m)
{
    // Cofactor expansion along the first row.
    float subDet0 = det3(float3x3(
        m[1][1], m[1][2], m[1][3],
        m[2][1], m[2][2], m[2][3],
        m[3][1], m[3][2], m[3][3]
    ));
    float subDet1 = det3(float3x3(
        m[1][0], m[1][2], m[1][3],
        m[2][0], m[2][2], m[2][3],
        m[3][0], m[3][2], m[3][3]
    ));
    float subDet2 = det3(float3x3(
        m[1][0], m[1][1], m[1][3],
        m[2][0], m[2][1], m[2][3],
        m[3][0], m[3][1], m[3][3]
    ));
    float subDet3 = det3(float3x3(
        m[1][0], m[1][1], m[1][2],
        m[2][0], m[2][1], m[2][2],
        m[3][0], m[3][1], m[3][2]
    ));
    return m[0][0] * subDet0
         - m[0][1] * subDet1
         + m[0][2] * subDet2
         - m[0][3] * subDet3;
}


// Inverse a 4x4 matrix by a 4D vector.
[NoDiffThis]
float4x4 inverseMatrix(float4x4 m)
{
    float4x4 inv = float4x4(1.0);
    float det = det4(m);
    if (det == 0.0) {
        // Matrix is non-invertible. You can choose to return an identity matrix,
        // a zero matrix, or handle the error as needed.
        inv = float4x4(1.0); // Here we return the identity matrix.
        return inv;
    }
    det = 1.0 / det;

    inv[0][0] =  m[1][1]*m[2][2]*m[3][3] - m[1][1]*m[2][3]*m[3][2] - m[2][1]*m[1][2]*m[3][3]
        + m[2][1]*m[1][3]*m[3][2] + m[3][1]*m[1][2]*m[2][3] - m[3][1]*m[1][3]*m[2][2];

    inv[0][1] = -m[0][1]*m[2][2]*m[3][3] + m[0][1]*m[2][3]*m[3][2] + m[2][1]*m[0][2]*m[3][3]
        - m[2][1]*m[0][3]*m[3][2] - m[3][1]*m[0][2]*m[2][3] + m[3][1]*m[0][3]*m[2][2];

    inv[0][2] = m[0][1]*m[1][2]*m[3][3] - m[0][1]*m[1][3]*m[3][2] - m[1][1]*m[0][2]*m[3][3]
        + m[1][1]*m[0][3]*m[3][2] + m[3][1]*m[0][2]*m[1][3] - m[3][1]*m[0][3]*m[1][2];

    inv[0][3] = -m[0][1]*m[1][2]*m[2][3] + m[0][1]*m[1][3]*m[2][2] + m[1][1]*m[0][2]*m[2][3]
        - m[1][1]*m[0][3]*m[2][2] - m[2][1]*m[0][2]*m[1][3] + m[2][1]*m[0][3]*m[1][2];

    inv[1][0] = -m[1][0]*m[2][2]*m[3][3] + m[1][0]*m[2][3]*m[3][2] + m[2][0]*m[1][2]*m[3][3]
        - m[2][0]*m[1][3]*m[3][2] - m[3][0]*m[1][2]*m[2][3] + m[3][0]*m[1][3]*m[2][2];

    inv[1][1] =  m[0][0]*m[2][2]*m[3][3] - m[0][0]*m[2][3]*m[3][2] - m[2][0]*m[0][2]*m[3][3]
        + m[2][0]*m[0][3]*m[3][2] + m[3][0]*m[0][2]*m[2][3] - m[3][0]*m[0][3]*m[2][2];

    inv[1][2] = -m[0][0]*m[1][2]*m[3][3] + m[0][0]*m[1][3]*m[3][2] + m[1][0]*m[0][2]*m[3][3]
        - m[1][0]*m[0][3]*m[3][2] - m[3][0]*m[0][2]*m[1][3] + m[3][0]*m[0][3]*m[1][2];

    inv[1][3] = m[0][0]*m[1][2]*m[2][3] - m[0][0]*m[1][3]*m[2][2] - m[1][0]*m[0][2]*m[2][3]
        + m[1][0]*m[0][3]*m[2][2] + m[2][0]*m[0][2]*m[1][3] - m[2][0]*m[0][3]*m[1][2];

    inv[2][0] =  m[1][0]*m[2][1]*m[3][3] - m[1][0]*m[2][3]*m[3][1] - m[2][0]*m[1][1]*m[3][3]
        + m[2][0]*m[1][3]*m[3][1] + m[3][0]*m[1][1]*m[2][3] - m[3][0]*m[1][3]*m[2][1];

    inv[2][1] = -m[0][0]*m[2][1]*m[3][3] + m[0][0]*m[2][3]*m[3][1] + m[2][0]*m[0][1]*m[3][3]
        - m[2][0]*m[0][3]*m[3][1] - m[3][0]*m[0][1]*m[2][3] + m[3][0]*m[0][3]*m[2][1];

    inv[2][2] =  m[0][0]*m[1][1]*m[3][3] - m[0][0]*m[1][3]*m[3][1] - m[1][0]*m[0][1]*m[3][3]
        + m[1][0]*m[0][3]*m[3][1] + m[3][0]*m[0][1]*m[1][3] - m[3][0]*m[0][3]*m[1][1];

    inv[2][3] = -m[0][0]*m[1][1]*m[2][3] + m[0][0]*m[1][3]*m[2][1] + m[1][0]*m[0][1]*m[2][3]
        - m[1][0]*m[0][3]*m[2][1] - m[2][0]*m[0][1]*m[1][3] + m[2][0]*m[0][3]*m[1][1];

    inv[3][0] = -m[1][0]*m[2][1]*m[3][2] + m[1][0]*m[2][2]*m[3][1] + m[2][0]*m[1][1]*m[3][2]
        - m[2][0]*m[1][2]*m[3][1] - m[3][0]*m[1][1]*m[2][2] + m[3][0]*m[1][2]*m[2][1];

    inv[3][1] =  m[0][0]*m[2][1]*m[3][2] - m[0][0]*m[2][2]*m[3][1] - m[2][0]*m[0][1]*m[3][2]
        + m[2][0]*m[0][2]*m[3][1] + m[3][0]*m[0][1]*m[2][2] - m[3][0]*m[0][2]*m[2][1];

    inv[3][2] = -m[0][0]*m[1][1]*m[3][2] + m[0][0]*m[1][2]*m[3][1] + m[1][0]*m[0][1]*m[3][2]
        - m[1][0]*m[0][2]*m[3][1] - m[3][0]*m[0][1]*m[1][2] + m[3][0]*m[0][2]*m[1][1];

    inv[3][3] =  m[0][0]*m[1][1]*m[2][2] - m[0][0]*m[1][2]*m[2][1] - m[1][0]*m[0][1]*m[2][2]
        + m[1][0]*m[0][2]*m[2][1] + m[2][0]*m[0][1]*m[1][2] - m[2][0]*m[0][2]*m[1][1];

    // Multiply each element by the reciprocal of the determinant
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            inv[i][j] *= det;
        }
    }
    return inv;
}

[Differentiable]
float max4(float4 x)
{
    return max(max(max(x.x, x.y), x.z), x.w);
}

[Differentiable]
float4 softmax(float4 value) {
    float4 value_exp = exp(value);
    float sumExp = value_exp.x + value_exp.y + value_exp.z + value_exp.w;
    return float4(value_exp.x / sumExp, value_exp.y / sumExp, value_exp.z / sumExp, value_exp.w / sumExp);
}

[Differentiable]
float4 stable_softmax(float4 value)
{
    // Subtract the max value to help reduce overflow issues
    float4 stable_ten = value - max4(value);
    float4 e_ten = exp(stable_ten);
    float sum = dot(e_ten, float4(1.0));
    return e_ten / (sum + EPS);
}