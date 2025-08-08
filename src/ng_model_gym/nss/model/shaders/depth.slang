// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
module depth;


[NoDiffThis]
public float linear_01_depth(float depth, float zNear, float zFar)
{
    return rcp(depth * (1.0 - (zFar / zNear)) + (zFar / zNear));
}
