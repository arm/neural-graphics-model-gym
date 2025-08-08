# pylint: skip-file
# SPDX-FileCopyrightText: Copyright 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

# NOTE: source code copied from FSR2, which has the following license:

# This file is part of the FidelityFX SDK.

# Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Reference please see:
# - https://github.com/GPUOpen-Effects/FidelityFX-FSR2/blob/master/src/ffx-fsr2-api/ffx_fsr2.cpp#L663

import torch


def depth_to_view_space_params(
    zNear: torch.Tensor,
    zFar: torch.Tensor,
    FovY: torch.Tensor,
    infinite: bool,
    renderSizeWidth: torch.Tensor,
    renderSizeHeight: torch.Tensor,
    inverted: torch.Tensor,
) -> torch.Tensor:
    # make sure it has no impact if near and far plane values are swapped in dispatch params
    # the flags "inverted" and "infinite" will decide what transform to use
    tmpMin = torch.minimum(zNear, zFar)
    tmpMax = torch.maximum(zNear, zFar)

    # Min and Max swap when inverted
    fMin = torch.where(inverted, tmpMax, tmpMin)
    fMax = torch.where(inverted, tmpMin, tmpMax)

    # a 0 0 0   x
    # 0 b 0 0   y
    # 0 0 c d   z
    # 0 0 e 0   1

    FLT_EPSILON = 1e-7
    fQ = fMax / (fMin - fMax)
    d = torch.tensor(-1.0)  # for clarity

    matrix_elem_c = torch.where(
        inverted,
        torch.where(
            infinite,
            0.0 + FLT_EPSILON,  # reversed, infinite
            fQ,  # reversed, non infinite
        ),
        torch.where(
            infinite,
            -1.0 - FLT_EPSILON,  # non reversed, infinite
            fQ,  # non reversed, non infinite
        ),
    )

    matrix_elem_e = torch.where(
        inverted,
        torch.where(
            infinite,
            fMax,  # reversed, infinite
            fQ * fMin,  # reversed, non infinite
        ),
        torch.where(
            infinite,
            -fMin - FLT_EPSILON,  # non reversed, infinite
            fQ * fMin,  # non reversed, non infinite
        ),
    )

    deviceToViewDepth = [0, 0, 0, 0]
    deviceToViewDepth[0] = d * matrix_elem_c
    deviceToViewDepth[1] = matrix_elem_e

    # revert x and y coords
    aspect = renderSizeWidth.to(torch.float32) / renderSizeHeight.to(torch.float32)
    cameraFovAngleVertical = FovY
    cotHalfFovY = torch.cos(0.5 * cameraFovAngleVertical) / torch.sin(
        0.5 * cameraFovAngleVertical
    )
    a = cotHalfFovY / aspect
    b = cotHalfFovY

    deviceToViewDepth[2] = 1.0 / a
    deviceToViewDepth[3] = 1.0 / b

    return torch.concat(deviceToViewDepth, dim=1)
