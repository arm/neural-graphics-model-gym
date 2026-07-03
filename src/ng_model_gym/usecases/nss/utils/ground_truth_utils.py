# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F


def resize_ground_truth_to_spatial_shape(
    tensor: torch.Tensor, target_spatial_shape: tuple[int, int]
) -> torch.Tensor:
    """Resize ground truth tensors to the rounded NSS output size."""

    target_h, target_w = int(target_spatial_shape[0]), int(target_spatial_shape[1])
    if tensor.shape[-2:] == (target_h, target_w):
        return tensor.contiguous()

    if tensor.ndim == 5:
        batch, time, channels, height, width = tensor.shape
        resized = F.interpolate(
            tensor.reshape(batch * time, channels, height, width),
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return resized.reshape(batch, time, channels, target_h, target_w).contiguous()

    return F.interpolate(
        tensor,
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).contiguous()
