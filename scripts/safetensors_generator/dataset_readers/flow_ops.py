# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn.functional as F


def _window(
    x: torch.Tensor,
    ksize: int = 3,
    step: int = 1,
    stride: int = 1,
    mode: str = "constant",
) -> torch.Tensor:
    """Extract KxK windows from x and return B x C x K*K x H x W patches."""
    batch_size, channels, height, width = x.shape

    pad_along_height = max(ksize - stride, 0)
    if height % stride != 0:
        pad_along_height = max(ksize - (height % stride), 0)

    pad_along_width = max(ksize - stride, 0)
    if width % stride != 0:
        pad_along_width = max(ksize - (width % stride), 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode=mode, value=0)

    patches = F.unfold(
        x,
        kernel_size=(ksize, ksize),
        stride=(stride, stride),
        padding=(0, 0),
        dilation=(step, step),
    )
    out_h = (height + stride - 1) // stride
    out_w = (width + stride - 1) // stride
    return torch.reshape(patches, (batch_size, channels, ksize**2, out_h, out_w))


def upscale_and_dilate_flow(
    tensor_to_dilate: torch.Tensor,
    depth: torch.Tensor,
    scale: Optional[float] = 2.0,
    kernel_size: Optional[int] = 3,
    interpolation: Optional[str] = "nearest",
    is_flow: Optional[bool] = True,
) -> torch.Tensor:
    """Depth-aware dilate and upsample helper used for synthetic motion-vector hints."""
    k = int(kernel_size) if kernel_size is not None else 3
    scale = float(scale) if scale is not None else 2.0

    channels = tensor_to_dilate.shape[1]
    in_height = tensor_to_dilate.shape[2]
    in_width = tensor_to_dilate.shape[3]
    out_height = int(in_height * scale)
    out_width = int(in_width * scale)

    depth_3x3 = _window(depth, ksize=k, mode="reflect")
    flow_3x3 = _window(tensor_to_dilate, ksize=k, mode="reflect")

    min_depth_idx = torch.argmin(depth_3x3, dim=2).unsqueeze(2)
    index = torch.tile(min_depth_idx, (1, channels, 1, 1, 1))
    dilated_tensor = torch.gather(flow_3x3, dim=2, index=index).squeeze(2)

    output_size = (out_height, out_width)
    if interpolation == "nearest":
        dilated_tensor_upsampled = F.interpolate(
            dilated_tensor, size=output_size, mode="nearest"
        )
    else:
        dilated_tensor_upsampled = F.interpolate(
            dilated_tensor, size=output_size, mode=interpolation, align_corners=False
        )

    flow_scale = scale if is_flow else torch.ones_like(dilated_tensor_upsampled)
    return dilated_tensor_upsampled * flow_scale
