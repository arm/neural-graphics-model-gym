# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Union

import numpy as np
import OpenEXR
import torch


def read_exr(
    path: Union[str, Path],
    channels: Union[str, list[str], None] = "RGBA",
    force_dtype: np.dtype = np.float32,
    axis: int = -1,
) -> np.ndarray:
    """
    Read an EXR file, inferring dtype per channel.

    Parameters:
        path: Path to the EXR file.
        channels: String (e.g., "RGB"), list of channel names, or None to read all.
        force_dtype: Desired final dtype (defaults to float32).
        axis: Axis to stack channels on (default: -1 for HWC).

    Returns:
        np.ndarray of shape (H, W, C) or (C, H, W), dtype=force_dtype.
    """
    f = OpenEXR.InputFile(str(path))
    header = f.header()

    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Determine which channels to read
    if channels is None:
        channels_to_read = list(header["channels"].keys())  # use all channels in file
    elif isinstance(channels, str):
        channels_to_read = list(channels)
    else:
        channels_to_read = list(channels)

    channel_data = []
    for ch in channels_to_read:
        if ch not in header["channels"]:
            raise ValueError(
                f"Channel '{ch}' not found in EXR. Available: {list(header['channels'].keys())}"
            )

        pixel_type = header["channels"][ch].type
        np_dtype, _ = pixel_type_to_dtype_and_bytes(pixel_type)

        raw = f.channel(ch)
        arr = np.frombuffer(raw, dtype=np_dtype).reshape((height, width))
        channel_data.append(arr.astype(force_dtype))

    return np.stack(channel_data, axis=axis)


def read_exr_torch(path: Path, dtype: np.dtype, channels: str = "RGB"):
    """Read an EXR file and return it as a PyTorch tensor"""
    return convert_data_to_torch_tensor(
        read_exr(path, force_dtype=dtype, channels=channels)
    )


def convert_data_to_torch_tensor(in_data) -> torch.Tensor:
    """a.k.a NHWC -> NCHW -> torch.Tensor"""
    if len(in_data.shape) == 3:
        in_data = np.transpose(in_data, axes=(2, 0, 1))
    elif len(in_data.shape) < 1:
        in_data = np.resize(in_data, (1,))
    return torch.from_numpy(np.copy(in_data)).unsqueeze(0).contiguous()


def pixel_type_to_dtype_and_bytes(ptype: OpenEXR.PixelType) -> tuple[np.dtype, int]:
    """Map OpenEXR.PixelType to dtype and bytes"""
    pval = ptype.v if hasattr(ptype, "v") else int(ptype)
    if pval == OpenEXR.HALF:
        return np.float16, 2
    if pval == OpenEXR.FLOAT:
        return np.float32, 4
    if pval == OpenEXR.UINT:
        return np.uint32, 4
    raise ValueError(f"Unsupported pixel type: {ptype}")
