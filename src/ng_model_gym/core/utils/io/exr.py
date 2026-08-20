# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
"""Helpers for writing RGB images to OpenEXR files."""

from pathlib import Path

import OpenEXR
import torch


def _as_float32_chw_rgb(frame: torch.Tensor) -> torch.Tensor:
    if not isinstance(frame, torch.Tensor):
        raise TypeError("EXR frame must be a torch.Tensor.")
    if frame.ndim == 4:
        if frame.shape[0] != 1:
            raise ValueError(
                f"EXR NCHW input must have batch size one; received {frame.shape[0]}."
            )
        frame = frame[0]
    elif frame.ndim != 3:
        raise ValueError(
            "EXR frame must be CHW or single-frame NCHW; "
            f"received shape {tuple(frame.shape)}."
        )

    if frame.shape[0] != 3:
        raise ValueError(
            f"EXR frame must have three RGB channels; received {frame.shape[0]}."
        )
    if not torch.is_floating_point(frame):
        raise TypeError("EXR frame must use a floating point dtype.")
    if frame.shape[1] <= 0 or frame.shape[2] <= 0:
        raise ValueError("EXR frame must have positive height and width.")
    if not bool(torch.isfinite(frame).all()):
        raise ValueError("EXR frame must contain only finite values.")

    converted = frame.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(converted).all()):
        raise ValueError(
            "EXR frame values must remain finite when represented as float32."
        )
    return converted


def write_rgb_float32_exr(path: str | Path, frame: torch.Tensor) -> None:
    """Write one RGB frame as three float32 channels without value transforms."""

    chw = _as_float32_chw_rgb(frame)
    channels = {
        channel_name: chw[channel_index].numpy()
        for channel_index, channel_name in enumerate("RGB")
    }
    header = {
        "compression": OpenEXR.ZIP_COMPRESSION,
        "type": OpenEXR.scanlineimage,
    }
    with OpenEXR.File(header, channels) as outfile:
        outfile.write(str(path))
