# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
"""Utilities for crafting synthetic camera-cut datasets used in unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE_SAFETENSOR = Path(
    "tests/usecases/nss/datasets/train/train_cropped_sample.safetensors"
)


def _load_base_tensors() -> tuple[dict[str, torch.Tensor], dict]:
    """Load tensors/metadata from the canonical NSS sample safetensor."""

    if not BASE_SAFETENSOR.exists():
        raise FileNotFoundError(f"Missing base safetensor at {BASE_SAFETENSOR}")

    with safe_open(BASE_SAFETENSOR, framework="pt", device="cpu") as src:
        metadata = dict(src.metadata())
        tensors = {name: src.get_tensor(name) for name in src.keys()}
    return tensors, metadata


def write_camera_cut_fixture(
    root_dir: Path,
    camera_cut_flags: Sequence[bool],
    *,
    include_camera_cut: bool = True,
    file_name: str = "camcut",
) -> Path:
    """Create a safetensor dataset seeded from the golden sample.

    Args:
        root_dir: Temporary directory owned by the caller.
        camera_cut_flags: Ordered bools describing every frame in the capture.
        include_camera_cut: When False, omit the tensor entirely to emulate legacy files.
        file_name: Stem for the generated safetensor.

    Returns:
        Directory containing the new safetensor, suitable for `create_simple_params().`
    """

    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    tensors, metadata = _load_base_tensors()
    original_length = int(metadata["Length"])
    requested_length = len(camera_cut_flags)
    if requested_length > original_length:
        raise ValueError(
            "Requested camera_cut length exceeds the base safetensor length"
        )

    target_path = root_dir / f"{file_name}.safetensors"

    # Trim per-frame tensors down to the requested timeline.
    for key, tensor in list(tensors.items()):
        if tensor.ndim > 0 and tensor.shape[0] == original_length:
            tensors[key] = tensor[:requested_length]

    metadata["Length"] = str(requested_length)

    if include_camera_cut:
        camera_cut_tensor = torch.tensor(camera_cut_flags, dtype=torch.bool).view(
            requested_length, -1
        )
        if camera_cut_tensor.shape[1] != 1:
            camera_cut_tensor = camera_cut_tensor[:, :1]
        tensors["camera_cut"] = camera_cut_tensor
    else:
        tensors.pop("camera_cut", None)

    save_file(tensors, target_path, metadata=metadata)
    return root_dir


def compute_expected_segments(
    camera_cut_flags: Sequence[bool], n_frames: int
) -> List[tuple[int, int]]:
    """Mirror NSSDataset sequence segmentation for assertions."""

    segments: List[tuple[int, int]] = []
    start = 0
    for idx, flag in enumerate(camera_cut_flags):
        if idx == start:
            continue
        if flag:
            if idx - start >= n_frames:
                segments.append((start, idx))
            start = idx
    if len(camera_cut_flags) - start >= n_frames:
        segments.append((start, len(camera_cut_flags)))
    return segments
