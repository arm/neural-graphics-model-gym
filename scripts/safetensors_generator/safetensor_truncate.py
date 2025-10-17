# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import safetensors
import torch
from safetensors.torch import save_file


def truncate_safetensor_file(
    in_path: Path, out_path: Path, desired_frames: int
) -> None:
    """Truncate a safetensor file frames to desired amount"""
    print(f"Generating new safetensor file with {desired_frames} frames")
    if desired_frames <= 0:
        raise ValueError("desired_frames must be positive.")

    with safetensors.safe_open(in_path, framework="pt", device="cpu") as st_file:
        metadata = st_file.metadata()

        st_frame_length = int(metadata["Length"])
        if desired_frames > st_frame_length:
            raise ValueError(
                f"x ({desired_frames}) cannot exceed Length ({st_frame_length})."
            )

        out_tensors = {}
        for feature in st_file.keys():
            feature_tensor = st_file.get_tensor(feature)
            # Safetensor file stores each feature frame stacked
            # e.g colour_linear shape is [100, #, #, #]
            if feature_tensor.ndim >= 1 and feature_tensor.shape[0] == st_frame_length:
                # Slice the tensor to get the desired frame amount
                out_tensors[feature] = st_file.get_slice(feature)[:desired_frames]
            else:
                # If original feature is not X frames long, copy it over
                out_tensors[feature] = feature_tensor

        metadata["Length"] = str(desired_frames)

    save_file(out_tensors, out_path, metadata=metadata)

    # Validate write was successful
    with safetensors.safe_open(
        in_path, framework="pt", device="cpu"
    ) as validate_in, safetensors.safe_open(
        out_path, framework="pt", device="cpu"
    ) as validate_out:
        for feature in validate_out.keys():
            truncated_tensor = validate_out.get_tensor(feature)
            if (
                truncated_tensor.ndim >= 1
                and "Length" in metadata
                and truncated_tensor.shape[0] == desired_frames
            ):
                original_tensor = validate_in.get_slice(feature)[:desired_frames]
            else:
                original_tensor = validate_in.get_tensor(feature)
            if not (
                original_tensor.shape == truncated_tensor.shape
                and original_tensor.dtype == truncated_tensor.dtype
                and torch.equal(original_tensor, truncated_tensor)
            ):
                raise RuntimeError("Original and truncated tensor shapes do not match")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Truncate a safetensor dataset to a smaller number of frames."
    )
    parser.add_argument("input", type=Path, help="Path to the input safetensor file.")
    parser.add_argument(
        "output", type=Path, help="Destination path for the truncated safetensor file."
    )
    parser.add_argument(
        "--frames",
        "-f",
        type=int,
        default=20,
        help="Desired number of frames to keep (default: 20).",
    )
    return parser.parse_args()


def main() -> None:
    """Main function"""
    args = _parse_args()
    truncate_safetensor_file(args.input, args.output, args.frames)


if __name__ == "__main__":
    main()
