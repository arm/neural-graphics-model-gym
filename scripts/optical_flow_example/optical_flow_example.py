# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

"""Minimal example for running BlockMatchV32 optical flow on two frames.

Expects two RGB images specified via --img-tm1 (t-1) and --img-t (t).
The script builds an all-zero MV hint field, runs the model, and returns a
dict with the flow tensor under key "output".

This is saved to disk as a Numpy array.
"""

import argparse

import imageio.v3 as iio
import numpy as np
import torch

from ng_model_gym.usecases.nfru.model.optical_flow import BlockMatchV32


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optical Flow on two images.")
    parser.add_argument("--img-tm1", required=True, help="Path to the 't-1' frame.")
    parser.add_argument("--img-t", required=True, help="Path to the 't' frame.")
    parser.add_argument(
        "--out",
        default="optical_flow_output.npy",
        required=False,
        help="Path to save the output.",
    )

    return parser.parse_args()


def main():
    """Run optical flow on a pair of images that are passed in."""
    args = _parse_args()

    flow_params = {
        "levels": 6,
        "template_sz": 5,
        "search_range": 3,
        "median_kernel": (3, 3),
        "blur_levels": (1, 2, 3, 4),
        "last_bm_level": 2,
        "performance_mode": "medium",
        "mv_hints": False,
        "din_sr": True,
    }

    # Initialize the model with the chosen flow parameters.
    bm = BlockMatchV32(flow_params=flow_params)

    # Load RGB frames from disk (t-1 and t).
    img_tm1 = iio.imread(args.img_tm1)
    img_t = iio.imread(args.img_t)

    # Convert to NCHW float tensors in [0, 1].
    img_tm1 = torch.tensor(img_tm1).unsqueeze(0).permute((0, 3, 1, 2)).float() / 255.0
    img_t = torch.tensor(img_t).unsqueeze(0).permute((0, 3, 1, 2)).float() / 255.0

    # Motion vector hints for Optical Flow
    # Replace this with motion vectors, otherwise provide all-zero values
    # The mv_hints flag should also be set if we want to use motion vector hints.
    mv = torch.zeros(img_tm1.shape[0], 2, img_tm1.shape[2], img_tm1.shape[3]).float()

    # Pass t, t_m1 and motion vector hints in via a dict.
    bm_input_dict = {"img_t": img_t, "img_tm1": img_tm1, "input_mv": mv}

    # Run the model; output is a dict with "output" (flow) and optional extras.
    bm_output = bm(bm_input_dict)

    # [B, 2, H, W]
    optical_flow = bm_output["output"]

    # Convert to Numpy array and save to disk.
    np.save(args.out, optical_flow.cpu().numpy())


if __name__ == "__main__":
    main()
