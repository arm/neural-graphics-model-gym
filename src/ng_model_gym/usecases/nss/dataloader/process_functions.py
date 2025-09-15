# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Dict

import torch
import torchvision.transforms.functional as transforms

from ng_model_gym.core.data.utils import tonemap_forward, ToneMapperMode
from ng_model_gym.core.utils.general_utils import clamp_tensor

logger = logging.getLogger(__name__)


def process_nss_data(
    data_frame: Dict[str, torch.Tensor],
    augment: bool = False,
    tonemapper: ToneMapperMode = ToneMapperMode.REINHARD,
    exposure: float = None,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Take in raw Neural Super Sampling (NSS) Safetensors data.
    The input data contains LR/HR pairs already.
    Data should already be in a sequence of T frames e.g. shape: (T, C, H, W)
    Apply exposure and tonemapping to RGB data.
    Apply augmentations if required. Augmentations are applied to RGB+D+VU at the same time.

    Args:
        data_frame: Dict of tensors - colour, depth, motion, truth, jitter,
            seq_id, img_id, z_near, z_far, exposure etc.
        augment: Augment the data or not
        tonemapper: Which tonemapper to use.
        exposure: Optional exposure value to use.

    Returns:
        Tuple consisting of input dictionary of tensors and ground truth tensor.
    """

    # Helper to expand scalars to ndims=4
    make_image_like = lambda t: t.unsqueeze(-1).unsqueeze(-1)

    # Apply an Exposure in Linear space
    if exposure is None:
        data_frame["exposure"] = torch.exp(make_image_like(data_frame.pop("exposure")))
    else:
        # Exposure can be provided as a scalar,
        # so we multiply by `ones_like` of original `exposure` for correct shape
        data_frame["exposure"] = torch.exp(
            make_image_like(
                torch.tensor(exposure) * torch.ones_like(data_frame.pop("exposure"))
            )
        )

    # Decide random augmentations
    flip_horz = torch.randint(low=0, high=2, size=(), dtype=torch.int32, device="cpu")
    flip_vert = torch.randint(low=0, high=2, size=(), dtype=torch.int32, device="cpu")
    rot_ac_90 = torch.randint(low=0, high=2, size=(), dtype=torch.int32, device="cpu")

    processed_inputs = {}

    def process_motion(to_process):
        v, u = torch.split(to_process, split_size_or_sections=1, dim=1)
        if flip_horz == 1:
            v = transforms.hflip(v)
            u = -transforms.hflip(u)
        if flip_vert == 1:
            v = -transforms.vflip(v)
            u = transforms.vflip(u)
        if rot_ac_90 == 1:
            v_before = v
            u_before = u
            v = -torch.rot90(u_before, k=1, dims=[2, 3])
            u = torch.rot90(v_before, k=1, dims=[2, 3])
        to_process = torch.concat([v, u], dim=1)
        return to_process

    def process_jitter(to_process):
        y, x = torch.split(to_process, split_size_or_sections=1, dim=1)
        if flip_horz == 1:
            x = -x
        if flip_vert == 1:
            y = -y
        if rot_ac_90 == 1:
            x_before = x
            y_before = y
            y = -x_before
            x = y_before
        to_process = torch.concat([y, x], dim=1)
        return to_process

    for key, to_process in data_frame.items():
        if augment:
            # NOTE: vector polarities must be adjusted when the images are flipped
            if "motion" in key:
                to_process = process_motion(to_process)
            elif "jitter" in key:
                to_process = process_jitter(to_process)
            elif to_process.ndim == 4:
                if flip_horz == 1:
                    to_process = transforms.hflip(to_process)
                if flip_vert == 1:
                    to_process = transforms.vflip(to_process)
                if rot_ac_90 == 1:
                    to_process = torch.rot90(to_process, k=1, dims=[2, 3])

        # Expand Scalars to image-like i.e., ndims=4
        if to_process.ndim == 2:
            to_process = make_image_like(to_process)

        processed_inputs[key] = to_process

    data = {k: v.to(torch.float32) for k, v in processed_inputs.items()}

    # Extract ground truth
    exposure = data["exposure"]
    truth_linear = data["ground_truth_linear"]
    colour_linear = data["colour_linear"]

    # Clamp max values based on exposure
    max_val = 65_504.0 / exposure
    truth_linear = clamp_tensor(truth_linear, torch.zeros_like(max_val), max_val)
    colour_linear = clamp_tensor(colour_linear, torch.zeros_like(max_val), max_val)

    # Apply exposure
    tru_exp = exposure * truth_linear
    col_exp = exposure * colour_linear

    # Forward Tonemap w/ chosen tonemapper
    ground_truth = tonemap_forward(tru_exp, mode=tonemapper)
    data["colour"] = tonemap_forward(col_exp, mode=tonemapper)

    return data, ground_truth
