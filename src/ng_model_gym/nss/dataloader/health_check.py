# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging

import torch
from torcheval.metrics.functional import peak_signal_noise_ratio as psnr
from tqdm import tqdm

from ng_model_gym.nss.dataloader.utils import DataLoaderMode
from ng_model_gym.nss.model.layers.dense_warp import DenseWarp
from ng_model_gym.nss.model.layers.down_sampling_2d import DownSampling2D

logger = logging.getLogger(__name__)

HIGH_PSNR_FOR_STATIC_WARP = 70.0


def health_check_dataset(
    dataset: torch.utils.data.DataLoader,
    loader_mode: DataLoaderMode,
    num_passes_required: float = 0.9,
):
    """Health check mvs and jitter on training dataset
    Args:
        dataset: PyTorch dataloader containing a dataset
        loader_mode (str): What mode to load the dataset in e.g. "train", "test", "val"
        num_passes_required (float): percentage expected to pass before allowing training
    """
    if loader_mode != DataLoaderMode.TRAIN:
        logger.info("DATASET: Health check is only supported for the train dataset")
        return

    logger.info("DATASET: Running Health Check...")
    (x, y) = next(iter(dataset))
    batch_size = y.shape[0]
    time_steps = y.shape[1]
    total_examples = (time_steps - 1) * batch_size
    required_passes = int(total_examples * num_passes_required)

    test_results = {"flow_passed": 0, "jitter_passed": 0}
    with tqdm(total=time_steps - 1) as pbar:
        for t in range(time_steps):
            if t == 0:
                gt_tm1 = y[:, 0, ...]
                continue

            # Grab Assets to test
            gt = y[:, t, ...]
            motion = x["motion"][:, t, ...]
            jittered = x["colour_linear"][:, t, ...]
            jitter = x["jitter"][:, t, ...]

            # Check Flow by warping GT t-1 -> t and comparing with t
            warped = DenseWarp()([gt_tm1, motion])

            psnr_before_warp = []
            psnr_after_warp = []
            for i in range(batch_size):
                psnr_before_warp.append(psnr(gt[i], gt_tm1[i], data_range=1.0))
                psnr_after_warp.append(psnr(gt[i], warped[i], data_range=1.0))

            psnr_before_warp = torch.stack(psnr_before_warp, dim=0)
            psnr_after_warp = torch.stack(psnr_after_warp, dim=0)
            # Handle static scene special case where sequences PSNR before is `inf`
            # but due to mv noise `psnr_after_warp` is very high value and not `inf`
            static_pass_cond = torch.logical_and(
                torch.isinf(psnr_before_warp),
                psnr_after_warp > HIGH_PSNR_FOR_STATIC_WARP,
            )
            passed_cases = torch.where(
                torch.logical_or(psnr_after_warp >= psnr_before_warp, static_pass_cond),
                1.0,
                0.0,
            )

            test_results["flow_passed"] += torch.sum(passed_cases).numpy()

            # Check Jitter by unjittering low-res frame and compare to downsampled GT
            gt_lr = DownSampling2D(size=2.0, interpolation="bilinear")(gt)
            unjittered = DenseWarp()([jittered, jitter])

            psnr_jittered = []
            psnr_unjittered = []
            for i in range(batch_size):
                psnr_jittered.append(psnr(gt_lr[i], jittered[i], data_range=1.0))
                psnr_unjittered.append(psnr(gt_lr[i], unjittered[i], data_range=1.0))

            psnr_jittered = torch.stack(psnr_jittered, dim=0)
            psnr_unjittered = torch.stack(psnr_unjittered, dim=0)

            passed_cases = torch.where(psnr_unjittered >= psnr_jittered, 1.0, 0.0)
            test_results["jitter_passed"] += torch.sum(passed_cases).numpy()

            # Store previous GT
            gt_tm1 = gt

            # Update TQDM
            pbar.update(1)
            pbar.set_postfix_str(test_results)

    assert (
        test_results["flow_passed"] > required_passes
    ), "Flow failed to pass all warping tests"
    assert (
        test_results["jitter_passed"] > required_passes
    ), "Jitter failed to pass all warping tests"
