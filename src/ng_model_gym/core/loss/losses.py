# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from typing import Any, Dict, List, Union

import lpips
import torch

from ng_model_gym.core.utils.general_utils import lerp_tensor
from ng_model_gym.usecases.nss.model.layers.dense_warp import DenseWarp

logger = logging.getLogger(__name__)


TensorData = Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]


class LossV1(torch.nn.Module):
    """Custom loss function."""

    def __init__(self, recurrent_samples: int, device):
        super().__init__()
        self.recurrent_samples = recurrent_samples
        self.device = device
        self.lpips_loss = lpips.LPIPS(net="vgg").to(self.device)
        self.warp = DenseWarp(interpolation="bilinear_oob_zero")
        self.l1 = torch.nn.L1Loss(reduction="none")

    def forward(
        self, y_true: torch.Tensor, y_pred_and_inps: TensorData
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | Any]]:
        """Forward pass."""
        y_pred = y_pred_and_inps["output"]
        y_pred_filt = y_pred_and_inps["out_filtered"]
        motion = y_pred_and_inps["motion"]

        # Get dimensions
        n, t, c, h, w = y_pred.shape

        # Extract Tensor(s) and merge the temporal dimension into the batch dimension
        # This is because `self.warp` expects 4-dim input tensors
        y_pred_t = torch.reshape(y_pred[:, 1:, ...], ((-1, c, h, w)))
        y_true_t = torch.reshape(y_true[:, 1:, ...], ((-1, c, h, w)))
        y_pred_tm1 = torch.reshape(y_pred[:, :-1, ...], ((-1, c, h, w)))
        y_true_tm1 = torch.reshape(y_true[:, :-1:, ...], ((-1, c, h, w)))
        motion_t = torch.reshape(motion[:, 1:, ...], ((-1, 2, h, w)))
        warped_y_pred_tm1 = self.warp([y_pred_tm1, motion_t])
        warped_y_true_tm1 = self.warp([y_true_tm1, motion_t])

        # Temporal Loss: exp((α * abs((Yt - W(Yt_tm1)) - (Yp - W(Yp_tm1)))) - 1.)
        # Ref: https://dl.acm.org/doi/pdf/10.1145/3528233.3530700
        alpha = 4.0
        res_pred = y_pred_t - warped_y_pred_tm1
        res_true = y_true_t - warped_y_true_tm1
        loss_temporal = (
            (torch.exp(alpha * abs(res_true - res_pred)) - 1.0)
            .view(n, t - 1, c, h, w)
            .mean(dim=(0, 2, 3, 4), keepdim=True)
        )

        # Spatial loss(es)
        # The direct output spatial loss is calculated on frames >1, because we only take
        # the low-resolution output frame when a reset event occurs (happens for every frame 0)
        loss_spatial_l1 = (
            self.l1(y_pred_t, y_true_t)
            .view(n, t - 1, c, h, w)
            .mean(dim=(0, 2, 3, 4), keepdim=True)
        )

        # LPIPS term
        # Elected to run this in half precision, as it gives a nice speed up to training flow,
        # that's not impacted accuracy.
        p_weight = 0.2
        loss_spatial_lpips_list = []
        with torch.autocast(device_type=str(self.device), dtype=torch.float16):
            for i in range(1, self.recurrent_samples):
                # The LPIPS call below runs in half precision
                lpips_val = self.lpips_loss(y_pred[:, i, ...], y_true[:, i, ...])
                loss_spatial_lpips_list.append(lpips_val)

        # Stack them up and convert back to float32 to merge with the rest of the loss
        loss_spatial_lpips = torch.stack(loss_spatial_lpips_list, dim=1).float()

        # Combine l1 and LPIPS
        loss_spatial = loss_spatial_l1 + p_weight * loss_spatial_lpips

        # Low-res single image filtered frame
        loss_spatial_filtered = self.l1(y_pred_filt, y_true).mean()

        # Generate Loss weighting,
        # idea: spatial loss weighting increases over the time dimension
        # tuned on desmos: https://www.desmos.com/calculator/orfecrnpna
        def weights(
            start: int, end: int, min_weight: float = 0.001, max_weight: float = 0.9
        ) -> torch.Tensor:
            t = torch.reshape(
                torch.linspace(start=start, end=end, steps=end, device=y_true.device),
                (1, end, 1, 1, 1),
            )
            return (
                max_weight
                + (min_weight - max_weight)
                * (1.0 + torch.cos(math.pi * (t / end)))
                * 0.5
            )

        # Merge the spatial and temporal terms with increasing contribution to the spatial
        # weight as time goes on (allows for temporally stable accumulation)
        st_weight = weights(start=1, end=t - 1)
        loss = loss_spatial_temporal = lerp_tensor(
            loss_temporal, loss_spatial, st_weight
        ).mean()

        # Add small contribution to the loss for the filtered frame, ensures it is learning
        # to anti-alias, even when the history is not rectified
        f_weight = 0.1
        loss = lerp_tensor(loss_spatial_temporal, loss_spatial_filtered, f_weight)

        metrics = {
            "loss_temporal": loss_temporal.mean(),
            "loss_spatial": loss_spatial.mean(),
            "loss_spatial_l1": loss_spatial_l1.mean(),
            "loss_spatial_lpips": loss_spatial_lpips.mean(),
            "loss_spatial_filtered": loss_spatial_filtered,
        }

        return loss, metrics
