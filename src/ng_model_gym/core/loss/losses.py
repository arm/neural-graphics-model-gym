# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Union

import lpips
import torch
import torch.nn.functional as F

from ng_model_gym.core.model.layers.dense_warp import DenseWarp
from ng_model_gym.core.utils.torch_utils import lerp_tensor

logger = logging.getLogger(__name__)


TensorData = Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]


class LossV0_1(torch.nn.Module):
    """Custom loss function."""

    def __init__(self, recurrent_samples: int, device):
        super().__init__()
        self.recurrent_samples = recurrent_samples
        self.device = device
        self.lpips_loss = lpips.LPIPS(net="vgg").to(self.device)
        self.warp = DenseWarp(interpolation="bilinear_oob_zero")
        self.l1 = torch.nn.L1Loss(reduction="none")

    def forward(self, y_true: torch.Tensor, y_pred: TensorData) -> torch.Tensor:
        """Forward pass."""

        y_pred_out = y_pred["output"]
        y_pred_filt = y_pred["out_filtered"]
        motion = y_pred["motion"]

        # Get dimensions
        n, t, c, h, w = y_pred_out.shape

        # Extract Tensor(s) and merge the temporal dimension into the batch dimension
        # This is because `self.warp` expects 4-dim input tensors
        y_pred_t = torch.reshape(y_pred_out[:, 1:, ...], ((-1, c, h, w)))
        y_true_t = torch.reshape(y_true[:, 1:, ...], ((-1, c, h, w)))
        y_pred_tm1 = torch.reshape(y_pred_out[:, :-1, ...], ((-1, c, h, w)))
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
                lpips_val = self.lpips_loss(y_pred_out[:, i, ...], y_true[:, i, ...])
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

        return loss


class LossV1(torch.nn.Module):
    """NSS v1 recurrent loss."""

    def __init__(
        self,
        recurrent_samples: int,
        device: torch.device,
        loss_args: Dict[str, Any],
    ):
        super().__init__()
        self.recurrent_samples = recurrent_samples
        self.device = device
        self.loss_args = loss_args or {}

        self.lpips_net = self._fetch_value(self.loss_args, "lpips_net", "alex")
        self.alpha = float(self._fetch_value(self.loss_args, "alpha", 4.0))
        self.lpips_weight = float(
            self._fetch_value(self.loss_args, "lpips_weight", 0.157)
        )
        self.filtered_weight = float(
            self._fetch_value(self.loss_args, "filtered_weight", 0.1)
        )
        self.min_weight = float(self._fetch_value(self.loss_args, "min_weight", 0.1))
        self.max_weight = float(self._fetch_value(self.loss_args, "max_weight", 0.9))

        self.theta_reg_weight = float(
            self._fetch_value(self.loss_args, "theta_reg_weight", 0.02)
        )
        self.theta_reg_channel = int(
            self._fetch_value(self.loss_args, "theta_reg_channel", 0)
        )
        self.theta_reg_target = float(
            self._fetch_value(self.loss_args, "theta_reg_target", 1.0)
        )
        self.alpha_reg_weight = float(
            self._fetch_value(self.loss_args, "alpha_reg_weight", 0.0001)
        )
        self.alpha_reg_channel = int(
            self._fetch_value(self.loss_args, "alpha_reg_channel", 1)
        )
        self.alpha_reg_target = float(
            self._fetch_value(self.loss_args, "alpha_reg_target", 0.0)
        )

        self.temporal_reg_weight = float(
            self._fetch_value(self.loss_args, "temporal_reg_weight", 0.7)
        )
        self.temporal_reg_channels = int(
            self._fetch_value(self.loss_args, "temporal_reg_channels", 1)
        )
        self.flicker_threshold = float(
            self._fetch_value(self.loss_args, "flicker_threshold", 0.02)
        )

        self.change_pred_weight = float(
            self._fetch_value(self.loss_args, "change_pred_weight", 0.7)
        )
        self.change_pred_sigma = float(
            self._fetch_value(self.loss_args, "change_pred_sigma", 0.06)
        )
        self.change_pred_eps = float(
            self._fetch_value(self.loss_args, "change_pred_eps", 1e-3)
        )
        self.change_pred_use_mask = self._as_bool(
            self._fetch_value(self.loss_args, "change_pred_use_mask", False)
        )
        self.change_pred_detach_w = self._as_bool(
            self._fetch_value(self.loss_args, "change_pred_detach_w", True)
        )
        self.change_pred_timesteps = int(
            self._fetch_value(self.loss_args, "change_pred_timesteps", 0)
        )
        self.first_frame_weight = float(
            self._fetch_value(self.loss_args, "first_frame_weight", 0.5)
        )

        self.disocclusion_threshold = float(
            self._fetch_value(self.loss_args, "temporal_disocclusion_threshold", 0.01)
        )
        self.mask_min_coverage = float(
            self._fetch_value(self.loss_args, "temporal_mask_min_coverage", 0.5)
        )
        self.mask_interp_mode = self._fetch_value(
            self.loss_args, "temporal_mask_interp", "nearest"
        )
        self.use_abs_depth_delta = self._as_bool(
            self._fetch_value(self.loss_args, "temporal_mask_use_abs_diff", False)
        )
        self.eps = 1e-6

        self.l1 = torch.nn.L1Loss(reduction="none")
        self.warp = DenseWarp(interpolation="bilinear_oob_zero")
        self.lpips_loss = lpips.LPIPS(net=self.lpips_net).to(self.device)

    def forward(self, y_true: torch.Tensor, y_pred: TensorData) -> torch.Tensor:
        """Compute the NSS v1 recurrent loss."""

        loss = self._compute_base_loss(y_true, y_pred)

        if self.theta_reg_weight > 0.0:
            loss = loss + self.theta_reg_weight * self._theta_regression_penalty(y_pred)
        if self.alpha_reg_weight > 0.0:
            loss = loss + self.alpha_reg_weight * self._alpha_regression_penalty(y_pred)
        if self.temporal_reg_weight > 0.0:
            loss = loss + self.temporal_reg_weight * self._temporal_flicker_penalty(
                y_pred
            )
        if self.change_pred_weight > 0.0:
            loss = loss + self.change_pred_weight * self._change_prediction_penalty(
                truth=y_true,
                pred=y_pred.get("output"),
                motion=y_pred.get("motion"),
                disocclusion_mask=y_pred.get("disocclusion_mask"),
            )

        return loss

    def _compute_base_loss(
        self, y_true: torch.Tensor, y_pred: TensorData
    ) -> torch.Tensor:
        """Compute the base NSS loss"""

        if not isinstance(y_pred, dict):
            raise TypeError("LossV1 expects y_pred to be a prediction dictionary.")

        y_pred_out = y_pred["output"]
        y_pred_filt = y_pred["out_filtered"]
        motion = y_pred["motion"]
        device = y_pred_out.device
        self._ensure_device(device)

        n, t, c, h, w = y_pred_out.shape
        if t <= 1:
            raise ValueError(
                "LossV1 expects temporal sequences with length greater than 1."
            )
        if motion.ndim == 5 and motion.shape[2] == 2 and motion.shape[-2:] != (h, w):
            raise ValueError(
                "LossV1 motion spatial dimensions mismatch: expected "
                f"{(h, w)}, got {tuple(motion.shape[-2:])}."
            )

        y_pred_t = y_pred_out[:, 1:, ...].reshape(-1, c, h, w)
        y_true_t = y_true[:, 1:, ...].reshape(-1, c, h, w)
        y_pred_tm1 = y_pred_out[:, :-1, ...].reshape(-1, c, h, w)
        y_true_tm1 = y_true[:, :-1, ...].reshape(-1, c, h, w)
        motion_t = motion[:, 1:, ...].reshape(-1, 2, h, w)

        loss_temporal = self._compute_temporal_loss(
            y_pred_t=y_pred_t,
            y_true_t=y_true_t,
            y_pred_tm1=y_pred_tm1,
            y_true_tm1=y_true_tm1,
            motion_t=motion_t,
            n=n,
            t=t,
            c=c,
            h=h,
            w=w,
        )
        loss_spatial_l1 = self._compute_spatial_l1(
            y_pred_t=y_pred_t, y_true_t=y_true_t, n=n, t=t, c=c, h=h, w=w
        )
        loss_spatial_lpips = self._compute_spatial_lpips(
            y_pred=y_pred_out, y_true=y_true, t=t, device=device
        )
        loss_spatial = loss_spatial_l1 + self.lpips_weight * loss_spatial_lpips

        st_weight = self._compute_st_weight(device=device, t=t)
        loss_spatial_temporal = lerp_tensor(
            loss_temporal, loss_spatial, st_weight
        ).mean()

        loss_filtered = self.l1(y_pred_filt, y_true).mean()
        loss = loss_spatial_temporal + self.filtered_weight * loss_filtered

        first_frame_loss = self._compute_first_frame_loss(
            y_pred_filt=y_pred_filt, y_true=y_true, device=device
        )
        return loss + self.first_frame_weight * first_frame_loss

    @staticmethod
    def _fetch_value(args: Any, key: str, default: Any) -> Any:
        """Fetch a loss argument while tolerating Config-like containers."""

        if not args:
            return default

        key_options = [key]
        for variant in (key.lower(), key.upper()):
            if variant not in key_options:
                key_options.append(variant)

        if isinstance(args, dict):
            for option in key_options:
                if option in args:
                    return args[option]
        else:
            getter = getattr(args, "get", None)
            if callable(getter):
                marker = object()
                for option in key_options:
                    value = getter(option, marker)
                    if value is not marker:
                        return value
            for attr_name in key_options:
                if hasattr(args, attr_name):
                    return getattr(args, attr_name)

        return default

    @staticmethod
    def _as_bool(value: Any) -> bool:
        """Convert config values to bool without treating 'false' as truthy."""

        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _ensure_device(self, device: torch.device) -> None:
        """Keep helper modules on the active prediction device."""

        self.device = device
        self.warp = self.warp.to(device)
        self.lpips_loss = self.lpips_loss.to(device)

    def _autocast_context(self, device: torch.device):
        if device.type == "cuda":
            return torch.autocast(device_type=device.type, dtype=torch.float16)
        return nullcontext()

    def _zero_scalar(self, *tensors: Optional[torch.Tensor]) -> torch.Tensor:
        for tensor in tensors:
            if tensor is not None:
                return tensor.new_zeros(())
        return torch.zeros((), device=self.device)

    def _compute_temporal_loss(
        self,
        *,
        y_pred_t: torch.Tensor,
        y_true_t: torch.Tensor,
        y_pred_tm1: torch.Tensor,
        y_true_tm1: torch.Tensor,
        motion_t: torch.Tensor,
        n: int,
        t: int,
        c: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        warped_y_pred_tm1 = self.warp([y_pred_tm1, motion_t])
        warped_y_true_tm1 = self.warp([y_true_tm1, motion_t])

        res_pred = y_pred_t - warped_y_pred_tm1
        res_true = y_true_t - warped_y_true_tm1
        loss_temporal = (
            torch.exp(self.alpha * torch.abs(res_true - res_pred)) - 1.0
        ).view(n, t - 1, c, h, w)
        return loss_temporal.mean(dim=(0, 2, 3, 4), keepdim=True)

    def _compute_spatial_l1(
        self,
        *,
        y_pred_t: torch.Tensor,
        y_true_t: torch.Tensor,
        n: int,
        t: int,
        c: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        loss_spatial_l1 = self.l1(y_pred_t, y_true_t).view(n, t - 1, c, h, w)
        return loss_spatial_l1.mean(dim=(0, 2, 3, 4), keepdim=True)

    def _compute_spatial_lpips(
        self,
        *,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        t: int,
        device: torch.device,
    ) -> torch.Tensor:
        loss_spatial_lpips_list = []
        with self._autocast_context(device):
            for step in range(1, t):
                loss_spatial_lpips_list.append(
                    self.lpips_loss(
                        y_pred[:, step, ...], y_true[:, step, ...], normalize=True
                    )
                )
        return torch.stack(loss_spatial_lpips_list, dim=1).float()

    def _compute_st_weight(self, *, device: torch.device, t: int) -> torch.Tensor:
        linspace = torch.linspace(1, t - 1, steps=t - 1, device=device)
        st_weight = (
            self.max_weight
            + (self.min_weight - self.max_weight)
            * (1.0 + torch.cos(math.pi * (linspace / (t - 1))))
            * 0.5
        )
        return st_weight.reshape(1, t - 1, 1, 1, 1)

    def _compute_first_frame_loss(
        self,
        *,
        y_pred_filt: torch.Tensor,
        y_true: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        pred0 = y_pred_filt[:, 0, ...]
        true0 = y_true[:, 0, ...]

        loss_l1 = self.l1(pred0, true0).mean()
        with self._autocast_context(device):
            lpips_val = self.lpips_loss(pred0, true0, normalize=True)
        return loss_l1 + self.lpips_weight * lpips_val.float().mean()

    def _build_temporal_weight_mask(  # pylint: disable=too-many-branches
        self,
        context: Dict[str, torch.Tensor],
        target_hw: tuple[int, int],
        device: torch.device,
        *,
        batch_size: int,
        timesteps: int,
    ) -> torch.Tensor:
        disocc = context.get("disocclusion_mask")
        reset_event = context.get("reset_event")
        num_pairs = max(timesteps - 1, 0) * batch_size
        if num_pairs == 0:
            return torch.ones((0, 1, *target_hw), device=device)

        if disocc is None or not isinstance(disocc, torch.Tensor):
            valid_mask = torch.ones((num_pairs, 1, *target_hw), device=device)
        else:
            if disocc.ndim < 2:
                raise ValueError(
                    "disocclusion_mask rank must include batch and timesteps "
                    f"dimensions; got shape {tuple(disocc.shape)}."
                )
            if disocc.shape[0] != batch_size:
                raise ValueError(
                    "disocclusion_mask batch mismatch: expected "
                    f"{batch_size}, got {disocc.shape[0]}."
                )
            if disocc.ndim not in (4, 5):
                raise ValueError(
                    "disocclusion_mask rank must be 4 or 5; got shape "
                    f"{tuple(disocc.shape)}."
                )
            if disocc.shape[1] <= 1:
                valid_mask = torch.ones((num_pairs, 1, *target_hw), device=device)
            else:
                if disocc.shape[1] != timesteps:
                    raise ValueError(
                        "disocclusion_mask timesteps mismatch: expected "
                        f"{timesteps}, got {disocc.shape[1]}."
                    )
                if disocc.ndim == 4:
                    disocc = disocc.unsqueeze(2)

                mask_curr = disocc[:, 1:, :1, ...].reshape(
                    -1, 1, disocc.shape[-2], disocc.shape[-1]
                )
                valid_mask = (
                    mask_curr.to(device=device) < self.disocclusion_threshold
                ).float()

                if valid_mask.shape[-2:] != target_hw:
                    interp_mode = (
                        "nearest" if min(target_hw) <= 1 else self.mask_interp_mode
                    )
                    valid_mask = F.interpolate(
                        valid_mask, size=target_hw, mode=interp_mode
                    )

        if reset_event is not None and isinstance(reset_event, torch.Tensor):
            if reset_event.ndim < 2:
                raise ValueError(
                    "reset_event rank must include batch and timesteps "
                    f"dimensions; got shape {tuple(reset_event.shape)}."
                )
            if reset_event.shape[0] != batch_size:
                raise ValueError(
                    "reset_event batch mismatch: expected "
                    f"{batch_size}, got {reset_event.shape[0]}."
                )
            if reset_event.ndim > 5:
                raise ValueError(
                    "reset_event rank must be between 2 and 5; got shape "
                    f"{tuple(reset_event.shape)}."
                )
            if reset_event.shape[1] <= 1:
                return valid_mask.clamp(min=0.0, max=1.0)
            if reset_event.shape[1] != timesteps:
                raise ValueError(
                    "reset_event timesteps mismatch: expected "
                    f"{timesteps}, got {reset_event.shape[1]}."
                )

            resets = reset_event[:, 1:, ...]
            while resets.ndim < 5:
                resets = resets.unsqueeze(-1)
            resets = resets[:, :, :1, ...].reshape(
                -1, 1, resets.shape[-2], resets.shape[-1]
            )
            resets = resets.to(device=device, dtype=valid_mask.dtype)
            if resets.shape[-2:] != target_hw:
                resets = F.interpolate(resets, size=target_hw, mode="nearest")
            valid_mask = valid_mask * (resets != 0).float()

        return valid_mask.clamp(min=0.0, max=1.0)

    def _theta_regression_penalty(
        self, context: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        temporal_params = context.get("temporal_params")
        motion = context.get("motion")
        if not isinstance(temporal_params, torch.Tensor) or temporal_params.ndim != 5:
            return self._zero_scalar(temporal_params, motion)

        b, t, c, h, w = temporal_params.shape
        if t <= 1 or self.theta_reg_channel < 0 or self.theta_reg_channel >= c:
            return temporal_params.new_zeros(())

        mask = self._build_temporal_weight_mask(
            context,
            (h, w),
            temporal_params.device,
            batch_size=b,
            timesteps=t,
        )
        theta = temporal_params[
            :, 1:, self.theta_reg_channel : self.theta_reg_channel + 1, :, :
        ].reshape(-1, 1, h, w)
        mask = mask.to(device=temporal_params.device, dtype=theta.dtype)
        weighted = (theta - self.theta_reg_target).pow(2) * mask
        return weighted.sum() / mask.sum().clamp_min(self.eps)

    def _alpha_regression_penalty(
        self, context: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        temporal_params = context.get("temporal_params")
        motion = context.get("motion")
        if not isinstance(temporal_params, torch.Tensor) or temporal_params.ndim != 5:
            return self._zero_scalar(temporal_params, motion)

        b, t, c, h, w = temporal_params.shape
        if t <= 1 or self.alpha_reg_channel < 0 or self.alpha_reg_channel >= c:
            return temporal_params.new_zeros(())

        mask = self._build_temporal_weight_mask(
            context,
            (h, w),
            temporal_params.device,
            batch_size=b,
            timesteps=t,
        )
        alpha = temporal_params[
            :, 1:, self.alpha_reg_channel : self.alpha_reg_channel + 1, :, :
        ].reshape(-1, 1, h, w)
        mask = mask.to(device=temporal_params.device, dtype=alpha.dtype)
        weighted = (alpha - self.alpha_reg_target).pow(2) * mask
        return weighted.sum() / mask.sum().clamp_min(self.eps)

    def _temporal_flicker_penalty(
        self, context: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        temporal_params = context.get("temporal_params")
        motion = context.get("motion")
        if (
            not isinstance(temporal_params, torch.Tensor)
            or not isinstance(motion, torch.Tensor)
            or temporal_params.ndim != 5
            or motion.ndim != 5
            or motion.shape[2] != 2
        ):
            return self._zero_scalar(temporal_params, motion)

        max_t = min(temporal_params.shape[1], motion.shape[1])
        if max_t <= 2 or self.temporal_reg_channels <= 0:
            return temporal_params.new_zeros(())

        tensor = temporal_params[:, :max_t, ...]
        motion = motion[:, :max_t, ...]
        b, t, c, h, w = tensor.shape
        ch = min(self.temporal_reg_channels, c)

        flow = motion.reshape(-1, 2, motion.shape[-2], motion.shape[-1])
        flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
        flow = flow.reshape(b, t, 2, h, w) * 0.5

        prev = tensor[:, :-2, :ch, :, :].reshape(-1, ch, h, w)
        curr = tensor[:, 1:-1, :ch, :, :].reshape(-1, ch, h, w)
        nxt = tensor[:, 2:, :ch, :, :].reshape(-1, ch, h, w)
        flow_to_curr = flow[:, 1:-1, :, :, :].reshape(-1, 2, h, w)
        flow_to_next = flow[:, 2:, :, :, :].reshape(-1, 2, h, w)

        prev_warp = self.warp([prev, flow_to_curr])
        curr_warp = self.warp([curr, flow_to_next])

        diff_prev = curr - prev_warp
        diff_next = nxt - curr_warp
        sign_flip = (
            (diff_prev * diff_next < 0)
            & (diff_prev.abs() > self.flicker_threshold)
            & (diff_next.abs() > self.flicker_threshold)
        )
        flicker = torch.minimum(diff_prev.abs(), diff_next.abs()) * sign_flip

        mask_pairs = self._build_temporal_weight_mask(
            context,
            (h, w),
            tensor.device,
            batch_size=b,
            timesteps=t,
        )
        if mask_pairs.numel() == 0:
            return tensor.new_zeros(())

        mask_pairs = mask_pairs.reshape(b, max(t - 1, 0), 1, h, w)
        mask_center = mask_pairs[:, 1:, ...] * mask_pairs[:, :-1, ...]
        if mask_center.shape[1] > 0:
            mask_center[:, 0, ...] = 0.0
        mask_center = mask_center.reshape(-1, 1, h, w).to(
            device=tensor.device, dtype=flicker.dtype
        )

        weighted = flicker.pow(2) * mask_center
        return weighted.sum() / (mask_center.sum() * ch).clamp_min(self.eps)

    def _change_prediction_penalty(  # pylint: disable=too-many-branches
        self,
        *,
        truth: torch.Tensor,
        pred: Optional[torch.Tensor],
        motion: Optional[torch.Tensor],
        disocclusion_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        tensors = tuple(
            tensor
            for tensor in (pred, truth, motion)
            if isinstance(tensor, torch.Tensor)
        )
        if (
            not isinstance(pred, torch.Tensor)
            or not isinstance(truth, torch.Tensor)
            or not isinstance(motion, torch.Tensor)
        ):
            return self._zero_scalar(*tensors)

        if pred.ndim != 5 or truth.ndim != 5 or motion.ndim != 5:
            return self._zero_scalar(pred, truth, motion)

        b, max_t, _channels, h, w = pred.shape
        if motion.shape[2] != 2:
            raise ValueError(
                "change prediction motion channel mismatch: expected shape "
                f"(batch, timesteps, 2, height, width), got {tuple(motion.shape)}."
            )
        if pred.shape[0] != truth.shape[0] or pred.shape[0] != motion.shape[0]:
            raise ValueError(
                "change prediction batch mismatch: pred batch "
                f"{pred.shape[0]}, truth batch {truth.shape[0]}, "
                f"motion batch {motion.shape[0]}."
            )
        if pred.shape[2] != truth.shape[2]:
            raise ValueError(
                "change prediction channel mismatch: pred channels "
                f"{pred.shape[2]}, truth channels {truth.shape[2]}."
            )
        if pred.shape[-2:] != truth.shape[-2:]:
            raise ValueError(
                "change prediction pred/truth spatial dimensions mismatch: "
                f"pred {tuple(pred.shape[-2:])}, truth {tuple(truth.shape[-2:])}."
            )
        if motion.shape[-2:] != (h, w):
            raise ValueError(
                "change prediction motion spatial dimensions mismatch: expected "
                f"{(h, w)}, got {tuple(motion.shape[-2:])}."
            )

        max_t = min(max_t, truth.shape[1], motion.shape[1])
        if max_t <= 1:
            return pred.new_zeros(())

        timesteps = self.change_pred_timesteps
        if timesteps <= 0:
            timesteps = max_t
        timesteps = min(timesteps, max_t)

        valid_mask = self._change_prediction_valid_mask(
            disocclusion_mask, batch_size=b, timesteps=timesteps, target_hw=(h, w)
        )

        total = pred.new_zeros(())
        count = 0
        aligned = []
        for step in range(1, timesteps):
            mv_step = motion[:, step, ...]

            if aligned:
                aligned = [self.warp([frame, mv_step]) for frame in aligned]

            prev_warped = self.warp([pred[:, step - 1, ...], mv_step])
            aligned.append(prev_warped)

            cur_pred = pred[:, step, ...]
            cur_truth = truth[:, step, ...]
            for warped in aligned:
                residual = cur_pred - warped
                err = (cur_truth - warped).abs()
                weight = torch.exp(-err / self.change_pred_sigma)
                if self.change_pred_detach_w:
                    weight = weight.detach()
                if valid_mask is not None:
                    weight = weight * valid_mask[:, step, ...]
                total = (
                    total
                    + torch.sqrt(
                        (weight * residual).pow(2) + self.change_pred_eps**2
                    ).mean()
                )
                count += 1

        if count == 0:
            return pred.new_zeros(())
        return total / count

    def _change_prediction_valid_mask(
        self,
        disocclusion_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        timesteps: int,
        target_hw: tuple[int, int],
    ) -> Optional[torch.Tensor]:
        if not self.change_pred_use_mask or not isinstance(
            disocclusion_mask, torch.Tensor
        ):
            return None

        if disocclusion_mask.ndim == 4:
            disocclusion_mask = disocclusion_mask.unsqueeze(2)
        elif disocclusion_mask.ndim != 5:
            raise ValueError(
                "change prediction disocclusion_mask rank must be 4 or 5; "
                f"got shape {tuple(disocclusion_mask.shape)}."
            )
        if disocclusion_mask.shape[0] != batch_size:
            raise ValueError(
                "change prediction disocclusion_mask batch mismatch: expected "
                f"{batch_size}, got {disocclusion_mask.shape[0]}."
            )
        if disocclusion_mask.shape[1] < timesteps:
            raise ValueError(
                "change prediction disocclusion_mask timesteps mismatch: expected "
                f"at least {timesteps}, got {disocclusion_mask.shape[1]}."
            )

        disocclusion_mask = disocclusion_mask[:, :timesteps, :1, ...]
        b, t, channels, mask_h, mask_w = disocclusion_mask.shape
        if (mask_h, mask_w) != target_hw:
            disocclusion_mask = F.interpolate(
                disocclusion_mask.reshape(b * t, channels, mask_h, mask_w),
                size=target_hw,
                mode="nearest",
            ).reshape(b, t, channels, *target_hw)

        return 1.0 - disocclusion_mask.to(device=self.device)


class LPIPSSpatialLossV1(torch.nn.Module):
    """Spatial loss used by the NFRU v1 reference pipeline."""

    def __init__(self, loss_args: Dict[str, float], device: torch.device):
        super().__init__()
        self.loss_args = loss_args
        if "alpha" not in self.loss_args:
            raise ValueError("LPIPSSpatialLossV1 requires 'alpha' in loss_args")

        self.alpha = float(self.loss_args["alpha"])
        self.device = device
        self.l1_loss = torch.nn.L1Loss(reduction="none")
        self.lpips_loss = lpips.LPIPS().to(self.device)

    def forward(self, y_true: torch.Tensor, y_pred: TensorData) -> torch.Tensor:
        """Blend L1 and LPIPS with the v1 configuration."""
        y_pred_out = y_pred["output"]

        loss_l1 = self.l1_loss(y_true, y_pred_out)
        loss_lpips = self.lpips_loss(y_true, y_pred_out, normalize=True)
        alpha_tensor = loss_l1.new_tensor(self.alpha)
        loss = lerp_tensor(loss_l1, loss_lpips, alpha_tensor)
        return loss.mean()
