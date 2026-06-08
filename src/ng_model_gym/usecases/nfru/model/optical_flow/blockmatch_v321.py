# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

"""BlockMatch v3.2.1 optical flow for NFRU."""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ng_model_gym.core.model.layers.dense_warp import DenseWarp
from ng_model_gym.usecases.nfru.model.constants import (
    _ARGMIN_OFFSET_INTENSITY,
    _BILINEAR_INTERPOLATION,
    _BINOMIAL_NORMALIZER,
    _BLOCKMATCH_BLUR_LEVELS,
    _BLOCKMATCH_LAST_BM_LEVEL,
    _BLOCKMATCH_LEVELS,
    _BLOCKMATCH_MEDIAN_KERNEL,
    _BLOCKMATCH_SEARCH_RANGE,
    _BLOCKMATCH_TEMPLATE_SIZE,
    _DEFAULT_SCALE_FACTOR,
    _HALF_GRADIENT_SCALE,
    _JOINT_BILATERAL_KERNEL_SIZE,
    _JOINT_BILATERAL_SIGMA_INTENSITY,
    _JOINT_BILATERAL_SIGMA_SPATIAL,
    _KERNEL_SIZE_3,
    _LUMA_GREEN_WEIGHT,
    _MAX_SUBPIXEL_MAGNITUDE,
    _NEAREST_INTERPOLATION,
    _SUBPIXEL_EPS,
    _UINT8_MAX,
)
from ng_model_gym.usecases.nfru.utils.down_sampling_2d import DownSampling2D


def cast(img: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert tensors while preserving BlockMatch's uint8/float conventions."""
    in_type = img.dtype
    if in_type in (torch.float16, torch.float32) and dtype == torch.uint8:
        img = torch.clamp(torch.round(img * _UINT8_MAX), 0.0, _UINT8_MAX)
    img = img.to(dtype)
    if in_type == torch.uint8 and dtype in (torch.float16, torch.float32):
        img = img / _UINT8_MAX
    return img


def gradient(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicate-padded centered finite differences."""
    f_pad = F.pad(f, (1, 1, 1, 1), mode="replicate")
    grad_y = (f_pad[:, :, 2:, 1:-1] - f_pad[:, :, :-2, 1:-1]) / _HALF_GRADIENT_SCALE
    grad_x = (f_pad[:, :, 1:-1, 2:] - f_pad[:, :, 1:-1, :-2]) / _HALF_GRADIENT_SCALE
    return grad_y, grad_x


def _rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return 0.25 * (r + _LUMA_GREEN_WEIGHT * g + b)


def binomial_filter(image: torch.Tensor) -> torch.Tensor:
    """Separable [1 2 1] / 4 blur with replicate padding."""
    vert_padded = F.pad(image, (0, 0, 1, 1), mode="replicate")
    v_blur = (
        vert_padded[:, :, :-2, :]
        + _LUMA_GREEN_WEIGHT * image
        + vert_padded[:, :, 2:, :]
    ) / _BINOMIAL_NORMALIZER

    horiz_padded = F.pad(v_blur, (1, 1, 0, 0), mode="replicate")
    h_blur = (
        horiz_padded[:, :, :, :-2]
        + _LUMA_GREEN_WEIGHT * v_blur
        + horiz_padded[:, :, :, 2:]
    ) / _BINOMIAL_NORMALIZER
    return h_blur


def window(
    x: torch.Tensor,
    ksize: int = _KERNEL_SIZE_3,
    step: int = 1,
    stride: int = 1,
    mode: str = "constant",
) -> torch.Tensor:
    """Extract sliding windows as ``B x C x K*K x H x W``."""
    stride_h = stride_w = stride
    shape = x.shape

    pad_along_height = (
        max(ksize - stride_h, 0)
        if shape[2] % stride_h == 0
        else max(ksize - (shape[2] % stride_h), 0)
    )
    pad_along_width = (
        max(ksize - stride_w, 0)
        if shape[3] % stride_w == 0
        else max(ksize - (shape[3] % stride_w), 0)
    )
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
    out_h = int(np.ceil(shape[2] / float(stride)))
    out_w = int(np.ceil(shape[3] / float(stride)))
    return torch.reshape(patches, (shape[0], shape[1], ksize**2, out_h, out_w))


def upscale_and_dilate_flow(
    tensor_to_dilate: torch.Tensor,
    depth: torch.Tensor,
    scale: float = float(_DEFAULT_SCALE_FACTOR),
    kernel_size: int = _KERNEL_SIZE_3,
    interpolation: str = _NEAREST_INTERPOLATION,
    is_flow: bool = True,
) -> torch.Tensor:
    """Dilate by nearest depth then upsample, matching NFRU preprocessing."""
    depth_windows = window(depth, ksize=kernel_size, mode="reflect")
    flow_windows = window(tensor_to_dilate, ksize=kernel_size, mode="reflect")
    min_depth_idx = torch.argmin(depth_windows, dim=2).unsqueeze(2)
    index = torch.tile(min_depth_idx, (1, tensor_to_dilate.shape[1], 1, 1, 1))
    dilated = torch.gather(flow_windows, dim=2, index=index).squeeze(2)

    out_size = (
        int(tensor_to_dilate.shape[2] * scale),
        int(tensor_to_dilate.shape[3] * scale),
    )
    if interpolation == "nearest":
        dilated = F.interpolate(dilated, size=out_size, mode="nearest")
    else:
        dilated = F.interpolate(
            dilated, size=out_size, mode=interpolation, align_corners=False
        )
    if is_flow:
        dilated = dilated * scale
    return dilated


def calculate_subpixel(
    f: torch.Tensor, g: torch.Tensor, f_img: torch.Tensor, template_sz: int
) -> torch.Tensor:
    """Lucas-Kanade style subpixel refinement."""
    f = cast(f, dtype=torch.float16)
    g = cast(g, dtype=torch.float16)
    f_img = cast(f_img, dtype=torch.float16)

    dfy_img, dfx_img = gradient(f_img)
    dfy = ExtractTemplates(template_sz=template_sz)(dfy_img, dtype=torch.float16)
    dfx = ExtractTemplates(template_sz=template_sz)(dfx_img, dtype=torch.float16)

    pred = torch.zeros(
        (template_sz - 2, template_sz - 2), dtype=torch.float16, device=f.device
    )
    pred = F.pad(pred, (1, 1, 1, 1), mode="constant", value=1)
    pred = pred.reshape(-1)

    def redsum(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=-1, keepdim=True).to(torch.float32)

    a = redsum((dfx**2) * pred)
    b = redsum(dfx * dfy * pred)
    d = redsum((dfy**2) * pred)

    z = g - f
    p = redsum(z * dfx * pred)
    q = redsum(z * dfy * pred)

    det_a = a * d - b * b

    subpixel_u = d * p - b * q
    subpixel_v = a * q - b * p
    subpixel = torch.cat([subpixel_v, subpixel_u], dim=-1)
    subpixel = subpixel / (
        det_a + torch.tensor(0.0, dtype=torch.float32, device=subpixel.device)
    )

    invalid_mask = (det_a <= _SUBPIXEL_EPS) | (
        torch.abs(subpixel) >= _MAX_SUBPIXEL_MAGNITUDE
    )
    subpixel = torch.where(
        invalid_mask,
        torch.tensor(0.0, dtype=torch.float32, device=subpixel.device),
        subpixel,
    )
    subpixel = subpixel.squeeze(-2)
    return subpixel.to(torch.float16).permute(0, 3, 1, 2).contiguous()


def spiral_pattern(sz_y: int, sz_x: int) -> np.ndarray:
    """Create a spiral traversal pattern of shape (2*sz_y+1, 2*sz_x+1)."""
    pattern = np.zeros((2 * sz_y + 1, 2 * sz_x + 1), dtype=np.float32)

    radius = sz_y
    dy_end = radius + 1
    n = 0

    for dy in range(dy_end):
        for direction in range(4):
            dx_start = -dy
            limit = 1 if (direction == 0 and dy == 0) else dy
            dx_end = limit
            for dx in range(dx_start, dx_end):
                if direction == 0:
                    wx_b = radius + dx
                    wy_b = radius + dy
                elif direction == 1:
                    wx_b = radius + dy
                    wy_b = radius - dx
                elif direction == 2:
                    wx_b = radius - dx
                    wy_b = radius - dy
                else:
                    wx_b = radius - dy
                    wy_b = radius + dx
                n += 1
                pattern[wy_b, wx_b] = n

    return pattern


class ArgMinCentered(nn.Module):
    """Argmin with center-aware tie-breaking using multiple patterns."""

    def forward(
        self,
        inputs: torch.Tensor,
        sz: int,
        mode: str = "default",
        cv_offset: bool = False,
        offset_intensity: float = _ARGMIN_OFFSET_INTENSITY,
    ) -> torch.Tensor:
        """Return int32 argmin indices over the last dimension."""
        x = inputs
        device = x.device
        dtype = torch.float32

        dim = sz * 2 + 1
        num_ch = dim * dim

        if cv_offset:
            rng = torch.arange(-sz, sz + 1, device=device, dtype=dtype)
            gy, gx = torch.meshgrid(rng, rng, indexing="ij")
            pattern = torch.abs(gy) + torch.abs(gx)
            offset_mask = pattern.reshape(num_ch).to(dtype) * offset_intensity
        else:
            offset_mask = torch.zeros((num_ch,), dtype=dtype, device=device)

        x = x.to(dtype)
        x = x + offset_mask
        min_val, _ = torch.min(x, dim=-1, keepdim=True)
        default_mask = (x == min_val).to(dtype)

        if mode == "default":
            mask = default_mask
        else:
            rng = torch.arange(-sz, sz + 1, device=device, dtype=dtype)
            gy, gx = torch.meshgrid(rng, rng, indexing="ij")

            if mode == "square":
                pattern = torch.maximum(torch.abs(gy), torch.abs(gx))
            elif mode == "diamond":
                pattern = torch.abs(gy) + torch.abs(gx)
            elif mode == "radial":
                pattern = gy.pow(2) + gx.pow(2)
            elif mode == "spiral":
                pattern_np = spiral_pattern(sz, sz)
                pattern = torch.tensor(pattern_np, dtype=dtype, device=device)
            else:
                raise ValueError("mode for ArgMinCentered not recognized")

            epsilon = torch.tensor(_SUBPIXEL_EPS, dtype=dtype, device=device)
            central_bias = (pattern.reshape(num_ch) * epsilon).to(dtype)
            central_mask = default_mask - central_bias
            central_mask_max, _ = torch.max(central_mask, dim=-1, keepdim=True)
            mask = torch.where(
                central_mask == central_mask_max,
                torch.tensor(1.0, dtype=dtype, device=device),
                torch.tensor(0.0, dtype=dtype, device=device),
            ).to(dtype)

        # Final deterministic tie-break by channel index (prefer smallest index).
        epsilon = torch.tensor(_SUBPIXEL_EPS, dtype=dtype, device=device)
        bias = (
            torch.linspace(0, num_ch - 1, steps=num_ch, dtype=dtype, device=device)
            * epsilon
        ).to(dtype)
        final_mask = mask - bias
        final_mask_max, _ = torch.max(final_mask, dim=-1, keepdim=True)

        single_val = torch.where(
            final_mask == final_mask_max,
            torch.tensor(1.0, dtype=dtype, device=device),
            torch.tensor(0.0, dtype=dtype, device=device),
        )

        ch_range = torch.arange(num_ch, dtype=dtype, device=device)
        min_idx = (single_val * ch_range).sum(dim=-1).to(torch.int32)
        return min_idx


class ExtractSearchWindows(nn.Module):
    """Build local search windows from input patches for block matching."""

    def __init__(self, template_sz: int, max_sr: int = _BLOCKMATCH_SEARCH_RANGE, **kw):
        super().__init__(**kw)
        self.template_sz = template_sz
        self.max_sr = max_sr
        self.max_sz = 2 * max_sr + 1

    def forward(self, inputs: torch.Tensor, search_range: int) -> torch.Tensor:
        """Extract per-pixel candidate patches for local block matching."""
        pad = self.max_sz // 2
        inputs = F.pad(inputs, (pad, pad, pad, pad), mode="constant", value=0)

        # First patch extraction.
        k = self.template_sz
        patches1 = F.unfold(
            inputs.to(torch.float32), kernel_size=(k, k), stride=1, padding=k // 2
        )
        batch = patches1.shape[0]
        h_p = inputs.shape[2]
        w_p = inputs.shape[3]
        patches1 = patches1.view(batch, k * k, h_p, w_p)

        # Second patch extraction over patch grid.
        s = self.max_sz
        patches2 = F.unfold(patches1, kernel_size=(s, s), stride=1, padding=0)
        h0 = h_p - 2 * pad
        w0 = w_p - 2 * pad
        patches2 = patches2.view(batch, k * k, s, s, h0, w0)
        windows_max = patches2.permute(0, 4, 5, 2, 3, 1).contiguous()

        cost_volume_sz = 2 * search_range + 1
        offset = self.max_sr - search_range
        r = torch.arange(offset, offset + cost_volume_sz, device=inputs.device)
        windows = windows_max[:, :, :, r][:, :, :, :, r]
        b2, h2, w2, _, _, _ = windows.shape
        windows = windows.reshape(b2, h2, w2, cost_volume_sz * cost_volume_sz, k * k)
        return windows.to(torch.uint8)


class ExtractTemplates(nn.Module):
    """Extract per-pixel template patches used in matching and refinement."""

    def __init__(self, template_sz: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.template_sz = template_sz

    def forward(self, inputs: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Extract centered template patches at each pixel location."""
        shape = inputs.shape
        inputs = cast(inputs, dtype=dtype)
        k = self.template_sz
        patches = F.unfold(
            inputs.to(torch.float32), kernel_size=(k, k), stride=1, padding=k // 2
        )
        batch = patches.shape[0]
        height, width = shape[2], shape[3]
        patches = patches.view(batch, -1, height, width)
        patches = patches[:, : k * k, :, :]
        patches = patches.permute(0, 2, 3, 1).unsqueeze(-2)
        return patches.to(dtype)


class CalculateVector(nn.Module):
    """Select best-matching displacement vectors from local cost volumes."""

    def __init__(self, search_range: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_range = search_range
        self.argmin_centered = ArgMinCentered()

    def forward(self, inputs):
        """Convert windows and templates into vectors and winning templates."""
        w1, w2 = inputs
        # w1: [B,H,W,C,K2], w2: [B,H,W,1,K2]
        batch, height, width, _, kernel_sq = w1.shape

        w1_i16 = w1.to(torch.int16)
        w2_i16 = w2.to(torch.int16)

        rng = torch.arange(-self.search_range, self.search_range + 1, device=w1.device)
        jj, ii = torch.meshgrid(rng, rng, indexing="ij")
        vec_lut = -1.0 * torch.stack([jj, ii], dim=-1).reshape(-1, 2).to(torch.float16)

        # SAD cost volume over template window.
        diff = torch.abs(w1_i16 - w2_i16)
        cost_volume_i32 = torch.sum(diff, dim=-1, dtype=torch.int32)

        input_mv_idx = (2 * self.search_range + 1) ** 2

        # ArgMinCentered over cost_volume.
        min_idx_block_match = self.argmin_centered(
            cost_volume_i32[..., :input_mv_idx], sz=self.search_range, mode="spiral"
        )
        min_idx_block_match = min_idx_block_match.to(torch.long).unsqueeze(-1)

        # Gather min cost from block match.
        idx_bm_exp = min_idx_block_match.expand(batch, height, width, 1)
        min_cost_block_match = torch.gather(cost_volume_i32, dim=3, index=idx_bm_exp)

        last_dim = cost_volume_i32.shape[-1]
        has_input_mv = (last_dim - 1) == input_mv_idx

        if has_input_mv:
            cost_at_input_mv = cost_volume_i32[..., -1:].contiguous()
            input_mv_mask = cost_at_input_mv < min_cost_block_match
            input_mv_idx_tensor = torch.full_like(
                min_idx_block_match, fill_value=input_mv_idx, dtype=torch.long
            )
            min_idx = torch.where(
                input_mv_mask, input_mv_idx_tensor, min_idx_block_match
            )
            min_cost_volume = torch.minimum(cost_at_input_mv, min_cost_block_match)
        else:
            input_mv_mask = torch.zeros_like(min_cost_block_match, dtype=torch.bool)
            min_idx = min_idx_block_match
            min_cost_volume = min_cost_block_match

        # Gather vector from LUT using block_match index.
        vector = vec_lut[min_idx_block_match.squeeze(-1)]

        # Gather min templates using min_idx.
        idx = min_idx.unsqueeze(-1).expand(batch, height, width, 1, kernel_sq)
        min_templates = torch.gather(w1, dim=3, index=idx)

        return vector, min_templates, input_mv_mask, min_cost_volume


class JointBilateralFilter(nn.Module):
    """Edge-aware smoothing for flow vectors guided by the template image."""

    def __init__(
        self,
        kernel_sz: int,
        sigma_spatial: float,
        sigma_intensity: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kernel_sz = kernel_sz
        sigma_spatial_div = 1.0 / (2.0 * (sigma_spatial**2.0))
        self.register_buffer(
            "kernel",
            -sigma_spatial_div
            * torch.ones((1, kernel_sz * kernel_sz), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "sigma_intensity_div",
            torch.tensor(1.0 / (2.0 * (sigma_intensity**2.0)), dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _extract_block(
        x: torch.Tensor, sz: int, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        n, c, h, w = x.shape
        x = cast(x, dtype=torch.float32)
        patches = F.unfold(x, kernel_size=(sz, sz), stride=1, padding=sz // 2)
        patches = patches.view(n, c, sz * sz, h, w).permute(0, 3, 4, 2, 1)
        return patches.to(dtype)

    def forward(self, inputs):
        """Apply edge-aware smoothing to the current flow field."""
        t, vector_curr = inputs
        t = cast(t, dtype=torch.float32)
        vector_curr = cast(vector_curr, dtype=torch.float32)

        compare_pix = t.permute(0, 2, 3, 1).unsqueeze(-2)
        compare_kernel = self._extract_block(t, self.kernel_sz)
        intensity_diff = torch.sum((compare_pix - compare_kernel) ** 2, dim=-1)
        kernel = self.kernel.to(t.device)
        sigma_intensity_div = self.sigma_intensity_div.to(t.device)
        one = torch.tensor(1.0, dtype=torch.float32, device=t.device)
        zero = torch.tensor(0.0, dtype=torch.float32, device=t.device)
        coeff = torch.clamp(
            one - torch.abs(kernel - intensity_diff * sigma_intensity_div),
            zero,
            one,
        )
        coeff = coeff.unsqueeze(-1)

        vector_block = self._extract_block(vector_curr, self.kernel_sz)
        num = torch.sum(vector_block * coeff, dim=-2)
        den = torch.sum(coeff, dim=-2)
        out = num / den
        return out.to(torch.float16).permute(0, 3, 1, 2).contiguous()


def median_filter2d(
    x: torch.Tensor,
    ksize: tuple[int, int] = _BLOCKMATCH_MEDIAN_KERNEL,
    padding: str = "SYMMETRIC",
) -> torch.Tensor:
    """Small-window median filter used by the blockmatch pass."""
    kh, kw = ksize
    pad_mode = "replicate" if padding == "SYMMETRIC" else "constant"
    x_pad = F.pad(x, (kw // 2, kw // 2, kh // 2, kh // 2), mode=pad_mode, value=0)
    patches = F.unfold(x_pad, kernel_size=(kh, kw), stride=1, padding=0)
    batch, _, hw = patches.shape
    channels = x.shape[1]
    patches = patches.view(batch, channels, kh * kw, hw)
    if (kh * kw) % 2 == 1:
        k = (kh * kw + 1) // 2
        median = torch.kthvalue(patches, k, dim=2).values
    else:
        k1 = (kh * kw) // 2
        k2 = k1 + 1
        v1 = torch.kthvalue(patches, k1, dim=2).values
        v2 = torch.kthvalue(patches, k2, dim=2).values
        median = (v1 + v2) / 2.0
    out = median.view(batch, channels, x.shape[2], x.shape[3])
    return out.to(x.dtype)


class CalculateFlow(nn.Module):
    """Compute multi-scale optical flow with block matching and optional MV hints."""

    def __init__(
        self,
        levels: int = _BLOCKMATCH_LEVELS,
        template_sz: int = _BLOCKMATCH_TEMPLATE_SIZE,
        search_range: int = _BLOCKMATCH_SEARCH_RANGE,
        median_kernel: tuple[int, int] = _BLOCKMATCH_MEDIAN_KERNEL,
        blur_levels: Optional[list[int]] = None,
        last_bm_level: int = _BLOCKMATCH_LAST_BM_LEVEL,
        performance_mode: str = "medium",
        min_cv_output: bool = False,
        mv_hints: bool = True,
        mean_flow_l1_norm_hint: float = 0.0,
    ):
        super().__init__()
        self.levels = levels
        self.search_range = search_range
        self.template_sz = template_sz
        self.median_kernel = median_kernel
        self.blur_levels = (
            blur_levels if blur_levels is not None else list(_BLOCKMATCH_BLUR_LEVELS)
        )
        self.last_bm_level = last_bm_level
        self.performance_mode = performance_mode
        self.min_cv_output = min_cv_output
        self.mv_hints = mv_hints
        self.mean_flow_l1_norm_hint = float(mean_flow_l1_norm_hint)

        self._down = DownSampling2D(
            size=_DEFAULT_SCALE_FACTOR, interpolation=_BILINEAR_INTERPOLATION
        )
        self._warp_bilinear = DenseWarp(interpolation=_BILINEAR_INTERPOLATION)

    @staticmethod
    def pad_to_even(
        tensors: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], dict[str, int]]:
        """Replicate-pad tensors so pyramid downsampling always sees even sizes."""
        height, width = tensors[0].shape[2], tensors[0].shape[3]
        h_pad = height % 2
        w_pad = width % 2
        padded = [F.pad(t, (0, w_pad, 0, h_pad), mode="replicate") for t in tensors]
        initial_shape = {"height": height, "width": width}
        return padded, initial_shape

    # The reference blockmatch pass has a fixed stage-by-stage control flow.
    # pylint: disable=too-many-branches
    def forward(
        self,
        search_frame: torch.Tensor,
        template_frame: torch.Tensor,
        input_mv: torch.Tensor,
        mean_flow_l1_norm_hint: Optional[Union[float, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Estimate flow over the multi-scale blockmatch pyramid."""
        additional_outputs: dict[str, torch.Tensor] = {}

        if mean_flow_l1_norm_hint is None:
            mean_flow_l1_norm_hint = self.mean_flow_l1_norm_hint
        elif isinstance(mean_flow_l1_norm_hint, torch.Tensor):
            mean_flow_l1_norm_hint = float(mean_flow_l1_norm_hint.detach().item())
        else:
            mean_flow_l1_norm_hint = float(mean_flow_l1_norm_hint)

        custom_search_range = self.search_range
        if mean_flow_l1_norm_hint > 0:
            scale = 2 ** (self.levels - 1 - self.last_bm_level)
            search_range = math.floor(mean_flow_l1_norm_hint / scale) + 1
            custom_search_range = int(min(max(search_range, 1), self.search_range))

        input_mv = input_mv.to(torch.float16)
        (search_frame, template_frame), initial_shape = self.pad_to_even(
            [search_frame, template_frame]
        )

        search_frame = cast(search_frame, dtype=torch.uint8)
        template_frame = cast(template_frame, dtype=torch.uint8)

        search_pyramid = [search_frame]
        template_pyramid = [template_frame]
        shape_pyramid = [initial_shape]

        for i in range(1, self.levels):
            prev_search_img = cast(search_pyramid[i - 1], dtype=torch.float32)
            prev_template_img = cast(template_pyramid[i - 1], dtype=torch.float32)

            if i - 1 in self.blur_levels:
                prev_search_img = binomial_filter(prev_search_img)
                prev_template_img = binomial_filter(prev_template_img)

            search_img_down = self._down(prev_search_img)
            template_img_down = self._down(prev_template_img)
            search_img_down = cast(search_img_down, dtype=torch.uint8)
            template_img_down = cast(template_img_down, dtype=torch.uint8)
            (search_img_down, template_img_down), shape_down = self.pad_to_even(
                [search_img_down, template_img_down]
            )

            search_pyramid.append(search_img_down)
            template_pyramid.append(template_img_down)
            shape_pyramid.append(shape_down)

        search_pyramid.reverse()
        template_pyramid.reverse()
        shape_pyramid.reverse()
        target_idx = (len(search_pyramid) - 1) - self.last_bm_level
        target_shape = search_pyramid[target_idx].shape

        mv_h, mv_w = input_mv.shape[-2], input_mv.shape[-1]
        target_h, target_w = target_shape[-2], target_shape[-1]
        pad_h = max(target_h - mv_h, 0)
        pad_w = max(target_w - mv_w, 0)
        if pad_h > 0 or pad_w > 0:
            input_mv = F.pad(input_mv, (0, pad_w, 0, pad_h), mode="replicate")
        input_mv = input_mv[:, :, :target_h, :target_w]

        vectors = []
        vector_prev = None

        for idx, (search_img, template_img) in enumerate(
            zip(search_pyramid, template_pyramid, strict=False)
        ):
            if idx != 0:
                vector_prev = vectors[idx - 1]
                vector_prev = cast(vector_prev, torch.float32)
                vector_prev = F.interpolate(
                    vector_prev,
                    scale_factor=float(_DEFAULT_SCALE_FACTOR),
                    mode=_BILINEAR_INTERPOLATION,
                    align_corners=False,
                ) * float(_DEFAULT_SCALE_FACTOR)

                search_img_f = cast(search_img, torch.float32)
                search_img_warped = self._warp_bilinear([search_img_f, vector_prev])
                search_img_warped = cast(search_img_warped, torch.uint8)
                vector_prev = cast(vector_prev, torch.float16)
            else:
                search_img_warped = search_img

            windows = ExtractSearchWindows(
                template_sz=self.template_sz, max_sr=self.search_range
            )(search_img_warped, search_range=custom_search_range)

            if self.mv_hints and idx == target_idx:
                mv_search_img = cast(search_img, torch.float32)
                mv_warped = self._warp_bilinear(
                    [mv_search_img, input_mv.to(torch.float32)]
                )
                mv_warped = cast(mv_warped, torch.uint8)
                mv_windows = ExtractTemplates(template_sz=self.template_sz)(
                    mv_warped, dtype=torch.uint8
                )
                windows = torch.cat([windows, mv_windows], dim=-2)

            templates = ExtractTemplates(template_sz=self.template_sz)(
                template_img, dtype=torch.uint8
            )

            (
                vector_curr,
                min_templates,
                input_mv_mask,
                min_cost_volume,
            ) = CalculateVector(search_range=custom_search_range)([windows, templates])

            # Subpixel estimation.
            subpixel_est = calculate_subpixel(
                templates, min_templates, template_img, self.template_sz
            )
            vector_curr = vector_curr.permute(0, 3, 1, 2).contiguous()
            vector_curr = vector_curr + subpixel_est

            if vector_prev is not None:
                vector_curr = vector_curr + vector_prev

            vector_curr = median_filter2d(
                vector_curr, ksize=self.median_kernel, padding="SYMMETRIC"
            )

            if self.performance_mode != "fast":
                vector_curr = JointBilateralFilter(
                    kernel_sz=_JOINT_BILATERAL_KERNEL_SIZE,
                    sigma_spatial=_JOINT_BILATERAL_SIGMA_SPATIAL,
                    sigma_intensity=_JOINT_BILATERAL_SIGMA_INTENSITY,
                )([template_img, vector_curr])

            # Replace predicted flow with hint MVs.
            if idx == target_idx:
                mask = input_mv_mask.permute(0, 3, 1, 2).expand_as(vector_curr)
                vector_curr = torch.where(mask, input_mv, vector_curr)

            shape = shape_pyramid[idx]
            h_t, w_t = shape["height"], shape["width"]
            vector_curr = vector_curr[:, :, :h_t, :w_t]
            min_cost_volume = min_cost_volume[:, :h_t, :w_t, :]
            if idx == target_idx:
                input_mv = input_mv[:, :, :h_t, :w_t]

            vectors.append(vector_curr)

            if idx == target_idx:
                if self.min_cv_output:
                    additional_outputs["min_cost_volume"] = min_cost_volume
                break

        return vector_curr, additional_outputs


class BlockMatchV321(nn.Module):
    """BlockMatch v3.2.1 wrapper used by examples and NFRU training."""

    def __init__(
        self,
        rgb_in: bool = True,
        output_polarity: str = "positive",
        mv_hints_polarity: str = "positive",
        template_frame_id: str = "tm1",
        granularity_scaling: bool = False,
        min_cv_output: bool = False,
        flow_params: Optional[dict] = None,
    ):
        super().__init__()
        self.rgb_in = rgb_in
        self.output_polarity = output_polarity
        self.mv_hints_polarity = mv_hints_polarity
        self.template_frame_id = template_frame_id
        self.granularity_scaling = granularity_scaling
        params = dict(flow_params or {})
        params.pop("din_sr", None)
        self.granularity = 2 ** int(params.get("last_bm_level", 2))
        self.min_cv_output = bool(params.get("min_cv_output", min_cv_output))
        self.calc_flow = CalculateFlow(**params, min_cv_output=self.min_cv_output)

    # Wrapper keeps explicit polarity and template-direction control flow.
    # pylint: disable=too-many-branches
    def forward(self, inputs: dict):
        """Run blockmatch-v321 on input dict and return an output dict."""
        # NCHW
        img_t = inputs["img_t"]
        img_tm1 = inputs["img_tm1"]
        input_mv = inputs["input_mv"]

        # Convert to grayscale if needed.
        if self.rgb_in:
            img_t = _rgb_to_y(img_t)
            img_tm1 = _rgb_to_y(img_tm1)

        # Decide frame direction.
        if self.template_frame_id == "tm1":
            search_frame = img_t
            template_frame = img_tm1
        elif self.template_frame_id == "t":
            search_frame = img_tm1
            template_frame = img_t
        else:
            raise ValueError(
                f"template_frame_id not recognised: {self.template_frame_id}"
            )

        # Set output flow polarity.
        if self.output_polarity == "positive":
            reversed_polarity = -1.0
        elif self.output_polarity == "negative":
            reversed_polarity = 1.0
        else:
            raise ValueError(
                "output_polarity not set to 'positive' or 'negative': "
                f"{self.output_polarity}"
            )

        # Set polarity for warps with mv hints.
        if self.mv_hints_polarity == "positive":
            warp_polarity = -1.0
        elif self.mv_hints_polarity == "negative":
            warp_polarity = 1.0
        else:
            raise ValueError(
                "mv_hints_polarity not set to 'positive' or 'negative': "
                f"{self.mv_hints_polarity}"
            )

        input_mv_scaled = input_mv
        if not self.granularity_scaling:
            input_mv_scaled = input_mv_scaled / float(self.granularity)

        mean_flow_l1_norm_hint = inputs.get("mean_flow_l1_norm_hint")
        if mean_flow_l1_norm_hint is not None and not self.granularity_scaling:
            mean_flow_l1_norm_hint = mean_flow_l1_norm_hint / float(self.granularity)

        flow_out, additional_outputs = self.calc_flow(
            search_frame=search_frame,
            template_frame=template_frame,
            input_mv=input_mv_scaled * warp_polarity,
            mean_flow_l1_norm_hint=mean_flow_l1_norm_hint,
        )
        flow_out = flow_out * reversed_polarity
        if not self.granularity_scaling:
            flow_out = flow_out * float(self.granularity)

        if bool(self.min_cv_output):
            final_output = {"output": flow_out, **additional_outputs}
        else:
            final_output = {"output": flow_out}

        return final_output
