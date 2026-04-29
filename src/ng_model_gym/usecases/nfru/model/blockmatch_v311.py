# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

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
    _COST_CHUNK_SIZE,
    _DEFAULT_SCALE_FACTOR,
    _HALF_GRADIENT_SCALE,
    _JOINT_BILATERAL_KERNEL_SIZE,
    _JOINT_BILATERAL_SIGMA_INTENSITY,
    _JOINT_BILATERAL_SIGMA_SPATIAL,
    _KERNEL_SIZE_3,
    _LUMA_GREEN_WEIGHT,
    _MAX_SUBPIXEL_MAGNITUDE,
    _SUBPIXEL_EPS,
    _UINT8_MAX,
    _UINT8_MAX_INT,
)
from ng_model_gym.usecases.nfru.utils.down_sampling_2d import DownSampling2D

logger = logging.getLogger(__name__)


def cast(img: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Match the reference uint8/float conversion behavior."""
    in_type = img.dtype
    if in_type in (torch.float16, torch.float32) and dtype == torch.uint8:
        img = torch.clamp(torch.round(img * _UINT8_MAX), 0.0, _UINT8_MAX)
    img = img.to(dtype)
    if in_type == torch.uint8 and dtype in (torch.float16, torch.float32):
        img = img / _UINT8_MAX
    return img


def gradient(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite-difference image gradients."""
    f_pad = F.pad(f, (1, 1, 1, 1), mode="replicate")
    grad_y = (f_pad[:, :, 2:, 1:-1] - f_pad[:, :, :-2, 1:-1]) / _HALF_GRADIENT_SCALE
    grad_x = (f_pad[:, :, 1:-1, 2:] - f_pad[:, :, 1:-1, :-2]) / _HALF_GRADIENT_SCALE
    return grad_y, grad_x


def _rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return 0.25 * (r + _LUMA_GREEN_WEIGHT * g + b)


def binomial_filter(image: torch.Tensor) -> torch.Tensor:
    """Separable [1 2 1] / 4 blur with symmetric padding."""
    vert_padded = F.pad(image, (0, 0, 1, 1), mode="replicate")
    v_blur = (
        vert_padded[:, :, :-2, :]
        + _LUMA_GREEN_WEIGHT * image
        + vert_padded[:, :, 2:, :]
    ) / _BINOMIAL_NORMALIZER

    horiz_padded = F.pad(v_blur, (1, 1, 0, 0), mode="replicate")
    return (
        horiz_padded[:, :, :, :-2]
        + _LUMA_GREEN_WEIGHT * v_blur
        + horiz_padded[:, :, :, 2:]
    ) / _BINOMIAL_NORMALIZER


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
    scale: Optional[float] = float(_DEFAULT_SCALE_FACTOR),
    kernel_size: Optional[int] = _KERNEL_SIZE_3,
    interpolation: Optional[str] = "nearest",
    is_flow: Optional[bool] = True,
) -> torch.Tensor:
    """Dilate by nearest depth, then upsample like the reference helper."""
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
    template: torch.Tensor,
    matched: torch.Tensor,
    template_img: torch.Tensor,
    template_sz: int,
) -> torch.Tensor:
    """Lucas-Kanade style subpixel refinement over the best block match."""
    template = cast(template, dtype=torch.float16)
    matched = cast(matched, dtype=torch.float16)
    template_img = cast(template_img, dtype=torch.float16)

    grad_y_img, grad_x_img = gradient(template_img)
    grad_y = ExtractTemplates(template_sz=template_sz)(grad_y_img, dtype=torch.float16)
    grad_x = ExtractTemplates(template_sz=template_sz)(grad_x_img, dtype=torch.float16)

    pred = torch.zeros(
        (template_sz - 2, template_sz - 2), dtype=torch.float16, device=template.device
    )
    pred = F.pad(pred, (1, 1, 1, 1), mode="constant", value=1).reshape(-1)

    def redsum(x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=-1, keepdim=True).to(torch.float32)

    a = redsum((grad_x**2) * pred)
    b = redsum(grad_x * grad_y * pred)
    d = redsum((grad_y**2) * pred)

    diff = matched - template
    p = redsum(diff * grad_x * pred)
    q = redsum(diff * grad_y * pred)

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


class ArgMinCentered(nn.Module):
    """Argmin with center-aware tie breaking."""

    def forward(
        self,
        inputs: torch.Tensor,
        sz: int,
        mode: str = "default",
        cv_offset: bool = False,
        offset_intensity: float = _ARGMIN_OFFSET_INTENSITY,
    ) -> torch.Tensor:
        """Pick the lowest-cost index with an optional center-biased tie break."""
        x = inputs.to(torch.float32)
        device = x.device
        dim = sz * 2 + 1
        num_ch = dim * dim

        if cv_offset:
            rng = torch.arange(-sz, sz + 1, device=device, dtype=torch.float32)
            gy, gx = torch.meshgrid(rng, rng, indexing="ij")
            pattern = torch.abs(gy) + torch.abs(gx)
            offset_mask = pattern.reshape(num_ch) * offset_intensity
        else:
            offset_mask = torch.zeros((num_ch,), dtype=torch.float32, device=device)

        x = x + offset_mask
        min_val, _ = torch.min(x, dim=-1, keepdim=True)
        default_mask = (x == min_val).to(torch.float32)

        if mode == "default":
            mask = default_mask
        else:
            rng = torch.arange(-sz, sz + 1, device=device, dtype=torch.float32)
            gy, gx = torch.meshgrid(rng, rng, indexing="ij")

            if mode == "square":
                pattern = torch.maximum(torch.abs(gy), torch.abs(gx))
            elif mode == "diamond":
                pattern = torch.abs(gy) + torch.abs(gx)
            elif mode == "radial":
                pattern = gy.pow(2) + gx.pow(2)
            elif mode == "spiral":
                pattern = torch.tensor(
                    spiral_pattern(sz, sz), dtype=torch.float32, device=device
                )
            else:
                raise ValueError("mode for ArgMinCentered not recognized")

            epsilon = torch.tensor(_SUBPIXEL_EPS, dtype=torch.float32, device=device)
            central_bias = pattern.reshape(num_ch) * epsilon
            central_mask = default_mask - central_bias
            central_mask_max, _ = torch.max(central_mask, dim=-1, keepdim=True)
            mask = torch.where(
                central_mask == central_mask_max,
                torch.tensor(1.0, dtype=torch.float32, device=device),
                torch.tensor(0.0, dtype=torch.float32, device=device),
            )

        epsilon = torch.tensor(_SUBPIXEL_EPS, dtype=torch.float32, device=device)
        bias = (
            torch.linspace(
                0, num_ch - 1, steps=num_ch, dtype=torch.float32, device=device
            )
            * epsilon
        )
        final_mask = mask - bias
        final_mask_max, _ = torch.max(final_mask, dim=-1, keepdim=True)
        one_hot = torch.where(
            final_mask == final_mask_max,
            torch.tensor(1.0, dtype=torch.float32, device=device),
            torch.tensor(0.0, dtype=torch.float32, device=device),
        )
        ch_range = torch.arange(num_ch, dtype=torch.float32, device=device)
        return (one_hot * ch_range).sum(dim=-1).to(torch.int32)


def spiral_pattern(sz_y: int, sz_x: int) -> np.ndarray:
    """Create the spiral traversal pattern."""
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


class ExtractSearchWindows(nn.Module):
    """Extract blockmatch search windows."""

    def __init__(
        self,
        template_sz: int,
        max_sr: int = _BLOCKMATCH_SEARCH_RANGE,
        memory_efficient: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.template_sz = template_sz
        self.max_sr = max_sr
        self.max_sz = 2 * max_sr + 1
        self.memory_efficient = memory_efficient
        if self.memory_efficient:
            logger.info("Using memory-efficient search window extraction")

    def _forward_memory_efficient(
        self, inputs: torch.Tensor, search_range: int
    ) -> torch.Tensor:
        pad = self.max_sz // 2
        inputs = F.pad(inputs, (pad, pad, pad, pad), mode="constant", value=0)

        kernel = self.template_sz
        patches1 = F.unfold(
            inputs.to(torch.float32),
            kernel_size=(kernel, kernel),
            stride=1,
            padding=kernel // 2,
        )
        height_padded = inputs.shape[2]
        width_padded = inputs.shape[3]
        patches1 = patches1.view(
            inputs.shape[0], kernel * kernel, height_padded, width_padded
        )

        height = height_padded - 2 * pad
        width = width_padded - 2 * pad
        cost_volume_sz = 2 * search_range + 1
        offset = self.max_sr - search_range

        patches1_hwc = patches1.permute(0, 2, 3, 1).contiguous()
        del patches1

        batch = patches1_hwc.shape[0]
        kernel_sq = patches1_hwc.shape[3]
        stride_b, stride_h, stride_w, stride_k = patches1_hwc.stride()
        windows_5d = torch.as_strided(
            patches1_hwc,
            size=(batch, height, width, cost_volume_sz, cost_volume_sz, kernel_sq),
            stride=(stride_b, stride_h, stride_w, stride_h, stride_w, stride_k),
            storage_offset=offset * stride_h + offset * stride_w,
        )

        windows = windows_5d.to(torch.uint8).reshape(
            batch, height, width, cost_volume_sz * cost_volume_sz, kernel_sq
        )
        del patches1_hwc
        return windows

    def _forward_default(self, inputs: torch.Tensor, search_range: int) -> torch.Tensor:
        pad = self.max_sz // 2
        inputs = F.pad(inputs, (pad, pad, pad, pad), mode="constant", value=0)

        kernel = self.template_sz
        patches1 = F.unfold(
            inputs.to(torch.float32),
            kernel_size=(kernel, kernel),
            stride=1,
            padding=kernel // 2,
        )
        height_padded = inputs.shape[2]
        width_padded = inputs.shape[3]
        patches1 = patches1.view(
            inputs.shape[0], kernel * kernel, height_padded, width_padded
        )

        search_sz = self.max_sz
        patches2 = F.unfold(
            patches1, kernel_size=(search_sz, search_sz), stride=1, padding=0
        )
        height = height_padded - 2 * pad
        width = width_padded - 2 * pad
        patches2 = patches2.view(
            inputs.shape[0], kernel * kernel, search_sz, search_sz, height, width
        )
        windows_max = patches2.permute(0, 4, 5, 2, 3, 1).contiguous()

        cost_volume_sz = 2 * search_range + 1
        offset = self.max_sr - search_range
        r = torch.arange(offset, offset + cost_volume_sz, device=inputs.device)
        windows = windows_max[:, :, :, r][:, :, :, :, r]
        windows = windows.reshape(
            inputs.shape[0],
            height,
            width,
            cost_volume_sz * cost_volume_sz,
            kernel * kernel,
        )
        return windows.to(torch.uint8)

    def forward(self, inputs: torch.Tensor, search_range: int) -> torch.Tensor:
        """Dispatch to the configured dense or memory-efficient extractor."""
        if self.memory_efficient:
            return self._forward_memory_efficient(inputs, search_range)
        return self._forward_default(inputs, search_range)


class ExtractTemplates(nn.Module):
    """Extract centered blockmatch templates."""

    def __init__(self, template_sz: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.template_sz = template_sz

    def forward(self, inputs: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Extract centered templates and cast them to the requested dtype."""
        shape = inputs.shape
        inputs = cast(inputs, dtype=dtype)
        kernel = self.template_sz
        patches = F.unfold(
            inputs.to(torch.float32),
            kernel_size=(kernel, kernel),
            stride=1,
            padding=kernel // 2,
        )
        patches = patches.view(shape[0], -1, shape[2], shape[3])
        patches = patches[:, : kernel * kernel, :, :]
        patches = patches.permute(0, 2, 3, 1).unsqueeze(-2)
        return patches.to(dtype)


class CalculateVector(nn.Module):
    """Compute best matching integer motion vectors from the cost volume."""

    _COST_CHUNK = _COST_CHUNK_SIZE

    def __init__(self, search_range: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_range = search_range
        self.argmin_centered = ArgMinCentered()

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert window and template tensors into vectors and winning templates."""
        windows, templates = inputs
        batch, height, width, channels, kernel_sq = windows.shape

        rng = torch.arange(
            -self.search_range, self.search_range + 1, device=windows.device
        )
        jj, ii = torch.meshgrid(rng, rng, indexing="ij")
        vec_lut = -1.0 * torch.stack([jj, ii], dim=-1).reshape(-1, 2).to(torch.float16)

        templates_i16 = templates.to(torch.int16)
        cost_chunks: list[torch.Tensor] = []
        for start in range(0, channels, self._COST_CHUNK):
            end = min(start + self._COST_CHUNK, channels)
            chunk_i16 = windows[..., start:end, :].to(torch.int16)
            diff_chunk = torch.abs(chunk_i16 - templates_i16)
            del chunk_i16
            cost_chunks.append(torch.sum(diff_chunk, dim=-1, dtype=torch.int32))
            del diff_chunk
        del templates_i16
        cost_volume_i32 = torch.cat(cost_chunks, dim=-1)

        input_mv_idx = (2 * self.search_range + 1) ** 2
        min_idx_block_match = (
            self.argmin_centered(
                cost_volume_i32[..., :input_mv_idx],
                sz=self.search_range,
                mode="spiral",
                cv_offset=False,
            )
            .to(torch.long)
            .unsqueeze(-1)
        )

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

        vector = vec_lut[min_idx_block_match.squeeze(-1)]
        idx = min_idx.unsqueeze(-1).expand(batch, height, width, 1, kernel_sq)
        min_templates = torch.gather(windows, dim=3, index=idx)
        return vector, min_templates, input_mv_mask, min_cost_volume


class JointBilateralFilter(nn.Module):
    """Edge-aware vector smoothing."""

    def __init__(
        self,
        kernel_sz: int,
        sigma_spatial: float,
        sigma_intensity: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        sigma_spatial_div = 1.0 / (2.0 * (sigma_spatial**2.0))
        self.kernel_sz = kernel_sz
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
        batch, channels, height, width = x.shape
        x = cast(x, dtype=torch.float32)
        patches = F.unfold(x, kernel_size=(sz, sz), stride=1, padding=sz // 2)
        patches = patches.view(batch, channels, sz * sz, height, width).permute(
            0, 3, 4, 2, 1
        )
        return patches.to(dtype)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Smooth the candidate motion field while preserving image edges."""
        template, vector_curr = inputs
        template = cast(template, dtype=torch.float32)
        vector_curr = cast(vector_curr, dtype=torch.float32)
        kernel = self.kernel.to(template.device)
        sigma_intensity_div = self.sigma_intensity_div.to(template.device)

        compare_pix = template.permute(0, 2, 3, 1).unsqueeze(-2)
        compare_kernel = self._extract_block(template, self.kernel_sz)
        intensity_diff = torch.sum((compare_pix - compare_kernel) ** 2, dim=-1)

        one = torch.tensor(1.0, dtype=torch.float32, device=template.device)
        zero = torch.tensor(0.0, dtype=torch.float32, device=template.device)
        coeff = torch.clamp(
            one - torch.abs(kernel - intensity_diff * sigma_intensity_div),
            zero,
            one,
        ).unsqueeze(-1)

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
    """Small-window median filter used by the reference blockmatch pass."""
    kh, kw = ksize
    pad_mode = "replicate" if padding == "SYMMETRIC" else "constant"
    x_pad = F.pad(x, (kw // 2, kw // 2, kh // 2, kh // 2), mode=pad_mode, value=0)
    patches = F.unfold(x_pad, kernel_size=(kh, kw), stride=1, padding=0)
    patches = patches.view(x.shape[0], x.shape[1], kh * kw, -1)
    if (kh * kw) % 2 == 1:
        k = (kh * kw + 1) // 2
        median = torch.kthvalue(patches, k, dim=2).values
    else:
        k1 = (kh * kw) // 2
        k2 = k1 + 1
        v1 = torch.kthvalue(patches, k1, dim=2).values
        v2 = torch.kthvalue(patches, k2, dim=2).values
        median = (v1 + v2) / 2.0
    out = median.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    return out.to(x.dtype)


class CalculateFlow(nn.Module):
    """Multi-level blockmatching flow estimation."""

    def __init__(
        self,
        levels: int = _BLOCKMATCH_LEVELS,
        template_sz: int = _BLOCKMATCH_TEMPLATE_SIZE,
        search_range: int = _BLOCKMATCH_SEARCH_RANGE,
        median_kernel: tuple[int, int] = _BLOCKMATCH_MEDIAN_KERNEL,
        blur_levels: Optional[list[int]] = None,
        last_bm_level: int = _BLOCKMATCH_LAST_BM_LEVEL,
        min_cv_output: bool = False,
        oob_replacement: bool = False,
        mv_hints: bool = True,
        memory_efficient: bool = False,
    ):
        super().__init__()
        self.levels = levels
        self.search_range = int(search_range)
        self.template_sz = template_sz
        self.median_kernel = median_kernel
        self.blur_levels = (
            blur_levels if blur_levels is not None else list(_BLOCKMATCH_BLUR_LEVELS)
        )
        self.last_bm_level = last_bm_level
        self.min_cv_output = min_cv_output
        self.oob_replacement = oob_replacement
        self.mv_hints = mv_hints

        self._down = DownSampling2D(
            size=_DEFAULT_SCALE_FACTOR, interpolation=_BILINEAR_INTERPOLATION
        )
        self._warp_bilinear = DenseWarp(interpolation=_BILINEAR_INTERPOLATION)
        self._warp_nearest_oob_zero = DenseWarp(interpolation="nearest_oob_zero")

        self._search_windows = ExtractSearchWindows(
            template_sz=self.template_sz,
            max_sr=self.search_range,
            memory_efficient=memory_efficient,
        )
        self._calc_vector = CalculateVector(search_range=self.search_range)
        self._joint_bilateral = JointBilateralFilter(
            kernel_sz=_JOINT_BILATERAL_KERNEL_SIZE,
            sigma_spatial=_JOINT_BILATERAL_SIGMA_SPATIAL,
            sigma_intensity=_JOINT_BILATERAL_SIGMA_INTENSITY,
        )

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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Estimate flow over the multi-scale blockmatch pyramid."""
        additional_outputs: dict[str, torch.Tensor] = {}

        input_mv = input_mv.to(torch.float16)
        (search_frame, template_frame), initial_shape = self.pad_to_even(
            [search_frame, template_frame]
        )

        search_frame = cast(search_frame, dtype=torch.uint8)
        template_frame = cast(template_frame, dtype=torch.uint8)

        search_pyramid = [search_frame]
        template_pyramid = [template_frame]
        shape_pyramid = [initial_shape]

        for level in range(1, self.levels):
            prev_search_img = cast(search_pyramid[level - 1], dtype=torch.float32)
            prev_template_img = cast(template_pyramid[level - 1], dtype=torch.float32)

            if level - 1 in self.blur_levels:
                prev_search_img = binomial_filter(prev_search_img)
                prev_template_img = binomial_filter(prev_template_img)

            search_down = cast(self._down(prev_search_img), dtype=torch.uint8)
            template_down = cast(self._down(prev_template_img), dtype=torch.uint8)
            (search_down, template_down), shape_down = self.pad_to_even(
                [search_down, template_down]
            )
            search_pyramid.append(search_down)
            template_pyramid.append(template_down)
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

        vectors: list[torch.Tensor] = []
        for idx, (search_img, template_img) in enumerate(
            zip(search_pyramid, template_pyramid, strict=False)
        ):
            if idx != 0:
                vector_prev = vectors[idx - 1].to(torch.float32)
                vector_prev = F.interpolate(
                    vector_prev,
                    scale_factor=float(_DEFAULT_SCALE_FACTOR),
                    mode=_BILINEAR_INTERPOLATION,
                    align_corners=False,
                ) * float(_DEFAULT_SCALE_FACTOR)

                search_img = cast(search_img, torch.float32)
                search_img = self._warp_bilinear([search_img, vector_prev])
                search_img = cast(search_img, torch.uint8)
                vector_prev = vector_prev.to(torch.float16)
            else:
                vector_prev = None

            windows = self._search_windows(search_img, search_range=self.search_range)

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
            ) = self._calc_vector((windows, templates))
            subpixel_est = calculate_subpixel(
                templates, min_templates, template_img, self.template_sz
            )
            vector_curr = vector_curr.permute(0, 3, 1, 2).contiguous() + subpixel_est

            if vector_prev is not None:
                vector_curr = vector_curr + vector_prev

            vector_curr = median_filter2d(
                vector_curr, ksize=self.median_kernel, padding="SYMMETRIC"
            )
            vector_curr = self._joint_bilateral((template_img, vector_curr))

            if idx == target_idx:
                mask = input_mv_mask.permute(0, 3, 1, 2).expand_as(vector_curr)
                vector_curr = torch.where(mask, input_mv, vector_curr)

            if self.oob_replacement and self.mv_hints and idx == target_idx:
                blank_img = cast(
                    torch.ones_like(template_img)[:, :1, :, :], torch.float32
                )
                oob_mask = self._warp_nearest_oob_zero([blank_img, input_mv])
                vector_curr = torch.where(oob_mask == 0, input_mv, vector_curr)
                max_cost = torch.tensor(
                    _UINT8_MAX_INT * self.template_sz**2 + 1,
                    dtype=torch.int32,
                    device=vector_curr.device,
                )
                min_cost_volume = torch.where(
                    oob_mask.permute(0, 2, 3, 1) == 0, max_cost, min_cost_volume
                )

            shape = shape_pyramid[idx]
            height = shape["height"]
            width = shape["width"]
            vector_curr = vector_curr[:, :, :height, :width]
            min_cost_volume = min_cost_volume[:, :height, :width, :]
            if idx == target_idx:
                input_mv = input_mv[:, :, :height, :width]

            vectors.append(vector_curr)

            if idx == target_idx:
                if self.min_cv_output:
                    additional_outputs["min_cost_volume"] = min_cost_volume
                break

        return vector_curr, additional_outputs


@dataclass(frozen=True)
class BlockMatchV311Config:
    """Reference blockmatch-v311 parameters used by NFRU v1."""

    rgb_in: bool = True
    levels: int = _BLOCKMATCH_LEVELS
    template_sz: int = _BLOCKMATCH_TEMPLATE_SIZE
    search_range: int = _BLOCKMATCH_SEARCH_RANGE
    median_kernel: tuple[int, int] = _BLOCKMATCH_MEDIAN_KERNEL
    blur_levels: tuple[int, int, int, int] = _BLOCKMATCH_BLUR_LEVELS
    last_bm_level: int = _BLOCKMATCH_LAST_BM_LEVEL
    mv_hints: bool = True
    min_cv_output: bool = False
    oob_replacement: bool = False
    output_polarity: str = "positive"
    mv_hints_polarity: str = "positive"
    template_frame_id: str = "tm1"


class BlockMatchV311(nn.Module):
    """Private NFRU-local blockmatch-v311 runtime helper."""

    def __init__(
        self,
        config: BlockMatchV311Config | None = None,
        memory_efficient: bool = True,
    ):
        super().__init__()
        self.config = config or BlockMatchV311Config()
        self.memory_efficient = memory_efficient
        self.calc_flow = CalculateFlow(
            levels=self.config.levels,
            template_sz=self.config.template_sz,
            search_range=self.config.search_range,
            median_kernel=self.config.median_kernel,
            blur_levels=list(self.config.blur_levels),
            last_bm_level=self.config.last_bm_level,
            min_cv_output=self.config.min_cv_output,
            oob_replacement=self.config.oob_replacement,
            mv_hints=self.config.mv_hints,
            memory_efficient=self.memory_efficient,
        )

    def forward(
        self, img_t: torch.Tensor, img_tm1: torch.Tensor, input_mv: torch.Tensor
    ) -> torch.Tensor:
        """Run the configured blockmatch pipeline and return the signed flow."""
        self.calc_flow.to(img_t.device)
        # Keep the search-window policy stable across train/eval. Validation runs
        # full-resolution frames and can OOM if eval falls back to the dense path.
        self.calc_flow._search_windows.memory_efficient = self.memory_efficient

        if self.config.rgb_in:
            img_t = _rgb_to_y(img_t)
            img_tm1 = _rgb_to_y(img_tm1)

        if self.config.template_frame_id == "tm1":
            search_frame = img_t
            template_frame = img_tm1
        elif self.config.template_frame_id == "t":
            search_frame = img_tm1
            template_frame = img_t
        else:
            raise ValueError(
                f"template_frame_id not recognised: {self.config.template_frame_id}"
            )

        if self.config.output_polarity == "positive":
            reversed_polarity = -1.0
        elif self.config.output_polarity == "negative":
            reversed_polarity = 1.0
        else:
            raise ValueError(
                f"output_polarity not recognised: {self.config.output_polarity}"
            )

        if self.config.mv_hints_polarity == "positive":
            warp_polarity = -1.0
        elif self.config.mv_hints_polarity == "negative":
            warp_polarity = 1.0
        else:
            raise ValueError(
                "mv_hints_polarity not set to 'positive' or 'negative': "
                f"{self.config.mv_hints_polarity}"
            )

        flow_out, _ = self.calc_flow(
            search_frame=search_frame,
            template_frame=template_frame,
            input_mv=input_mv * warp_polarity,
        )
        return flow_out * reversed_polarity
