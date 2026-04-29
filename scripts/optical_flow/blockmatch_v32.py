# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

"""The BlockMatchV3.2 optical flow algorithm. See class BlockMatchV32 for entry point."""

from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn

from ng_model_gym.core.model.layers.dense_warp import DenseWarp
from ng_model_gym.usecases.nfru.utils.down_sampling_2d import DownSampling2D
from scripts.safetensors_generator.dataset_readers.argmin_centered import ArgMinCentered


def _cast(img: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    in_type = img.dtype
    if in_type in (torch.float16, torch.float32) and dtype == torch.uint8:
        img = torch.clamp(torch.round(img * 255.0), 0.0, 255.0)
    img = img.to(dtype)
    if in_type == torch.uint8 and dtype in (torch.float16, torch.float32):
        img = img / 255.0
    return img


def _gradient(f: torch.Tensor):
    f_pad = F.pad(f, (1, 1, 1, 1), mode="replicate")
    FY = (f_pad[:, :, 2:, 1:-1] - f_pad[:, :, :-2, 1:-1]) / 2.0
    FX = (f_pad[:, :, 1:-1, 2:] - f_pad[:, :, 1:-1, :-2]) / 2.0
    return FY, FX


def _rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return 0.25 * (r + 2.0 * g + b)


def _binomial_filter(image: torch.Tensor) -> torch.Tensor:
    """Separable binomial [1 2 1]/4 with SYMMETRIC padding"""
    vert_padded = F.pad(image, (0, 0, 1, 1), mode="replicate")
    v_blur = (vert_padded[:, :, :-2, :] + 2.0 * image + vert_padded[:, :, 2:, :]) / 4.0

    horiz_padded = F.pad(v_blur, (1, 1, 0, 0), mode="replicate")
    h_blur = (
        horiz_padded[:, :, :, :-2] + 2.0 * v_blur + horiz_padded[:, :, :, 2:]
    ) / 4.0
    return h_blur


def _calculate_subpixel(
    f: torch.Tensor, g: torch.Tensor, f_img: torch.Tensor, template_sz: int
) -> torch.Tensor:
    f = _cast(f, dtype=torch.float16)
    g = _cast(g, dtype=torch.float16)
    f_img = _cast(f_img, dtype=torch.float16)

    dfy_img, dfx_img = _gradient(f_img)
    dfy = ExtractTemplates(template_sz=template_sz)(dfy_img, dtype=torch.float16)
    dfx = ExtractTemplates(template_sz=template_sz)(dfx_img, dtype=torch.float16)

    pred = torch.zeros(
        (template_sz - 2, template_sz - 2), dtype=torch.float16, device=f.device
    )
    pred = F.pad(pred, (1, 1, 1, 1), mode="constant", value=1)
    pred = pred.reshape(-1)

    def redsum(x):
        return torch.sum(x, dim=-1, keepdim=True).to(torch.float32)

    a = redsum((dfx**2) * pred)
    b = redsum(dfx * dfy * pred)
    d = redsum((dfy**2) * pred)

    z = g - f
    p = redsum(z * dfx * pred)
    q = redsum(z * dfy * pred)

    det_A = a * d - b * b

    subpixel_u = d * p - b * q
    subpixel_v = a * q - b * p
    subpixel = torch.cat([subpixel_v, subpixel_u], dim=-1)
    subpixel = subpixel / (
        det_A + torch.tensor(0.0, dtype=torch.float32, device=subpixel.device)
    )

    invalid_mask = (det_A <= 1e-7) | (torch.abs(subpixel) >= 1.0)
    subpixel = torch.where(
        invalid_mask,
        torch.tensor(0.0, dtype=torch.float32, device=subpixel.device),
        subpixel,
    )
    subpixel = subpixel.squeeze(-2)
    return subpixel.to(torch.float16).permute(0, 3, 1, 2).contiguous()


def _median_filter2d(
    x: torch.Tensor, ksize=(3, 3), padding: str = "SYMMETRIC"
) -> torch.Tensor:
    kh, kw = ksize
    pad_mode = "replicate" if padding == "SYMMETRIC" else "constant"
    x_pad = F.pad(x, (kw // 2, kw // 2, kh // 2, kh // 2), mode=pad_mode, value=0)
    patches = F.unfold(x_pad, kernel_size=(kh, kw), stride=1, padding=0)
    B, _, HW = patches.shape
    C = x.shape[1]
    patches = patches.view(B, C, kh * kw, HW)
    if (kh * kw) % 2 == 1:
        k = (kh * kw + 1) // 2
        median = torch.kthvalue(patches, k, dim=2).values
    else:
        k1 = (kh * kw) // 2
        k2 = k1 + 1
        v1 = torch.kthvalue(patches, k1, dim=2).values
        v2 = torch.kthvalue(patches, k2, dim=2).values
        median = (v1 + v2) / 2.0
    out = median.view(B, C, x.shape[2], x.shape[3])
    return out.to(x.dtype)


class ExtractSearchWindows(nn.Module):
    """Build local search windows from input patches for block matching."""

    def __init__(self, template_sz: int, max_sr: int = 3, **kw):
        super().__init__(**kw)
        self.template_sz = template_sz
        self.max_sr = max_sr
        self.max_sz = 2 * max_sr + 1

    def forward(self, inputs: torch.Tensor, search_range: int) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        pad = self.max_sz // 2
        inputs = F.pad(inputs, (pad, pad, pad, pad), mode="constant", value=0)

        # First patch extraction.
        k = self.template_sz
        patches1 = F.unfold(
            inputs.to(torch.float32), kernel_size=(k, k), stride=1, padding=k // 2
        )
        B, _, _ = patches1.shape
        H_p = inputs.shape[2]
        W_p = inputs.shape[3]
        patches1 = patches1.view(B, k * k, H_p, W_p)

        # Second patch extraction over patch grid.
        s = self.max_sz
        patches2 = F.unfold(patches1, kernel_size=(s, s), stride=1, padding=0)
        H0 = H_p - 2 * pad
        W0 = W_p - 2 * pad
        patches2 = patches2.view(B, k * k, s, s, H0, W0)
        windows_max = patches2.permute(0, 4, 5, 2, 3, 1).contiguous()

        cost_volume_sz = 2 * search_range + 1
        offset = self.max_sr - search_range
        r = torch.arange(offset, offset + cost_volume_sz, device=inputs.device)
        windows = windows_max[:, :, :, r][:, :, :, :, r]
        B2, H2, W2, _, _, _ = windows.shape
        windows = windows.reshape(B2, H2, W2, cost_volume_sz * cost_volume_sz, k * k)
        return windows.to(torch.uint8)


class ExtractTemplates(nn.Module):
    """Extract per-pixel template patches used in matching and refinement."""

    def __init__(self, template_sz: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.template_sz = template_sz

    def forward(self, inputs: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        sh = inputs.shape
        inputs = _cast(inputs, dtype=dtype)
        k = self.template_sz
        patches = F.unfold(
            inputs.to(torch.float32), kernel_size=(k, k), stride=1, padding=k // 2
        )
        B, _, _ = patches.shape
        H, W = sh[2], sh[3]
        patches = patches.view(B, -1, H, W)
        patches = patches[:, : k * k, :, :]
        patches = patches.permute(0, 2, 3, 1).unsqueeze(-2)
        return patches.to(dtype)


class CalculateVector(nn.Module):
    """Select best-matching displacement vectors from the local cost volume."""

    def __init__(self, search_range: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_range = search_range
        self.argmin_centered = ArgMinCentered()

    def forward(self, inputs):
        # pylint: disable=missing-function-docstring
        w1, w2 = inputs
        # w1: [B,H,W,C,K2], w2: [B,H,W,1,K2]
        B, H, W, _, K2 = w1.shape

        w1_i16 = w1.to(torch.int16)
        w2_i16 = w2.to(torch.int16)

        rng = torch.arange(-self.search_range, self.search_range + 1, device=w1.device)
        jj, ii = torch.meshgrid(rng, rng, indexing="ij")
        self.vecLUT = -1.0 * torch.stack([jj, ii], dim=-1).reshape(-1, 2).to(
            torch.float16
        )

        # SAD cost volume over template window
        diff = torch.abs(w1_i16 - w2_i16)
        cost_volume_i32 = torch.sum(diff, dim=-1, dtype=torch.int32)  # [B,H,W,num_ch]

        input_mv_idx = (2 * self.search_range + 1) ** 2

        # ArgMinCentered over cost_volume
        min_idx_block_match = self.argmin_centered(
            cost_volume_i32[..., :input_mv_idx], sz=self.search_range, mode="spiral"
        )
        min_idx_block_match = min_idx_block_match.to(torch.long).unsqueeze(-1)

        # Gather min cost from block match
        idx_bm = min_idx_block_match
        idx_bm_exp = idx_bm.expand(B, H, W, 1)
        min_cost_block_match = torch.gather(cost_volume_i32, dim=3, index=idx_bm_exp)

        last_dim = cost_volume_i32.shape[-1]
        has_input_mv = (last_dim - 1) == input_mv_idx

        if has_input_mv:
            cost_at_input_mv = cost_volume_i32[..., -1:].contiguous()
            input_mv_mask = cost_at_input_mv < min_cost_block_match
            input_mv_idx_tensor = torch.full_like(
                idx_bm, fill_value=input_mv_idx, dtype=torch.long
            )
            min_idx = torch.where(input_mv_mask, input_mv_idx_tensor, idx_bm)
            min_cost_volume = torch.minimum(cost_at_input_mv, min_cost_block_match)
        else:
            input_mv_mask = torch.zeros_like(min_cost_block_match, dtype=torch.bool)
            min_idx = idx_bm
            min_cost_volume = min_cost_block_match

        # Gather vector from LUT using block_match index
        vector = self.vecLUT[min_idx_block_match.squeeze(-1)]

        # Gather min templates using min_idx
        idx = min_idx.unsqueeze(-1).expand(B, H, W, 1, K2)
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
        self.kernel = -sigma_spatial_div * torch.ones(
            (1, kernel_sz * kernel_sz), dtype=torch.float32
        )
        self.sigma_intensity_div = torch.tensor(
            1.0 / (2.0 * (sigma_intensity**2.0)), dtype=torch.float32
        )

    @staticmethod
    def _extract_block(
        x: torch.Tensor, sz: int, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        N, C, H, W = x.shape
        x = _cast(x, dtype=torch.float32)
        patches = F.unfold(x, kernel_size=(sz, sz), stride=1, padding=sz // 2)
        patches = patches.view(N, C, sz * sz, H, W).permute(0, 3, 4, 2, 1)
        return patches.to(dtype)

    def forward(self, inputs):
        # pylint: disable=missing-function-docstring
        t, vector_curr = inputs
        t = _cast(t, dtype=torch.float32)
        vector_curr = _cast(vector_curr, dtype=torch.float32)

        compare_pix = t.permute(0, 2, 3, 1).unsqueeze(-2)
        compare_kernel = self._extract_block(t, self.kernel_sz)
        intensity_diff = torch.sum((compare_pix - compare_kernel) ** 2, dim=-1)
        one = torch.tensor(1.0, dtype=torch.float32, device=t.device)
        zero = torch.tensor(0.0, dtype=torch.float32, device=t.device)
        coeff = torch.clamp(
            one - torch.abs(self.kernel - intensity_diff * self.sigma_intensity_div),
            zero,
            one,
        )
        coeff = coeff.unsqueeze(-1)

        vector_block = self._extract_block(vector_curr, self.kernel_sz)
        num = torch.sum(vector_block * coeff, dim=-2)
        den = torch.sum(coeff, dim=-2)
        out = num / den
        return out.to(torch.float16).permute(0, 3, 1, 2).contiguous()


class CalculateFlow(nn.Module):
    """Compute multi-scale optical flow with block matching and MV hints."""

    def __init__(
        self,
        *args,
        levels=6,
        template_sz=5,
        search_range=3,
        median_kernel=(3, 3),
        blur_levels=(1, 2, 3, 4),
        last_bm_level=2,
        performance_mode="medium",
        min_cv_output=False,
        mv_hints=True,
        din_sr=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.levels = levels
        self.search_range = search_range
        self.template_sz = template_sz
        self.median_kernel = median_kernel
        self.blur_levels = blur_levels
        self.last_bm_level = last_bm_level
        self.performance_mode = performance_mode
        self.min_cv_output = min_cv_output
        self.mv_hints = mv_hints
        self.din_sr = din_sr

        self._down = DownSampling2D(size=2, interpolation="bilinear")
        self._warp_bilinear = DenseWarp(interpolation="bilinear")

    @staticmethod
    def pad_to_even(tensors: list):
        # pylint: disable=missing-function-docstring
        H, W = tensors[0].shape[2], tensors[0].shape[3]
        h_pad = H % 2
        w_pad = W % 2
        padded = [F.pad(t, (0, w_pad, 0, h_pad), mode="replicate") for t in tensors]
        initial_shape = {"height": H, "width": W}
        return padded, initial_shape

    def forward(
        self,
        search_frame: torch.Tensor,
        template_frame: torch.Tensor,
        input_mv: torch.Tensor,
    ):
        # pylint: disable=missing-function-docstring, too-many-branches
        additional_outputs = {}

        custom_search_range = self.search_range
        if self.din_sr:
            avg_mv = torch.mean(torch.abs(input_mv))
            scale = 2 ** (self.levels - 1 - self.last_bm_level)
            custom_search_range = int(
                torch.clamp(
                    torch.floor(2.0 * avg_mv / scale) + 1, 1, self.search_range
                ).item()
            )

        input_mv = input_mv.to(torch.float16)
        (search_frame, template_frame), initial_shape = self.pad_to_even(
            [search_frame, template_frame]
        )

        search_frame = _cast(search_frame, dtype=torch.uint8)
        template_frame = _cast(template_frame, dtype=torch.uint8)

        search_pyramid = [search_frame]
        template_pyramid = [template_frame]
        shape_pyramid = [initial_shape]
        for i in range(1, self.levels):
            prev_search_img = search_pyramid[i - 1]
            prev_template_img = template_pyramid[i - 1]

            prev_search_img = _cast(prev_search_img, dtype=torch.float32)
            prev_template_img = _cast(prev_template_img, dtype=torch.float32)

            if i - 1 in self.blur_levels:
                prev_search_img = _binomial_filter(prev_search_img)
                prev_template_img = _binomial_filter(prev_template_img)

            search_img_down = self._down(prev_search_img)
            template_img_down = self._down(prev_template_img)
            search_img_down = _cast(search_img_down, dtype=torch.uint8)
            template_img_down = _cast(template_img_down, dtype=torch.uint8)
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
            zip(search_pyramid, template_pyramid)
        ):
            if idx != 0:
                vector_prev = vectors[idx - 1]
                vector_prev = _cast(vector_prev, torch.float32)
                vector_prev = (
                    F.interpolate(
                        vector_prev,
                        scale_factor=2.0,
                        mode="bilinear",
                        align_corners=False,
                    )
                    * 2.0
                )

                search_img_f = _cast(search_img, torch.float32)
                search_img_warped = self._warp_bilinear([search_img_f, vector_prev])
                search_img_warped = _cast(search_img_warped, torch.uint8)
                vector_prev = _cast(vector_prev, torch.float16)
            else:
                search_img_warped = search_img

            windows = ExtractSearchWindows(
                template_sz=self.template_sz, max_sr=self.search_range
            )(search_img_warped, search_range=custom_search_range)

            if self.mv_hints and idx == target_idx:
                mv_search_img = _cast(search_img, torch.float32)
                mv_warped = self._warp_bilinear(
                    [mv_search_img, input_mv.to(torch.float32)]
                )
                mv_warped = _cast(mv_warped, torch.uint8)
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

            # Subpixel estimation
            subpixel_est = _calculate_subpixel(
                templates, min_templates, template_img, self.template_sz
            )
            vector_curr = vector_curr.permute(0, 3, 1, 2).contiguous()
            vector_curr = vector_curr + subpixel_est

            if vector_prev is not None:
                vector_curr = vector_curr + vector_prev

            vector_curr = _median_filter2d(
                vector_curr, ksize=self.median_kernel, padding="SYMMETRIC"
            )

            if self.performance_mode != "fast":
                vector_curr = JointBilateralFilter(
                    kernel_sz=5, sigma_spatial=7.4, sigma_intensity=0.14
                )([template_img, vector_curr])

            # Replace predicted flow with hint MVs
            if idx == target_idx:
                mask = input_mv_mask.permute(0, 3, 1, 2).expand_as(vector_curr)
                vector_curr = torch.where(mask, input_mv, vector_curr)

            shape = shape_pyramid[idx]
            Ht, Wt = shape["height"], shape["width"]
            vector_curr = vector_curr[:, :, :Ht, :Wt]
            min_cost_volume = min_cost_volume[:, :Ht, :Wt, :]
            if idx == target_idx:
                input_mv = input_mv[:, :, :Ht, :Wt]

            vectors.append(vector_curr)

            if idx == target_idx:
                if self.min_cv_output:
                    additional_outputs["min_cost_volume"] = min_cost_volume
                break

        return vector_curr, additional_outputs


class Polarity(Enum):
    """The flow polarity choices."""

    POSITIVE = 1
    NEGATIVE = 2


class TemplateFrameId(Enum):
    """The frame ID to use for templating."""

    T = 1
    TM1 = 2


class BlockMatchV32(nn.Module):
    """Model implementing the BlockMatchV3.2 optical flow algorithm"""

    def __init__(
        self,
        rgb_in=True,
        output_polarity=Polarity.POSITIVE,
        mv_hints_polarity=Polarity.POSITIVE,
        template_frame_id=TemplateFrameId.TM1,
        min_cv_output=False,
        flow_params=None,
    ):
        super().__init__()
        self.rgb_in = rgb_in
        self.output_polarity = output_polarity
        self.mv_hints_polarity = mv_hints_polarity
        self.template_frame_id = template_frame_id
        self.min_cv_output = min_cv_output
        self.calc_flow = CalculateFlow(**flow_params, min_cv_output=min_cv_output)

    def forward(self, inputs: dict):
        # pylint: disable=missing-function-docstring

        # NCHW
        img_t = inputs["img_t"]
        img_tm1 = inputs["img_tm1"]
        input_mv = inputs["input_mv"]

        # Convert to grayscale if needed.
        if self.rgb_in:
            img_t = _rgb_to_y(img_t)
            img_tm1 = _rgb_to_y(img_tm1)

        # Decide frame direction.
        if self.template_frame_id == TemplateFrameId.TM1:
            search_frame = img_t
            template_frame = img_tm1
        elif self.template_frame_id == TemplateFrameId.T:
            search_frame = img_tm1
            template_frame = img_t
        else:
            raise ValueError(
                f"template_frame_id not recognised: {self.template_frame_id}"
            )

        # Set output flow polarity.
        if self.output_polarity == Polarity.POSITIVE:
            reversed_polarity = -1.0
        elif self.output_polarity == Polarity.NEGATIVE:
            reversed_polarity = 1.0
        else:
            raise ValueError(
                f"output_polarity not set to 'positive' or 'negative': {self.output_polarity}"
            )

        # Set polarity for warps with mv hints.
        if self.mv_hints_polarity == Polarity.POSITIVE:
            warp_polarity = -1.0
        elif self.mv_hints_polarity == Polarity.NEGATIVE:
            warp_polarity = 1.0
        else:
            raise ValueError(
                f"mv_hints_polarity not set to 'positive' or 'negative': {self.mv_hints_polarity}"
            )

        flow_out, additional_outputs = self.calc_flow(
            search_frame=search_frame,
            template_frame=template_frame,
            input_mv=input_mv * warp_polarity,
        )
        flow_out = flow_out * reversed_polarity

        if self.min_cv_output:
            final_output = {"output": flow_out, **additional_outputs}
        else:
            final_output = {"output": flow_out}

        return final_output
