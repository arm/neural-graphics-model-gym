# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code

""" The BlockMatchV3 optical flow algorithm. See class BlockMatchV3. """

import torch
import torch.nn.functional as F
from typing_extensions import override

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
    subpixel = torch.cat([subpixel_v, subpixel_u], dim=-1)  # [B,H,W,1,2]
    subpixel = subpixel / (
        det_A + torch.tensor(0.0, dtype=torch.float32, device=subpixel.device)
    )

    invalid_mask = (det_A <= 1e-7) | (torch.abs(subpixel) >= 1.0)
    subpixel = torch.where(
        invalid_mask,
        torch.tensor(0.0, dtype=torch.float32, device=subpixel.device),
        subpixel,
    )
    subpixel = subpixel.squeeze(-2)  # [B,H,W,2]
    return subpixel.to(torch.float16).permute(0, 3, 1, 2).contiguous()  # [B,2,H,W]


class ExtractSearchWindows(torch.nn.Module):
    """Build local search windows from input patches for block matching."""

    def __init__(self, template_sz: int, max_sr: int = 3, **kw):
        super().__init__(**kw)
        self.template_sz = template_sz
        self.max_sr = max_sr
        self.max_sz = 2 * max_sr + 1

    @override
    def forward(self, inputs: torch.Tensor, search_range: int) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        pad = self.max_sz // 2
        inputs = F.pad(inputs, (pad, pad, pad, pad), mode="constant", value=0)

        # First patch extraction
        k = self.template_sz
        patches1 = F.unfold(
            inputs.to(torch.float32), kernel_size=(k, k), stride=1, padding=k // 2
        )
        H_p = inputs.shape[2]
        W_p = inputs.shape[3]
        patches1 = patches1.view(inputs.shape[0], k * k, H_p, W_p)  # C assumed 1

        # Second path extraction over patch grid
        s = self.max_sz
        patches2 = F.unfold(patches1, kernel_size=(s, s), stride=1, padding=0)
        H0 = H_p - 2 * pad
        W0 = W_p - 2 * pad
        patches2 = patches2.view(inputs.shape[0], k * k, s, s, H0, W0)
        windows_max = patches2.permute(
            0, 4, 5, 2, 3, 1
        ).contiguous()  # [B,H0,W0,s,s,k*k]

        cost_volume_sz = 2 * search_range + 1
        offset = self.max_sr - search_range
        r = torch.arange(offset, offset + cost_volume_sz, device=inputs.device)
        windows = windows_max[:, :, :, r][:, :, :, :, r]
        windows = windows.reshape(
            inputs.shape[0], H0, W0, cost_volume_sz * cost_volume_sz, k * k
        )
        return windows.to(torch.uint8)


class ExtractTemplates(torch.nn.Module):
    """Extract per-pixel template patches used in matching and refinement."""

    def __init__(self, template_sz: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.template_sz = template_sz

    @override
    def forward(self, inputs: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        sh = inputs.shape
        inputs = _cast(inputs, dtype=dtype)
        k = self.template_sz
        patches = F.unfold(
            inputs.to(torch.float32), kernel_size=(k, k), stride=1, padding=k // 2
        )
        patches = patches.view(sh[0], -1, sh[2], sh[3])  # [B, k*k*C, H, W]
        patches = patches[:, : k * k, :, :]
        patches = patches.permute(0, 2, 3, 1).unsqueeze(-2)  # [B,H,W,1,k*k]
        return patches.to(dtype)


class CalculateVector(torch.nn.Module):
    """Select best-matching displacement vectors from the local cost volume."""

    def __init__(self, search_range: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_range = search_range
        self.argmin_centered = ArgMinCentered()

    @override
    def forward(self, inputs):
        # pylint: disable=missing-function-docstring
        w1, w2 = inputs
        B, H, W, _, K2 = w1.shape

        w1_i16 = w1.to(torch.int16)
        w2_i16 = w2.to(torch.int16)

        rng = torch.arange(-self.search_range, self.search_range + 1, device=w1.device)
        jj, ii = torch.meshgrid(rng, rng, indexing="ij")
        vec_lut = -1.0 * torch.stack([jj, ii], dim=-1).reshape(-1, 2).to(torch.float16)

        diff = torch.abs(w1_i16 - w2_i16)
        cost_volume_i32 = torch.sum(diff, dim=-1, dtype=torch.int32)  # [B,H,W,num_ch]

        input_mv_idx = (2 * self.search_range + 1) ** 2
        min_idx_block_match = self.argmin_centered(
            cost_volume_i32[..., :input_mv_idx],
            sz=self.search_range,
            mode="spiral",
            cv_offset=False,
        )
        min_idx_block_match = min_idx_block_match.to(torch.long).unsqueeze(
            -1
        )  # [B,H,W,1]

        idx_bm_exp = min_idx_block_match.expand(B, H, W, 1)
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
        idx = min_idx.unsqueeze(-1).expand(B, H, W, 1, K2)
        min_templates = torch.gather(w1, dim=3, index=idx)

        return vector, min_templates, input_mv_mask, min_cost_volume


class JointBilateralFilter(torch.nn.Module):
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
        N, C, H, W = x.shape
        x = _cast(x, dtype=torch.float32)
        patches = F.unfold(x, kernel_size=(sz, sz), stride=1, padding=sz // 2)
        patches = patches.view(N, C, sz * sz, H, W).permute(0, 3, 4, 2, 1)
        return patches.to(dtype)

    @override
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


def _median_filter2d(
    x: torch.Tensor, ksize=(3, 3), padding: str = "SYMMETRIC"
) -> torch.Tensor:
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


class CalculateFlow(torch.nn.Module):
    """Compute multi-scale optical flow with block matching and MV hints."""

    def __init__(
        self,
        levels=6,
        template_sz=5,
        search_range=3,
        median_kernel=(3, 3),
        blur_levels=None,
        last_bm_level=2,
        min_cv_output=False,
        oob_replacement=False,
        mv_hints=True,
    ):
        super().__init__()
        self.levels = levels
        self.search_range = int(search_range)
        self.template_sz = template_sz
        self.median_kernel = median_kernel
        self.blur_levels = blur_levels if blur_levels is not None else [1, 2, 3, 4]
        self.last_bm_level = last_bm_level
        self.min_cv_output = min_cv_output
        self.oob_replacement = oob_replacement
        self.mv_hints = mv_hints

        self._down = DownSampling2D(size=2, interpolation="bilinear")
        self._warp_bilinear = DenseWarp(interpolation="bilinear")
        self._warp_nearest_oob_zero = DenseWarp(interpolation="nearest_oob_zero")

        self._search_windows = ExtractSearchWindows(
            template_sz=self.template_sz, max_sr=self.search_range
        )
        self._calc_vector = CalculateVector(search_range=self.search_range)
        self._joint_bilateral = JointBilateralFilter(
            kernel_sz=5, sigma_spatial=7.4, sigma_intensity=0.14
        )

    @staticmethod
    def _pad_to_even(tensors: list):
        H, W = tensors[0].shape[2], tensors[0].shape[3]
        h_pad = H % 2
        w_pad = W % 2
        padded = [F.pad(t, (0, w_pad, 0, h_pad), mode="replicate") for t in tensors]
        initial_shape = {"height": H, "width": W}
        return padded, initial_shape

    @override
    def forward(
        self,
        search_frame: torch.Tensor,
        template_frame: torch.Tensor,
        input_mv: torch.Tensor,
    ):
        # pylint: disable=missing-function-docstring
        additional_outputs = {}

        input_mv = input_mv.to(torch.float16)
        (search_frame, template_frame, input_mv), initial_shape = self._pad_to_even(
            [search_frame, template_frame, input_mv]
        )

        search_frame = _cast(search_frame, dtype=torch.uint8)
        template_frame = _cast(template_frame, dtype=torch.uint8)

        search_pyramid = [search_frame]
        template_pyramid = [template_frame]
        shape_pyramid = [initial_shape]
        input_mv_pyramid = [input_mv]

        for i in range(1, self.levels):
            prev_search_img = _cast(search_pyramid[i - 1], dtype=torch.float32)
            prev_template_img = _cast(template_pyramid[i - 1], dtype=torch.float32)
            prev_input_mv = _cast(input_mv_pyramid[i - 1], dtype=torch.float32)

            # pylint: disable-next=superfluous-parens
            if (i - 1) in self.blur_levels:
                prev_search_img = _binomial_filter(prev_search_img)
                prev_template_img = _binomial_filter(prev_template_img)

            search_img_down = self._down(prev_search_img)
            template_img_down = self._down(prev_template_img)
            prev_input_mv = self._down(prev_input_mv) / 2.0

            search_img_down = _cast(search_img_down, dtype=torch.uint8)
            template_img_down = _cast(template_img_down, dtype=torch.uint8)
            prev_input_mv = prev_input_mv.to(torch.float16)

            (
                search_img_down,
                template_img_down,
                prev_input_mv,
            ), shape_down = self._pad_to_even(
                [search_img_down, template_img_down, prev_input_mv]
            )

            search_pyramid.append(search_img_down)
            template_pyramid.append(template_img_down)
            shape_pyramid.append(shape_down)
            input_mv_pyramid.append(prev_input_mv)

        search_pyramid.reverse()
        template_pyramid.reverse()
        shape_pyramid.reverse()
        input_mv_pyramid.reverse()

        vectors = []

        for idx, (search_img, template_img, input_mv_lvl) in enumerate(
            zip(search_pyramid, template_pyramid, input_mv_pyramid)
        ):
            mv_search_img = _cast(search_img, torch.float32)
            mv_warped = self._warp_bilinear(
                [mv_search_img, input_mv_lvl.to(torch.float32)]
            )
            mv_warped = _cast(mv_warped, torch.uint8)
            mv_windows = ExtractTemplates(template_sz=self.template_sz)(
                mv_warped, dtype=torch.uint8
            )

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

                search_img = _cast(search_img, torch.float32)
                search_img = self._warp_bilinear([search_img, vector_prev])
                search_img = _cast(search_img, torch.uint8)
                vector_prev = _cast(vector_prev, torch.float16)
            else:
                vector_prev = None

            windows = self._search_windows(search_img, search_range=self.search_range)

            if self.mv_hints and idx == (len(search_pyramid) - 1) - self.last_bm_level:
                windows = torch.cat([windows, mv_windows], dim=-2)

            templates = ExtractTemplates(template_sz=self.template_sz)(
                template_img, dtype=torch.uint8
            )

            (
                vector_curr,
                min_templates,
                input_mv_mask,
                min_cost_volume,
            ) = self._calc_vector([windows, templates])
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
            vector_curr = self._joint_bilateral([template_img, vector_curr])

            mask = input_mv_mask.permute(0, 3, 1, 2).expand_as(vector_curr)
            vector_curr = torch.where(mask, input_mv_lvl, vector_curr)

            if (
                self.oob_replacement
                and idx == (len(search_pyramid) - 1) - self.last_bm_level
            ):
                blank_img = _cast(
                    torch.ones_like(template_img)[:, :1, :, :], torch.float32
                )
                oob_mask = self._warp_nearest_oob_zero([blank_img, input_mv_lvl])
                vector_curr = torch.where(oob_mask == 0, input_mv_lvl, vector_curr)
                max_cost = torch.tensor(
                    255 * self.template_sz**2 + 1,
                    dtype=torch.int32,
                    device=vector_curr.device,
                )
                min_cost_volume = torch.where(
                    oob_mask.permute(0, 2, 3, 1) == 0, max_cost, min_cost_volume
                )

            shape = shape_pyramid[idx]
            Ht, Wt = shape["height"], shape["width"]
            vector_curr = vector_curr[:, :, :Ht, :Wt]
            min_cost_volume = min_cost_volume[:, :Ht, :Wt, :]
            input_mv_lvl = input_mv_lvl[:, :, :Ht, :Wt]

            vectors.append(vector_curr)

            if idx == (len(search_pyramid) - 1) - self.last_bm_level:
                if self.min_cv_output:
                    additional_outputs["min_cost_volume"] = min_cost_volume
                break

        return vector_curr, additional_outputs


class BlockMatchV3(torch.nn.Module):
    """Model implementing the BlockMatchV3 optical flow algorithm"""

    @override
    def forward(self, inputs: dict):
        # pylint: disable=missing-function-docstring
        img_t = inputs["img_t"]
        img_tm1 = inputs["img_tm1"]
        input_mv = inputs["input_mv"]

        img_t = _rgb_to_y(img_t)
        img_tm1 = _rgb_to_y(img_tm1)

        search_frame = img_t
        template_frame = img_tm1

        # Set output flow polarity
        reversed_polarity = 1.0

        # Set polarity for warps with mv hints
        warp_polarity = 1.0

        flow_out, _ = CalculateFlow()(
            search_frame=search_frame,
            template_frame=template_frame,
            input_mv=input_mv * warp_polarity,
        )
        flow_out = flow_out * reversed_polarity

        return {"output": flow_out}
