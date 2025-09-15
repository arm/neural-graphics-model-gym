# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code
from typing import List

import numpy as np
import torch

from ng_model_gym.usecases.nss.model.shaders.slang_utils import load_slang_module

# pylint: disable=abstract-method


class PostProcessV1(torch.autograd.Function):  # pylint: disable=invalid-name
    """Neural Super Sampling (NSS) PostProcess Shader in PyTorch."""

    @staticmethod
    def forward(
        ctx,
        colour: torch.Tensor,
        history: torch.Tensor,
        t_kpn_params: torch.Tensor,
        temporal_params: torch.Tensor,
        motion: torch.Tensor,
        exposure: torch.Tensor,
        jitter: torch.Tensor,
        offset_lut: torch.Tensor,
        scale: torch.Tensor,
        idx_modulo: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Forward pass"""
        output, out_filtered = post_process_v1_fwd(
            colour,
            history,
            t_kpn_params,
            temporal_params,
            motion,
            exposure,
            jitter,
            offset_lut,
            scale,
            idx_modulo,
        )
        ctx.save_for_backward(
            colour,
            history,
            t_kpn_params,
            temporal_params,
            motion,
            exposure,
            jitter,
            offset_lut,
            scale,
            idx_modulo,
            output,
            out_filtered,
        )

        return output, out_filtered

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor, grad_out_filtered: torch.Tensor
    ) -> List[torch.Tensor]:
        """Backward pass"""
        (
            colour,
            history,
            t_kpn_params,
            temporal_params,
            motion,
            exposure,
            jitter,
            offset_lut,
            scale,
            idx_modulo,
            output,
            out_filtered,
        ) = ctx.saved_tensors

        (
            grad_history,
            grad_t_kpn_params,
            grad_temporal_params,
        ) = post_process_v1_bwd(
            colour,
            history,
            t_kpn_params,
            temporal_params,
            motion,
            exposure,
            jitter,
            offset_lut,
            scale,
            idx_modulo,
            [output, grad_output],
            [out_filtered, grad_out_filtered],
        )

        return (
            None,
            grad_history,
            grad_t_kpn_params,
            grad_temporal_params,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@torch.library.custom_op("nss_v1::post_process_v1_fwd", mutates_args=())
def post_process_v1_fwd(
    colour: torch.Tensor,
    history: torch.Tensor,
    t_kpn_params: torch.Tensor,
    temporal_params: torch.Tensor,
    motion: torch.Tensor,
    exposure: torch.Tensor,
    jitter: torch.Tensor,
    offset_lut: torch.Tensor,
    scale: torch.Tensor,
    idx_modulo: torch.Tensor,
) -> List[torch.Tensor]:
    """Forward pass"""
    m = load_slang_module()
    output = torch.zeros_like(history)
    out_filtered = torch.zeros_like(history)

    block_sz = 512
    dispath_size = [output.shape[0], output.shape[2], output.shape[3]]
    kernel_with_args = m.post_process_v1(
        colour=colour,
        history=history,
        t_kpn_params=t_kpn_params,
        temporal_params=temporal_params,
        motion=motion,
        exposure=exposure,
        jitter=jitter,
        offset_lut=offset_lut,
        scale=scale,
        idx_modulo=idx_modulo,
        output=output,
        out_filtered=out_filtered,
    )
    kernel_with_args.launchRaw(
        blockSize=(block_sz, 1, 1),
        gridSize=(int((np.prod(dispath_size) + block_sz - 1) // block_sz), 1, 1),
    )

    return output, out_filtered


# pylint: disable=unused-argument
@torch.library.register_fake("nss_v1::post_process_v1_fwd")
def post_process_v1_fwd_abstract(
    colour: torch.Tensor,
    history: torch.Tensor,
    t_kpn_params: torch.Tensor,
    temporal_params: torch.Tensor,
    motion: torch.Tensor,
    exposure: torch.Tensor,
    jitter: torch.Tensor,
    offset_lut: torch.Tensor,
    scale: torch.Tensor,
    idx_modulo: torch.Tensor,
) -> List[torch.Tensor]:
    """Forward pass"""
    return torch.empty_like(history), torch.empty_like(history)


# pylint: enable=unused-argument


@torch.library.custom_op("nss_v1::post_process_v1_bwd", mutates_args=())
def post_process_v1_bwd(
    colour: torch.Tensor,
    history: torch.Tensor,
    t_kpn_params: torch.Tensor,
    temporal_params: torch.Tensor,
    motion: torch.Tensor,
    exposure: torch.Tensor,
    jitter: torch.Tensor,
    offset_lut: torch.Tensor,
    scale: torch.Tensor,
    idx_modulo: torch.Tensor,
    output: List[torch.Tensor],
    out_filtered: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Backward pass"""
    m = load_slang_module()
    grad_history = torch.zeros_like(history)
    grad_t_kpn_params = torch.zeros_like(t_kpn_params)
    grad_temporal_params = torch.zeros_like(temporal_params)

    block_sz = 256
    dispath_size = [output[0].shape[0], output[0].shape[2], output[0].shape[3]]
    kernel_with_args = m.post_process_v1.bwd(
        colour=colour,
        history=(history, grad_history),
        t_kpn_params=(t_kpn_params, grad_t_kpn_params),
        temporal_params=(temporal_params, grad_temporal_params),
        motion=motion,
        exposure=exposure,
        jitter=jitter,
        offset_lut=offset_lut,
        scale=scale,
        idx_modulo=idx_modulo,
        output=tuple(output),
        out_filtered=tuple(out_filtered),
    )
    kernel_with_args.launchRaw(
        blockSize=(block_sz, 1, 1),
        gridSize=(int((np.prod(dispath_size) + block_sz - 1) // block_sz), 1, 1),
    )

    return (
        grad_history,
        grad_t_kpn_params,
        grad_temporal_params,
    )


# pylint: disable=unused-argument
@torch.library.register_fake("nss_v1::post_process_v1_bwd")
def post_process_v1_bwd_abstract(
    colour: torch.Tensor,
    history: torch.Tensor,
    t_kpn_params: torch.Tensor,
    temporal_params: torch.Tensor,
    motion: torch.Tensor,
    exposure: torch.Tensor,
    jitter: torch.Tensor,
    offset_lut: torch.Tensor,
    scale: torch.Tensor,
    idx_modulo: torch.Tensor,
    output: List[torch.Tensor],
    out_filtered: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Backward pass"""
    return (
        torch.empty_like(history),
        torch.empty_like(t_kpn_params),
        torch.empty_like(temporal_params),
    )


# pylint: enable=unused-argument


class PostProcessV1_ShaderAccurate(
    torch.autograd.Function
):  # pylint: disable=invalid-name
    """Neural Super Sampling (NSS) "shader accurate" PostProcess Shader in PyTorch."""

    @staticmethod
    def forward(
        ctx,
        colour: torch.Tensor,
        history: torch.Tensor,
        t_kpn_params: torch.Tensor,
        temporal_params: torch.Tensor,
        motion: torch.Tensor,
        nearest_offset: torch.Tensor,
        exposure: torch.Tensor,
        jitter: torch.Tensor,
        offset_lut: torch.Tensor,
        scale: torch.Tensor,
        idx_modulo: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Shader accurate forward pass"""
        output, out_filtered = post_process_v1_sa_fwd(
            colour,
            history,
            t_kpn_params,
            temporal_params,
            motion,
            nearest_offset,
            exposure,
            jitter,
            offset_lut,
            scale,
            idx_modulo,
        )
        ctx.save_for_backward(
            colour,
            history,
            t_kpn_params,
            temporal_params,
            motion,
            nearest_offset,
            exposure,
            jitter,
            offset_lut,
            scale,
            idx_modulo,
            output,
            out_filtered,
        )

        return output, out_filtered

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor, grad_out_filtered: torch.Tensor
    ) -> List[torch.Tensor]:
        """Shader accurate backward pass"""
        (
            colour,
            history,
            t_kpn_params,
            temporal_params,
            motion,
            nearest_offset,
            exposure,
            jitter,
            offset_lut,
            scale,
            idx_modulo,
            output,
            out_filtered,
        ) = ctx.saved_tensors

        (
            grad_history,
            grad_t_kpn_params,
            grad_temporal_params,
        ) = post_process_v1_sa_bwd(
            colour,
            history,
            t_kpn_params,
            temporal_params,
            motion,
            nearest_offset,
            exposure,
            jitter,
            offset_lut,
            scale,
            idx_modulo,
            [output, grad_output],
            [out_filtered, grad_out_filtered],
        )

        return (
            None,
            grad_history,
            grad_t_kpn_params,
            grad_temporal_params,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@torch.library.custom_op("nss_v1::post_process_v1_sa_fwd", mutates_args=())
def post_process_v1_sa_fwd(
    colour: torch.Tensor,
    history: torch.Tensor,
    t_kpn_params: torch.Tensor,
    temporal_params: torch.Tensor,
    motion: torch.Tensor,
    nearest_offset: torch.Tensor,
    exposure: torch.Tensor,
    jitter: torch.Tensor,
    offset_lut: torch.Tensor,
    scale: torch.Tensor,
    idx_modulo: torch.Tensor,
) -> List[torch.Tensor]:
    """Shader accurate forward pass"""
    m = load_slang_module()
    output = torch.zeros_like(history)
    out_filtered = torch.zeros_like(history)

    block_sz = 512
    dispath_size = [output.shape[0], output.shape[2], output.shape[3]]
    kernel_with_args = m.post_process_v1_shader_accurate(
        colour=colour,
        history=history,
        t_kpn_params=t_kpn_params,
        temporal_params=temporal_params,
        motion=motion,
        nearest_offset=nearest_offset,
        exposure=exposure,
        jitter=jitter,
        offset_lut=offset_lut,
        scale=scale,
        idx_modulo=idx_modulo,
        output=output,
        out_filtered=out_filtered,
    )
    kernel_with_args.launchRaw(
        blockSize=(block_sz, 1, 1),
        gridSize=(int((np.prod(dispath_size) + block_sz - 1) // block_sz), 1, 1),
    )

    return output, out_filtered


# pylint: disable=unused-argument
@torch.library.register_fake("nss_v1::post_process_v1_sa_fwd")
def post_process_v1_sa_fwd_abstract(
    colour: torch.Tensor,
    history: torch.Tensor,
    t_kpn_params: torch.Tensor,
    temporal_params: torch.Tensor,
    motion: torch.Tensor,
    nearest_offset: torch.Tensor,
    exposure: torch.Tensor,
    jitter: torch.Tensor,
    offset_lut: torch.Tensor,
    scale: torch.Tensor,
    idx_modulo: torch.Tensor,
) -> List[torch.Tensor]:
    """Abstract shader accurate forward pass"""
    return torch.empty_like(history), torch.empty_like(history)


# pylint: enable=unused-argument


@torch.library.custom_op("nss_v1::post_process_v1_sa_bwd", mutates_args=())
def post_process_v1_sa_bwd(
    colour: torch.Tensor,
    history: torch.Tensor,
    t_kpn_params: torch.Tensor,
    temporal_params: torch.Tensor,
    motion: torch.Tensor,
    nearest_offset: torch.Tensor,
    exposure: torch.Tensor,
    jitter: torch.Tensor,
    offset_lut: torch.Tensor,
    scale: torch.Tensor,
    idx_modulo: torch.Tensor,
    output: List[torch.Tensor],
    out_filtered: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Shader accurate backward pass"""
    m = load_slang_module()
    grad_history = torch.zeros_like(history)
    grad_t_kpn_params = torch.zeros_like(t_kpn_params)
    grad_temporal_params = torch.zeros_like(temporal_params)

    block_sz = 256
    dispath_size = [output[0].shape[0], output[0].shape[2], output[0].shape[3]]
    kernel_with_args = m.post_process_v1_shader_accurate.bwd(
        colour=colour,
        history=(history, grad_history),
        t_kpn_params=(t_kpn_params, grad_t_kpn_params),
        temporal_params=(temporal_params, grad_temporal_params),
        motion=motion,
        nearest_offset=nearest_offset,
        exposure=exposure,
        jitter=jitter,
        offset_lut=offset_lut,
        scale=scale,
        idx_modulo=idx_modulo,
        output=tuple(output),
        out_filtered=tuple(out_filtered),
    )
    kernel_with_args.launchRaw(
        blockSize=(block_sz, 1, 1),
        gridSize=(int((np.prod(dispath_size) + block_sz - 1) // block_sz), 1, 1),
    )

    return (
        grad_history,
        grad_t_kpn_params,
        grad_temporal_params,
    )


# pylint: disable=unused-argument
@torch.library.register_fake("nss_v1::post_process_v1_sa_bwd")
def post_process_v1_sa_bwd_abstract(
    colour: torch.Tensor,
    history: torch.Tensor,
    t_kpn_params: torch.Tensor,
    temporal_params: torch.Tensor,
    motion: torch.Tensor,
    nearest_offset: torch.Tensor,
    exposure: torch.Tensor,
    jitter: torch.Tensor,
    offset_lut: torch.Tensor,
    scale: torch.Tensor,
    idx_modulo: torch.Tensor,
    output: List[torch.Tensor],
    out_filtered: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Abstract shader accurate backward pass"""
    return (
        torch.empty_like(history),
        torch.empty_like(t_kpn_params),
        torch.empty_like(temporal_params),
    )


# pylint: enable=unused-argument
