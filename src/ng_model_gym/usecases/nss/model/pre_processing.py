# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code
from typing import List

import numpy as np
import torch

from ng_model_gym.core.model.shaders.slang_utils import load_slang_module


# pylint: disable=abstract-method
class PreProcessV1(torch.autograd.Function):  # pylint: disable=invalid-name
    """Neural Super Sampling (NSS) PreProcess layer in PyTorch."""

    @staticmethod
    def forward(
        ctx,
        colour: torch.Tensor,
        history: torch.Tensor,
        motion: torch.Tensor,
        depth: torch.Tensor,
        depth_tm1: torch.Tensor,
        jitter: torch.Tensor,
        jitter_tm1: torch.Tensor,
        feedback_tm1: torch.Tensor,
        derivative_tm1: torch.Tensor,
        depth_params: torch.Tensor,
        exposure: torch.Tensor,
        render_size: torch.Tensor,
        dm_scale: torch.Tensor,
        shader_dir: str,
        shader_file: str,
    ):
        """Forward pass."""

        output_tensor, out_luma_derivative, out_depth_t = pre_process_v1_fwd(
            colour=colour,
            history=history,
            motion=motion,
            depth=depth,
            depth_tm1=depth_tm1,
            jitter=jitter,
            jitter_tm1=jitter_tm1,
            feedback_tm1=feedback_tm1,
            derivative_tm1=derivative_tm1,
            depth_params=depth_params,
            exposure=exposure,
            render_size=render_size,
            dm_scale=dm_scale,
            shader_dir=shader_dir,
            shader_file=shader_file,
        )

        ctx.save_for_backward(
            colour,
            history,
            motion,
            depth,
            depth_tm1,
            jitter,
            jitter_tm1,
            feedback_tm1,
            derivative_tm1,
            depth_params,
            exposure,
            render_size,
            dm_scale,
            output_tensor,
            out_luma_derivative,
            out_depth_t,
        )

        ctx.shader_dir = shader_dir
        ctx.shader_file = shader_file

        return output_tensor, out_luma_derivative, out_depth_t

    # pylint: disable=unused-argument
    @staticmethod
    def backward(
        ctx,
        grad_output_tensor: torch.Tensor,
        grad_out_luma_derivative: torch.Tensor,
        grad_out_depth_t: torch.Tensor,
    ):
        """Backward pass."""

        (
            colour,
            history,
            motion,
            depth,
            depth_tm1,
            jitter,
            jitter_tm1,
            feedback_tm1,
            derivative_tm1,
            depth_params,
            exposure,
            render_size,
            dm_scale,
            output_tensor,
            out_luma_derivative,
            out_depth_t,
        ) = ctx.saved_tensors

        shader_dir = ctx.shader_dir
        shader_file = ctx.shader_file

        grad_history, grad_feedback_tm1, grad_dm_scale = pre_process_v1_bwd(
            colour=colour,
            history=history,
            motion=motion,
            depth=depth,
            depth_tm1=depth_tm1,
            jitter=jitter,
            jitter_tm1=jitter_tm1,
            feedback_tm1=feedback_tm1,
            derivative_tm1=derivative_tm1,
            depth_params=depth_params,
            exposure=exposure,
            render_size=render_size,
            dm_scale=dm_scale,
            output_tensor=[output_tensor, grad_output_tensor],
            out_luma_derivative=out_luma_derivative,
            out_depth_t=out_depth_t,
            shader_dir=shader_dir,
            shader_file=shader_file,
        )

        return (
            None,
            grad_history,
            None,
            None,
            None,
            None,
            None,
            grad_feedback_tm1,
            None,
            None,
            None,
            None,
            grad_dm_scale,
            None,
            None,
        )

    # pylint: enable=unused-argument


@torch.library.custom_op("nss_v1::pre_process_v1_fwd", mutates_args=())
def pre_process_v1_fwd(
    colour: torch.Tensor,
    history: torch.Tensor,
    motion: torch.Tensor,
    depth: torch.Tensor,
    depth_tm1: torch.Tensor,
    jitter: torch.Tensor,
    jitter_tm1: torch.Tensor,
    feedback_tm1: torch.Tensor,
    derivative_tm1: torch.Tensor,
    depth_params: torch.Tensor,
    exposure: torch.Tensor,
    render_size: torch.Tensor,
    dm_scale: torch.Tensor,
    shader_dir: str,
    shader_file: str,
) -> List[torch.Tensor]:
    """Forward pass."""

    m = load_slang_module(shader_dir, shader_file)

    # Define output(s)
    b, _, h, w = colour.shape
    output_tensor = torch.zeros((b, 12, h, w), device=colour.device)
    out_luma_derivative = torch.zeros_like(derivative_tm1)
    out_depth_t = torch.zeros_like(depth)

    # Define dispatch dimensions
    block_sz = 512
    dispatch_size = [
        output_tensor.shape[0],
        output_tensor.shape[2],
        output_tensor.shape[3],
    ]
    kernel_with_args = m.pre_process_v1(
        colour=colour,
        history=history,
        motion=motion,
        depth=depth,
        depth_tm1=depth_tm1,
        jitter=jitter,
        jitter_tm1=jitter_tm1,
        feedback_tm1=feedback_tm1,
        derivative_tm1=derivative_tm1,
        depth_params=depth_params,
        exposure=exposure,
        render_size=render_size,
        dm_scale=dm_scale,
        output_tensor=output_tensor,
        out_luma_derivative=out_luma_derivative,
        out_depth_t=out_depth_t,
    )

    kernel_with_args.launchRaw(
        blockSize=(block_sz, 1, 1),
        gridSize=(int((np.prod(dispatch_size) + block_sz - 1) // block_sz), 1, 1),
    )

    return output_tensor, out_luma_derivative, out_depth_t


# pylint: disable=unused-argument
@torch.library.register_fake("nss_v1::pre_process_v1_fwd")
def pre_process_v1_fwd_abstract(
    colour: torch.Tensor,
    history: torch.Tensor,
    motion: torch.Tensor,
    depth: torch.Tensor,
    depth_tm1: torch.Tensor,
    jitter: torch.Tensor,
    jitter_tm1: torch.Tensor,
    feedback_tm1: torch.Tensor,
    derivative_tm1: torch.Tensor,
    depth_params: torch.Tensor,
    exposure: torch.Tensor,
    render_size: torch.Tensor,
    dm_scale: torch.Tensor,
    shader_dir: str,
    shader_file: str,
) -> List[torch.Tensor]:
    """Abstract forward pass."""

    # Define output(s)
    b, _, h, w = colour.shape
    output_tensor = torch.zeros((b, 12, h, w), device=colour.device)
    out_luma_derivative = torch.zeros_like(derivative_tm1)
    out_depth_t = torch.zeros_like(depth)

    return (
        torch.empty_like(output_tensor),
        torch.empty_like(out_luma_derivative),
        torch.empty_like(out_depth_t),
    )


# pylint: enable=unused-argument


@torch.library.custom_op("nss_v1::pre_process_v1_bwd", mutates_args=())
def pre_process_v1_bwd(
    colour: torch.Tensor,
    history: torch.Tensor,
    motion: torch.Tensor,
    depth: torch.Tensor,
    depth_tm1: torch.Tensor,
    jitter: torch.Tensor,
    jitter_tm1: torch.Tensor,
    feedback_tm1: torch.Tensor,
    derivative_tm1: torch.Tensor,
    depth_params: torch.Tensor,
    exposure: torch.Tensor,
    render_size: torch.Tensor,
    dm_scale: torch.Tensor,
    output_tensor: List[torch.Tensor],
    out_luma_derivative: torch.Tensor,
    out_depth_t: torch.Tensor,
    shader_dir: str,
    shader_file: str,
) -> List[torch.Tensor]:
    """Backward pass."""

    m = load_slang_module(shader_dir, shader_file)

    # Gradients
    grad_history = torch.zeros_like(history)
    grad_feedback_tm1 = torch.zeros_like(feedback_tm1)
    grad_dm_scale = torch.zeros_like(dm_scale)

    # Define dispatch dimensions
    block_sz = 512
    dispatch_size = [
        output_tensor[0].shape[0],
        output_tensor[0].shape[2],
        output_tensor[0].shape[3],
    ]

    kernel_with_args = m.pre_process_v1.bwd(
        colour=colour,
        history=(history, grad_history),
        motion=motion,
        depth=depth,
        depth_tm1=depth_tm1,
        jitter=jitter,
        jitter_tm1=jitter_tm1,
        feedback_tm1=(feedback_tm1, grad_feedback_tm1),
        derivative_tm1=derivative_tm1,
        depth_params=depth_params,
        exposure=exposure,
        render_size=render_size,
        dm_scale=(dm_scale, grad_dm_scale),
        output_tensor=tuple(output_tensor),
        out_luma_derivative=out_luma_derivative,
        out_depth_t=out_depth_t,
    )

    kernel_with_args.launchRaw(
        blockSize=(block_sz, 1, 1),
        gridSize=(int((np.prod(dispatch_size) + block_sz - 1) // block_sz), 1, 1),
    )

    return grad_history, grad_feedback_tm1, grad_dm_scale


# pylint: disable=unused-argument
@torch.library.register_fake("nss_v1::pre_process_v1_bwd")
def pre_process_v1_bwd_abstract(
    colour: torch.Tensor,
    history: torch.Tensor,
    motion: torch.Tensor,
    depth: torch.Tensor,
    depth_tm1: torch.Tensor,
    jitter: torch.Tensor,
    jitter_tm1: torch.Tensor,
    feedback_tm1: torch.Tensor,
    derivative_tm1: torch.Tensor,
    depth_params: torch.Tensor,
    exposure: torch.Tensor,
    render_size: torch.Tensor,
    dm_scale: torch.Tensor,
    output_tensor: List[torch.Tensor],
    out_luma_derivative: torch.Tensor,
    out_depth_t: torch.Tensor,
    shader_dir: str,
    shader_file: str,
) -> List[torch.Tensor]:
    """Abstract backward pass."""

    # Gradients
    grad_history = torch.zeros_like(history)
    grad_feedback_tm1 = torch.zeros_like(feedback_tm1)
    grad_dm_scale = torch.zeros_like(dm_scale)
    return (
        torch.empty_like(grad_history),
        torch.empty_like(grad_feedback_tm1),
        torch.empty_like(grad_dm_scale),
    )


# pylint: enable=unused-argument


class PreProcessV1_ShaderAccurate(
    torch.autograd.Function
):  # pylint: disable=invalid-name
    """Neural Super Sampling (NSS) "shader accurate" PreProcess shader in PyTorch."""

    @staticmethod
    def forward(
        ctx,
        colour: torch.Tensor,
        history: torch.Tensor,
        motion: torch.Tensor,
        depth: torch.Tensor,
        depth_tm1: torch.Tensor,
        nearest_offset_tm1: torch.Tensor,
        jitter: torch.Tensor,
        jitter_tm1: torch.Tensor,
        feedback_tm1: torch.Tensor,
        derivative_tm1: torch.Tensor,
        depth_params: torch.Tensor,
        exposure: torch.Tensor,
        render_size: torch.Tensor,
        dm_scale: torch.Tensor,
        shader_dir: str,
        shader_file: str,
    ):
        """Shader accurate forward pass"""

        output_tensor, out_luma_derivative, out_depth_t = pre_process_v1_sa_fwd(
            colour=colour,
            history=history,
            motion=motion,
            depth=depth,
            depth_tm1=depth_tm1,
            nearest_offset_tm1=nearest_offset_tm1,
            jitter=jitter,
            jitter_tm1=jitter_tm1,
            feedback_tm1=feedback_tm1,
            derivative_tm1=derivative_tm1,
            depth_params=depth_params,
            exposure=exposure,
            render_size=render_size,
            dm_scale=dm_scale,
            shader_dir=shader_dir,
            shader_file=shader_file,
        )

        ctx.save_for_backward(
            colour,
            history,
            motion,
            depth,
            depth_tm1,
            nearest_offset_tm1,
            jitter,
            jitter_tm1,
            feedback_tm1,
            derivative_tm1,
            depth_params,
            exposure,
            render_size,
            dm_scale,
            output_tensor,
            out_luma_derivative,
            out_depth_t,
        )

        ctx.shader_dir = shader_dir
        ctx.shader_file = shader_file

        return output_tensor, out_luma_derivative, out_depth_t

    # pylint: disable=unused-argument
    @staticmethod
    def backward(
        ctx,
        grad_output_tensor: torch.Tensor,
        grad_out_luma_derivative: torch.Tensor,
        grad_out_depth_t: torch.Tensor,
    ):
        """Shader accurate backward pass"""

        (
            colour,
            history,
            motion,
            depth,
            depth_tm1,
            nearest_offset_tm1,
            jitter,
            jitter_tm1,
            feedback_tm1,
            derivative_tm1,
            depth_params,
            exposure,
            render_size,
            dm_scale,
            output_tensor,
            out_luma_derivative,
            out_depth_t,
        ) = ctx.saved_tensors

        shader_dir = ctx.shader_dir
        shader_file = ctx.shader_file

        grad_history, grad_feedback_tm1, grad_dm_scale = pre_process_v1_sa_bwd(
            colour=colour,
            history=history,
            motion=motion,
            depth=depth,
            depth_tm1=depth_tm1,
            nearest_offset_tm1=nearest_offset_tm1,
            jitter=jitter,
            jitter_tm1=jitter_tm1,
            feedback_tm1=feedback_tm1,
            derivative_tm1=derivative_tm1,
            depth_params=depth_params,
            exposure=exposure,
            render_size=render_size,
            dm_scale=dm_scale,
            output_tensor=[output_tensor, grad_output_tensor],
            out_luma_derivative=out_luma_derivative,
            out_depth_t=out_depth_t,
            shader_dir=shader_dir,
            shader_file=shader_file,
        )

        return (
            None,
            grad_history,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_feedback_tm1,
            None,
            None,
            None,
            None,
            grad_dm_scale,
            None,
            None,
        )

    # pylint: enable=unused-argument


@torch.library.custom_op("nss_v1::pre_process_v1_sa_fwd", mutates_args=())
def pre_process_v1_sa_fwd(
    colour: torch.Tensor,
    history: torch.Tensor,
    motion: torch.Tensor,
    depth: torch.Tensor,
    depth_tm1: torch.Tensor,
    nearest_offset_tm1: torch.Tensor,
    jitter: torch.Tensor,
    jitter_tm1: torch.Tensor,
    feedback_tm1: torch.Tensor,
    derivative_tm1: torch.Tensor,
    depth_params: torch.Tensor,
    exposure: torch.Tensor,
    render_size: torch.Tensor,
    dm_scale: torch.Tensor,
    shader_dir: str,
    shader_file: str,
) -> List[torch.Tensor]:
    """Shader accurate forward pass"""

    m = load_slang_module(shader_dir, shader_file)

    # Define output(s)
    b, _, h, w = colour.shape
    output_tensor = torch.zeros((b, 12, h, w), device=colour.device)
    out_luma_derivative = torch.zeros_like(derivative_tm1)
    out_depth_t = torch.zeros_like(depth)

    # Define dispatch dimensions
    block_sz = 512
    dispatch_size = [
        output_tensor.shape[0],
        output_tensor.shape[2],
        output_tensor.shape[3],
    ]
    kernel_with_args = m.pre_process_v1_shader_accurate(
        colour=colour,
        history=history,
        motion=motion,
        depth=depth,
        depth_tm1=depth_tm1,
        nearest_offset_tm1=nearest_offset_tm1,
        jitter=jitter,
        jitter_tm1=jitter_tm1,
        feedback_tm1=feedback_tm1,
        derivative_tm1=derivative_tm1,
        depth_params=depth_params,
        exposure=exposure,
        render_size=render_size,
        dm_scale=dm_scale,
        output_tensor=output_tensor,
        out_luma_derivative=out_luma_derivative,
        out_depth_t=out_depth_t,
    )

    kernel_with_args.launchRaw(
        blockSize=(block_sz, 1, 1),
        gridSize=(int((np.prod(dispatch_size) + block_sz - 1) // block_sz), 1, 1),
    )

    return output_tensor, out_luma_derivative, out_depth_t


# pylint: disable=unused-argument
@torch.library.register_fake("nss_v1::pre_process_v1_sa_fwd")
def pre_process_v1_fwd_sa_abstract(
    colour: torch.Tensor,
    history: torch.Tensor,
    motion: torch.Tensor,
    depth: torch.Tensor,
    depth_tm1: torch.Tensor,
    nearest_offset_tm1: torch.Tensor,
    jitter: torch.Tensor,
    jitter_tm1: torch.Tensor,
    feedback_tm1: torch.Tensor,
    derivative_tm1: torch.Tensor,
    depth_params: torch.Tensor,
    exposure: torch.Tensor,
    render_size: torch.Tensor,
    dm_scale: torch.Tensor,
    shader_dir: str,
    shader_file: str,
) -> List[torch.Tensor]:
    """Shader accurate abstract forward pass"""

    # Define output(s)
    b, _, h, w = colour.shape
    output_tensor = torch.zeros((b, 12, h, w), device=colour.device)
    out_luma_derivative = torch.zeros_like(derivative_tm1)
    out_depth_t = torch.zeros_like(depth)

    return (
        torch.empty_like(output_tensor),
        torch.empty_like(out_luma_derivative),
        torch.empty_like(out_depth_t),
    )


# pylint: enable=unused-argument


@torch.library.custom_op("nss_v1::pre_process_v1_sa_bwd", mutates_args=())
def pre_process_v1_sa_bwd(
    colour: torch.Tensor,
    history: torch.Tensor,
    motion: torch.Tensor,
    depth: torch.Tensor,
    depth_tm1: torch.Tensor,
    nearest_offset_tm1: torch.Tensor,
    jitter: torch.Tensor,
    jitter_tm1: torch.Tensor,
    feedback_tm1: torch.Tensor,
    derivative_tm1: torch.Tensor,
    depth_params: torch.Tensor,
    exposure: torch.Tensor,
    render_size: torch.Tensor,
    dm_scale: torch.Tensor,
    output_tensor: List[torch.Tensor],
    out_luma_derivative: torch.Tensor,
    out_depth_t: torch.Tensor,
    shader_dir: str,
    shader_file: str,
) -> List[torch.Tensor]:
    """Shader accurate backward pass"""

    m = load_slang_module(shader_dir, shader_file)

    # Gradients
    grad_history = torch.zeros_like(history)
    grad_feedback_tm1 = torch.zeros_like(feedback_tm1)
    grad_dm_scale = torch.zeros_like(dm_scale)

    # Define dispatch dimensions
    block_sz = 512
    dispatch_size = [
        output_tensor[0].shape[0],
        output_tensor[0].shape[2],
        output_tensor[0].shape[3],
    ]

    kernel_with_args = m.pre_process_v1_shader_accurate.bwd(
        colour=colour,
        history=(history, grad_history),
        motion=motion,
        depth=depth,
        depth_tm1=depth_tm1,
        nearest_offset_tm1=nearest_offset_tm1,
        jitter=jitter,
        jitter_tm1=jitter_tm1,
        feedback_tm1=(feedback_tm1, grad_feedback_tm1),
        derivative_tm1=derivative_tm1,
        depth_params=depth_params,
        exposure=exposure,
        render_size=render_size,
        dm_scale=(dm_scale, grad_dm_scale),
        output_tensor=tuple(output_tensor),
        out_luma_derivative=out_luma_derivative,
        out_depth_t=out_depth_t,
    )

    kernel_with_args.launchRaw(
        blockSize=(block_sz, 1, 1),
        gridSize=(int((np.prod(dispatch_size) + block_sz - 1) // block_sz), 1, 1),
    )

    return grad_history, grad_feedback_tm1, grad_dm_scale


# pylint: disable=unused-argument
@torch.library.register_fake("nss_v1::pre_process_v1_sa_bwd")
def pre_process_v1_sa_bwd_abstract(
    colour: torch.Tensor,
    history: torch.Tensor,
    motion: torch.Tensor,
    depth: torch.Tensor,
    depth_tm1: torch.Tensor,
    nearest_offset_tm1: torch.Tensor,
    jitter: torch.Tensor,
    jitter_tm1: torch.Tensor,
    feedback_tm1: torch.Tensor,
    derivative_tm1: torch.Tensor,
    depth_params: torch.Tensor,
    exposure: torch.Tensor,
    render_size: torch.Tensor,
    dm_scale: torch.Tensor,
    output_tensor: List[torch.Tensor],
    out_luma_derivative: torch.Tensor,
    out_depth_t: torch.Tensor,
    shader_dir: str,
    shader_file: str,
) -> List[torch.Tensor]:
    """Abstract shader accurate backward pass"""

    # Gradients
    grad_history = torch.zeros_like(history)
    grad_feedback_tm1 = torch.zeros_like(feedback_tm1)
    grad_dm_scale = torch.zeros_like(dm_scale)
    return (
        torch.empty_like(grad_history),
        torch.empty_like(grad_feedback_tm1),
        torch.empty_like(grad_dm_scale),
    )


# pylint: enable=unused-argument
