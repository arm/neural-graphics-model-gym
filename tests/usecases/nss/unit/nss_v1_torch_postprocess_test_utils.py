# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import unittest
from dataclasses import dataclass
from typing import NamedTuple

import torch

from ng_model_gym.usecases.nss.model.model_v1 import NSSV1Model
from ng_model_gym.usecases.nss.model.torch_postprocess.filter import (
    filter_color,
    kpn_coordinates,
)
from ng_model_gym.usecases.nss.model.torch_postprocess.sampling import (
    decode_nearest_offsets,
    load_motion,
    sample_temporal_params,
)
from tests.usecases.nss.unit.nss_v1_test_utils import create_nss_v1_test_params

# Shape setup intentionally mirrors the shared fixture builder and Slang goldens.
# pylint: disable=duplicate-code

_SHADER_RTOL = 1.0e-5
_SHADER_ATOL = 5.0e-6
_QUALITIES = ("high", "mid", "low")


@dataclass
class NSSV1PostprocessCase:
    """Compact inputs for exercising an NSS v1 post-processing backend."""

    model: NSSV1Model
    inputs: dict[str, torch.Tensor]
    kpn_params: torch.Tensor
    temporal_params: torch.Tensor
    nearest_depth_offset: torch.Tensor
    derivative: torch.Tensor
    disocclusion_mask: torch.Tensor
    hr_shape: tuple[int, int, int, int]


def _rand(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Create one deterministic float32 tensor without using the global RNG."""

    return torch.rand(shape, generator=generator, device=device, dtype=torch.float32)


def _create_nearest_depth_offsets(
    model: NSSV1Model,
    shape: tuple[int, int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """Encode diagonal nearest-depth patterns in the shader's y/x layout."""

    batch, _, height, width = shape
    if not model.packed_nearest_offset_quad:
        # EncodeNearestDepthCoord stores ((x + 2) << 3) | (y + 2). Tensor
        # coordinates are row(y)/column(x), and the diagonal values make that
        # convention explicit without depending on arbitrary floating-point codes.
        encoded_codes = torch.tensor((0, 18, 36), device=device, dtype=torch.float32)
        pattern = (
            torch.arange(height, device=device).view(height, 1)
            + torch.arange(width, device=device).view(1, width)
        ) % encoded_codes.numel()
        encoded = encoded_codes[pattern] / 255.0
        return encoded.view(1, 1, height, width).expand(batch, -1, -1, -1).clone()

    # EncodeNearestDepthCoordNibble stores ((x + 1) << 2) | (y + 1).
    # post_process selects lane 00/10 from channel 0 and lane 01/11 from
    # channel 1, with the even lane in the low nibble.
    lane_nibbles = (0, 5, 10, 15)
    packed_bytes = torch.tensor(
        (
            lane_nibbles[0] | (lane_nibbles[1] << 4),
            lane_nibbles[2] | (lane_nibbles[3] << 4),
        ),
        device=device,
        dtype=torch.float32,
    )
    return (
        (packed_bytes / 255.0).view(1, 2, 1, 1).expand(batch, -1, height, width).clone()
    )


def create_nss_v1_postprocess_case(
    quality: str,
    device: torch.device,
    *,
    backend: str,
    scale: float = 2.0,
    reset_value: float = 1.0,
    exposure: float = 1.0,
    lr_shape: tuple[int, int] = (16, 24),
    seed: int = 1234,
    batch: int = 2,
) -> NSSV1PostprocessCase:
    """Create deterministic, nontrivial NSS v1 post-processing inputs."""

    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    params = create_nss_v1_test_params(quality)
    params.train.batch_size = batch
    params.model.processing_backend = backend
    params.model.scale = scale

    # Module initialization internally uses PyTorch's global CPU generator.
    # Isolate and seed that unavoidable framework behavior so this fixture is
    # deterministic without changing the caller's RNG state. All fixture
    # tensors below use the explicit device-local generator.
    model_generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(model_generator.get_state())
        model = NSSV1Model(params)
    model = model.to(device).eval()

    lr_height, lr_width = lr_shape
    colour_linear = 0.05 + 0.9 * _rand(
        (batch, 3, lr_height, lr_width), generator=generator, device=device
    )

    shape_inputs = {"colour_linear": colour_linear}
    (
        input_shape,
        process_shape,
        hr_shape,
        pad_shape,
        _,
    ) = model._calculate_dispatch_dims(
        shape_inputs
    )  # pylint: disable=protected-access

    history_random = _rand(hr_shape, generator=generator, device=device)
    history_ramp = torch.linspace(
        0.0,
        0.2,
        hr_shape[2] * hr_shape[3],
        device=device,
        dtype=torch.float32,
    ).reshape(1, 1, hr_shape[2], hr_shape[3])
    history = 0.1 + 0.6 * history_random + history_ramp
    ground_truth_linear = 0.05 + 0.95 * _rand(
        hr_shape, generator=generator, device=device
    )

    motion_lr = 0.08 * (
        2.0
        * _rand(
            (batch, 2, lr_height, lr_width),
            generator=generator,
            device=device,
        )
        - 1.0
    )
    jitter_values = torch.tensor(
        ((0.25, -0.25), (-0.125, 0.375)),
        device=device,
        dtype=torch.float32,
    )[:batch]
    jitter = jitter_values.reshape(batch, 2, 1, 1)

    inputs = {
        "colour_linear": colour_linear,
        "history": history,
        "ground_truth_linear": ground_truth_linear,
        "motion_lr": motion_lr,
        "exposure": torch.full(
            (batch, 1, 1, 1), exposure, device=device, dtype=torch.float32
        ),
        "jitter": jitter,
        "reset_event": torch.full(
            (batch, 1, 1, 1), reset_value, device=device, dtype=torch.float32
        ),
    }

    kpn_shape = (
        batch,
        model.autoencoder.kpn_ch,
        pad_shape[2] // 4,
        pad_shape[3] // 4,
    )
    kpn_params = 0.05 + 0.9 * _rand(kpn_shape, generator=generator, device=device)
    temporal_params = 0.05 + 0.9 * _rand(
        (batch, model.autoencoder.temporal_ch, pad_shape[2], pad_shape[3]),
        generator=generator,
        device=device,
    )

    derivative_shape = pad_shape if model.preprocess_half_res_input else input_shape
    nearest_shape = pad_shape if model.preprocess_half_res_input else process_shape
    nearest_depth_offset = _create_nearest_depth_offsets(model, nearest_shape, device)
    derivative = 0.05 + 0.9 * _rand(
        (batch, 4, derivative_shape[2], derivative_shape[3]),
        generator=generator,
        device=device,
    )
    disocclusion_mask = _rand(
        (batch, 2, process_shape[2], process_shape[3]),
        generator=generator,
        device=device,
    )

    return NSSV1PostprocessCase(
        model=model,
        inputs=inputs,
        kpn_params=kpn_params,
        temporal_params=temporal_params,
        nearest_depth_offset=nearest_depth_offset,
        derivative=derivative,
        disocclusion_mask=disocclusion_mask,
        hr_shape=hr_shape,
    )


def postprocess_gradient_probe(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Reduce post-process outputs with a spatially varying autograd probe."""

    output = outputs["output_linear"]
    filtered = outputs["out_filtered"]
    ramp = torch.linspace(
        0.25,
        1.25,
        output.numel(),
        device=output.device,
        dtype=output.dtype,
    ).reshape_as(output)
    return (output * ramp).mean() + (filtered * ramp.flip(-1)).mean()


def _create_core_forward_inputs(
    model: NSSV1Model,
    *,
    height: int = 8,
    width: int = 12,
) -> dict[str, torch.Tensor]:
    """Create one complete CPU frame for native preprocess/network/postprocess."""

    batch = 1
    output_height, output_width = model.get_output_spatial_shape(height, width)
    (
        _,
        _,
        _,
        pad_shape,
        _,
    ) = model._calculate_dispatch_dims(  # pylint: disable=protected-access
        {"colour_linear": torch.empty(batch, 3, height, width)}
    )
    derivative_spatial = (
        pad_shape[-2:] if model.preprocess_half_res_input else (height, width)
    )
    generator = torch.Generator().manual_seed(17)

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(shape, generator=generator, dtype=torch.float32)

    return {
        "colour_linear": rand(batch, 3, height, width) * 2.0,
        "history": rand(batch, 3, output_height, output_width),
        "ground_truth_linear": rand(batch, 3, output_height, output_width),
        "motion_lr": torch.zeros(batch, 2, height, width),
        "depth": torch.full((batch, 1, height, width), 0.5),
        "jitter": torch.tensor([0.125, -0.25]).reshape(batch, 2, 1, 1),
        "jitter_tm1": torch.tensor([-0.375, 0.25]).reshape(batch, 2, 1, 1),
        "temporal_params_tm1": rand(batch, 4, pad_shape[-2], pad_shape[-1]),
        "derivative_tm1": rand(batch, 4, *derivative_spatial),
        "depth_params": torch.tensor([0.0, 1.0, 1.0, 1.0]).reshape(batch, 4, 1, 1),
        "exposure": torch.full((batch, 1, 1, 1), 1.25),
        "render_size": torch.tensor([float(height), float(width)]).reshape(
            batch, 2, 1, 1
        ),
        "reset_event": torch.ones(batch, 1, 1, 1),
    }


class _StageFlags(NamedTuple):
    """Test-only convenience bundle for stage-function quality flags."""

    preprocess_half_res_input: bool
    use_sparse_filter_2x2: bool
    use_history_catmull: bool
    packed_nearest_offset_quad: bool
    sharp_theta: bool


def _decode_nearest_offsets(
    encoded: torch.Tensor,
    coordinates: torch.Tensor,
    input_size: tuple[int, int],
    settings: _StageFlags,
) -> torch.Tensor:
    return decode_nearest_offsets(
        encoded,
        coordinates,
        input_size,
        preprocess_half_res_input=settings.preprocess_half_res_input,
        packed_nearest_offset_quad=settings.packed_nearest_offset_quad,
    )


def _load_motion(
    motion: torch.Tensor,
    encoded: torch.Tensor,
    output_size: tuple[int, int],
    settings: _StageFlags,
) -> torch.Tensor:
    return load_motion(
        motion,
        encoded,
        output_size,
        preprocess_half_res_input=settings.preprocess_half_res_input,
        packed_nearest_offset_quad=settings.packed_nearest_offset_quad,
    )


def _kpn_coordinates(
    coordinates: torch.Tensor,
    input_size: tuple[int, int],
    kpn_size: tuple[int, int],
    temporal_size: tuple[int, int],
    settings: _StageFlags,
) -> torch.Tensor:
    return kpn_coordinates(
        coordinates,
        input_size,
        kpn_size,
        temporal_size,
        use_sparse_filter_2x2=settings.use_sparse_filter_2x2,
        preprocess_half_res_input=settings.preprocess_half_res_input,
    )


def _filter_color(
    color: torch.Tensor,
    kpn: torch.Tensor,
    lut: torch.Tensor,
    modulo: torch.Tensor,
    exposure: torch.Tensor,
    output_size: tuple[int, int],
    temporal_size: tuple[int, int],
    settings: _StageFlags,
):
    return filter_color(
        color,
        kpn,
        lut,
        modulo,
        exposure,
        output_size,
        temporal_size,
        preprocess_half_res_input=settings.preprocess_half_res_input,
        use_sparse_filter_2x2=settings.use_sparse_filter_2x2,
        filter_kernel_taps=lut.shape[-1],
    )


def _sample_temporal_params(
    temporal: torch.Tensor,
    output_size: tuple[int, int],
    preprocess_size: tuple[int, int] | torch.Tensor,
    settings: _StageFlags,
):
    preprocess = torch.as_tensor(
        preprocess_size, device=temporal.device, dtype=temporal.dtype
    )
    if preprocess.ndim == 1:
        preprocess = preprocess.reshape(1, 2).expand(temporal.shape[0], -1)
    return sample_temporal_params(
        temporal, output_size, preprocess, sharp_theta=settings.sharp_theta
    )


def _run_postprocess(case: NSSV1PostprocessCase) -> dict[str, torch.Tensor]:
    """Run one compact case through its configured model backend."""

    return case.model.postprocess(
        case.kpn_params,
        case.inputs,
        case.temporal_params,
        case.nearest_depth_offset,
        case.derivative,
        case.disocclusion_mask,
    )


def _assert_finite_hr_outputs(
    testcase: unittest.TestCase,
    outputs: dict[str, torch.Tensor],
    expected_shape: tuple[int, int, int, int],
) -> None:
    """Assert the two post-process products have finite HR values."""

    for key in ("output_linear", "out_filtered"):
        testcase.assertEqual(outputs[key].shape, expected_shape)
        testcase.assertEqual(outputs[key].dtype, torch.float32)
        testcase.assertEqual(outputs[key].device.type, "cpu")
        testcase.assertTrue(outputs[key].is_contiguous())
        testcase.assertTrue(torch.isfinite(outputs[key]).all().item())


def _set_constant_output_motion(
    case: NSSV1PostprocessCase,
    motion_yx: tuple[float, float],
) -> None:
    """Set LR motion so the shader observes a chosen output-pixel vector."""

    case.inputs["motion_lr"][:, 0].fill_(motion_yx[0] / case.model.scale)
    case.inputs["motion_lr"][:, 1].fill_(motion_yx[1] / case.model.scale)


def _set_high_frequency_history(case: NSSV1PostprocessCase) -> None:
    """Create an asymmetric pattern that exercises Catmull-Rom deringing."""

    height, width = case.inputs["history"].shape[-2:]
    rows = torch.arange(height, device=case.inputs["history"].device).view(-1, 1)
    columns = torch.arange(width, device=case.inputs["history"].device).view(1, -1)
    checkerboard = ((rows + columns) % 2).to(torch.float32)
    channels = torch.stack(
        (checkerboard, 0.25 + 0.5 * checkerboard, 1.0 - 0.75 * checkerboard)
    )
    case.inputs["history"] = (
        channels.unsqueeze(0)
        .expand(case.inputs["history"].shape[0], -1, -1, -1)
        .clone()
    )


def _set_corner_clamping_offsets(case: NSSV1PostprocessCase) -> None:
    """Put offsets that point beyond every LR border at encoded corners."""

    nearest = case.nearest_depth_offset
    if case.model.preprocess_half_res_input:
        logical_height = case.inputs["colour_linear"].shape[-2] // 2
        logical_width = case.inputs["colour_linear"].shape[-1] // 2
    else:
        logical_height, logical_width = case.inputs["colour_linear"].shape[-2:]
    corners = (
        (0, 0),
        (0, logical_width - 1),
        (logical_height - 1, 0),
        (logical_height - 1, logical_width - 1),
    )
    if case.model.packed_nearest_offset_quad:
        # Repeating a nibble in both packed bytes applies it to every lane.
        packed_codes = (0, 204, 51, 255)
        for code, (row, column) in zip(packed_codes, corners):
            nearest[:, :, row, column] = code / 255.0
        return

    # EncodeNearestDepthCoord values for y/x offsets (-2,-2), (-2,+2),
    # (+2,-2), and (+2,+2).
    for code, (row, column) in zip((0, 32, 4, 36), corners):
        nearest[:, 0, row, column] = code / 255.0


def _decode_high_offset_row_col(encoded: torch.Tensor) -> tuple[int, int]:
    """Independently decode one high-quality byte into tensor row/column."""

    code = int(torch.floor(encoded * 255.0 + 0.5).item())
    return (code & 0x7) - 2, ((code >> 3) & 0x7) - 2


def _decode_packed_nibble_row_col(packed_byte: torch.Tensor) -> tuple[int, int]:
    """Independently decode a repeated packed nibble into tensor row/column."""

    code = int(torch.floor(packed_byte * 255.0 + 0.5).item()) & 0xF
    return (code & 0x3) - 1, ((code >> 2) & 0x3) - 1


def _set_zero_nearest_offsets(case: NSSV1PostprocessCase) -> None:
    """Encode a zero row/column offset in every quality's texture layout."""

    code = 85 if case.model.packed_nearest_offset_quad else 18
    case.nearest_depth_offset.fill_(code / 255.0)


def _configure_temporal_contrast_case(case: NSSV1PostprocessCase) -> None:
    """Make history visibly distinct from filtered current color."""

    case.inputs["colour_linear"][:, 0].fill_(0.04)
    case.inputs["colour_linear"][:, 1].fill_(0.06)
    case.inputs["colour_linear"][:, 2].fill_(0.08)
    case.inputs["history"].fill_(0.75)
    case.inputs["motion_lr"].zero_()
    case.kpn_params.fill_(0.5)
    case.temporal_params[:, 0].fill_(0.9)
    case.temporal_params[:, 1].zero_()
    case.temporal_params[:, 2].fill_(1.0)
    case.temporal_params[:, 3].fill_(0.5)
    _set_zero_nearest_offsets(case)


def _configure_controlled_precise_case(case: NSSV1PostprocessCase) -> None:
    """Use exactly representable constants for strict live shader parity."""

    device = case.inputs["colour_linear"].device
    case.inputs["colour_linear"].copy_(
        torch.tensor((0.25, 0.5, 0.75), device=device).reshape(1, 3, 1, 1)
    )
    case.inputs["history"].copy_(
        torch.tensor((0.125, 0.375, 0.625), device=device).reshape(1, 3, 1, 1)
    )
    case.inputs["motion_lr"].zero_()
    case.inputs["exposure"].fill_(1.0)
    case.kpn_params.fill_(0.5)
    case.temporal_params.fill_(0.5)
    _set_zero_nearest_offsets(case)


def _configure_nearest_offset_probe(case: NSSV1PostprocessCase) -> None:
    """Create border-only motion and high-frequency history for offset probing."""

    _configure_temporal_contrast_case(case)
    height, width = case.inputs["history"].shape[-2:]
    rows = torch.arange(height, device=case.inputs["history"].device).view(-1, 1)
    columns = torch.arange(width, device=case.inputs["history"].device).view(1, -1)
    checkerboard = ((rows + columns) % 2).to(torch.float32)
    case.inputs["history"][:, 0] = 0.15 + 0.7 * checkerboard
    case.inputs["history"][:, 1] = 0.8 - 0.6 * checkerboard
    case.inputs["history"][:, 2] = 0.25 + 0.5 * checkerboard

    lr_height, lr_width = case.inputs["motion_lr"].shape[-2:]
    border_coords = (
        (0, 0),
        (0, lr_width - 1),
        (lr_height - 1, 0),
        (lr_height - 1, lr_width - 1),
    )
    output_motion = ((0.75, 0.5), (0.75, -0.5), (-0.75, 0.5), (-0.75, -0.5))
    for (row, column), motion_yx in zip(border_coords, output_motion):
        case.inputs["motion_lr"][:, 0, row, column] = motion_yx[0] / case.model.scale
        case.inputs["motion_lr"][:, 1, row, column] = motion_yx[1] / case.model.scale


def _set_behavior_sensitive_border_offsets(case: NSSV1PostprocessCase) -> None:
    """Move four near-corner LR samples beyond their nearest image borders."""

    _set_zero_nearest_offsets(case)
    lr_height, lr_width = case.inputs["colour_linear"].shape[-2:]
    if case.model.packed_nearest_offset_quad:
        logical_corners = (
            (0, 0),
            (0, lr_width // 2 - 1),
            (lr_height // 2 - 1, 0),
            (lr_height // 2 - 1, lr_width // 2 - 1),
        )
        for code, (row, column) in zip((0, 204, 51, 255), logical_corners):
            case.nearest_depth_offset[:, :, row, column] = code / 255.0
        return

    near_corners = (
        (1, 1),
        (1, lr_width - 2),
        (lr_height - 2, 1),
        (lr_height - 2, lr_width - 2),
    )
    for code, (row, column) in zip((0, 32, 4, 36), near_corners):
        case.nearest_depth_offset[:, 0, row, column] = code / 255.0


def _near_corner_output_pixels(
    case: NSSV1PostprocessCase,
) -> tuple[tuple[int, int], ...]:
    """Return output pixels whose motion lookup is controlled by border probes."""

    lr_height, lr_width = case.inputs["colour_linear"].shape[-2:]
    lr_pixels = (
        (1, 1),
        (1, lr_width - 2),
        (lr_height - 2, 1),
        (lr_height - 2, lr_width - 2),
    )
    return tuple(
        (int(row * case.model.scale), int(column * case.model.scale))
        for row, column in lr_pixels
    )
