# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

"""CPU and Slang-parity tests for NSS v1 Torch preprocessing."""

# pylint: disable=missing-function-docstring,too-many-lines

import unittest
from pathlib import Path
from unittest import mock

import torch

from ng_model_gym.core.model.shaders import slang_utils
from ng_model_gym.usecases.nss.model import torch_preprocess as torch_impl
from ng_model_gym.usecases.nss.model.model_v1 import NSSV1Model
from tests.usecases.nss.unit.nss_v1_test_utils import create_nss_v1_test_params

_SHADER_RTOL = 1e-5
_SHADER_ATOL = 5e-6
_PRECISE_SHADER_RTOL = 1e-6
_PRECISE_SHADER_ATOL = 1e-7
_PRECISE_VJP_RTOL = 1e-5
_PRECISE_VJP_ATOL = 1e-7
_GOLDEN_DIR = Path("tests/usecases/nss/unit/data/nss_v1_golden_values")
_RANDOM_PARITY_CASES = (
    {"seed": 0, "batch": 1, "height": 16, "width": 24},
    {"seed": 1, "batch": 2, "height": 15, "width": 21},
    {"seed": 2, "batch": 1, "height": 19, "width": 27},
)


def _create_model(
    quality: str,
    device: torch.device,
    *,
    backend: str = "torch",
    scale: float = 2.0,
    luma_derivative: bool = True,
) -> NSSV1Model:
    params = create_nss_v1_test_params(quality)
    params.model.processing_backend = backend
    params.model.scale = scale
    params.model.nss_v1_luma_derivative = luma_derivative
    model = NSSV1Model(params).to(device)
    model.eval()
    return model


def _create_inputs(
    model: NSSV1Model,
    *,
    batch: int = 1,
    height: int = 16,
    width: int = 24,
    device: torch.device = torch.device("cpu"),
    requires_grad: bool = False,
    constant_depth: bool = False,
    seed: int = 0,
    randomize: bool = False,
) -> dict[str, torch.Tensor]:
    """Create deterministic, non-degenerate one-frame preprocess inputs."""

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
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
    derivative_height, derivative_width = (
        pad_shape[-2:] if model.preprocess_half_res_input else (height, width)
    )

    colour = (
        torch.rand(batch, 3, height, width, generator=generator, device=device) * 4.0
    )
    history = torch.rand(
        batch,
        3,
        output_height,
        output_width,
        generator=generator,
        device=device,
    )
    feedback = torch.rand(
        batch,
        4,
        pad_shape[-2],
        pad_shape[-1],
        generator=generator,
        device=device,
    )
    derivative = torch.rand(
        batch,
        4,
        derivative_height,
        derivative_width,
        generator=generator,
        device=device,
    )
    if constant_depth:
        depth = torch.full((batch, 1, height, width), 0.5, device=device)
    else:
        depth = (
            torch.linspace(
                0.2,
                0.8,
                height * width,
                device=device,
                dtype=torch.float32,
            )
            .reshape(1, 1, height, width)
            .expand(batch, -1, -1, -1)
            .clone()
        )

    inputs = {
        "colour_linear": colour,
        "history": history,
        "motion_lr": torch.zeros(batch, 2, height, width, device=device),
        "depth": depth,
        "jitter": torch.tensor([0.125, -0.25], device=device)
        .reshape(1, 2, 1, 1)
        .expand(batch, -1, -1, -1)
        .clone(),
        "jitter_tm1": torch.tensor([-0.375, 0.25], device=device)
        .reshape(1, 2, 1, 1)
        .expand(batch, -1, -1, -1)
        .clone(),
        "temporal_params_tm1": feedback,
        "derivative_tm1": derivative,
        # Gives view_depth = 1 / depth and avoids poles in depth clipping.
        "depth_params": torch.tensor([0.0, 1.0, 1.0, 1.0], device=device)
        .reshape(1, 4, 1, 1)
        .expand(batch, -1, -1, -1)
        .clone(),
        "exposure": torch.full((batch, 1, 1, 1), 1.25, device=device),
        # NSS spatial vectors use (height, width) ordering.
        "render_size": torch.tensor([float(height), float(width)], device=device)
        .reshape(1, 2, 1, 1)
        .expand(batch, -1, -1, -1)
        .clone(),
    }
    if randomize:
        inputs["motion_lr"].uniform_(-1.0, 1.0, generator=generator)
        inputs["jitter"].uniform_(-0.5, 0.5, generator=generator)
        inputs["jitter_tm1"].uniform_(-0.5, 0.5, generator=generator)
        inputs["exposure"].uniform_(0.5, 2.0, generator=generator)
        if not constant_depth:
            inputs["depth"].uniform_(0.2, 0.8, generator=generator)
    if requires_grad:
        for key, value in inputs.items():
            if value.is_floating_point():
                inputs[key] = value.requires_grad_()
    return inputs


def _clone_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone().requires_grad_(value.requires_grad)
        for key, value in inputs.items()
    }


def _assert_normalized_bytes_equal(
    testcase: unittest.TestCase,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    actual_bytes = torch.round(actual.detach() * 255.0).to(torch.uint8)
    expected_bytes = torch.round(expected.detach() * 255.0).to(torch.uint8)
    testcase.assertTrue(
        torch.equal(actual_bytes, expected_bytes),
        "nearest-offset byte encodings differ",
    )


def _assert_preprocess_parity(testcase, torch_model, slang_model, inputs, **tolerance):
    torch_outputs = torch_model.preprocess(inputs)
    slang_outputs = slang_model.preprocess(_clone_inputs(inputs))
    for output_index, outputs in enumerate(zip(torch_outputs, slang_outputs)):
        with testcase.subTest(output_index=output_index):
            torch.testing.assert_close(*outputs, **tolerance)
    _assert_normalized_bytes_equal(testcase, torch_outputs[3], slang_outputs[3])


class TestNSSV1TorchPreprocessStages(unittest.TestCase):
    """Literal-oracle tests for shader-sensitive Torch helper semantics."""

    def test_explicit_bilinear_sampling_at_border(self) -> None:
        source = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        uv = torch.tensor([[[[0.0, 0.0], [0.5, 0.5]]]])

        clamped = torch_impl.sampling.bilinear_sample(source, uv, clamp_to_edge=True)
        zero_padded = torch_impl.sampling.bilinear_sample(
            source, uv, clamp_to_edge=False
        )

        torch.testing.assert_close(
            clamped,
            torch.tensor([[[[1.0, 2.5]]]]),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            zero_padded,
            torch.tensor([[[[0.25, 2.5]]]]),
            rtol=0.0,
            atol=0.0,
        )

    def test_bilinear_sampler_source_gradcheck(self) -> None:
        source = (
            torch.linspace(-1.0, 2.0, 12, dtype=torch.float64)
            .reshape(1, 2, 2, 3)
            .requires_grad_()
        )
        # Keep all samples away from borders and pixel-centre discontinuities.
        uv = torch.tensor([[[[0.37, 0.29], [0.61, 0.72]]]], dtype=torch.float64)

        self.assertTrue(
            torch.autograd.gradcheck(
                lambda value: torch_impl.sampling.bilinear_sample(value, uv),
                (source,),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
            )
        )

    def test_equal_depth_ordering_and_byte_encodings(self) -> None:
        depth = torch.full((1, 1, 8, 8), 0.5)
        pixel = torch.tensor([[[[3, 3]]]])
        uv = (pixel.to(torch.float32) + 0.5) / torch.tensor([8.0, 8.0])

        high_offset = torch_impl.sampling.find_nearest_depth_4x4(depth, uv)[2]
        low_offset = torch_impl.sampling.find_nearest_depth_4x4_from_pixels(
            depth, pixel
        )[2]
        self.assertTrue(torch.equal(high_offset, torch.tensor([[[[0, 0]]]])))
        self.assertTrue(torch.equal(low_offset, torch.tensor([[[[2, 2]]]])))

        high_byte = torch.round(
            torch_impl.sampling.encode_nearest_offsets(high_offset) * 255.0
        )
        self.assertTrue(torch.equal(high_byte, torch.tensor([[[[18.0]]]])))
        packed = torch_impl.sampling.pack_nearest_offsets(
            low_offset, low_offset, low_offset, low_offset
        )
        packed_bytes = torch.round(packed * 255.0)
        self.assertTrue(
            torch.equal(packed_bytes, torch.tensor([[[[255.0]], [[255.0]]]]))
        )

    def test_float_to_int32_is_truncating_and_saturating(self) -> None:
        values = torch.tensor(
            [-(2.0**40), -1.9, 0.0, 1.9, 2.0**40], dtype=torch.float64
        )
        expected = torch.tensor(
            [
                torch.iinfo(torch.int32).min,
                -1,
                0,
                1,
                torch.iinfo(torch.int32).max,
            ],
            dtype=torch.int32,
        )
        self.assertTrue(
            torch.equal(torch_impl.depth.saturating_float_to_int32(values), expected)
        )

        # This is the rounded float32 product produced for shader depth 1.0.
        shader_one = torch.tensor(1.0, dtype=torch.float32) * torch.tensor(
            float(torch.iinfo(torch.int32).max), dtype=torch.float32
        )
        self.assertEqual(
            torch_impl.depth.saturating_float_to_int32(shader_one).item(),
            torch.iinfo(torch.int32).max,
        )

    def test_depth_scatter_depth_one_and_int32_max_holes(self) -> None:
        int32_max = torch.iinfo(torch.int32).max
        for quarter_res_input, input_size in ((False, 4), (True, 8)):
            with self.subTest(quarter_res_input=quarter_res_input):
                motion = torch.zeros(1, 2, input_size, input_size)
                render_size = torch.tensor(
                    [float(input_size), float(input_size)]
                ).reshape(1, 2, 1, 1)
                depth_one = torch.ones(1, 1, input_size, input_size)
                reconstructed_one = torch_impl.depth.depth_scatter(
                    motion,
                    depth_one,
                    render_size,
                    (2, 2),
                    quarter_res_input=quarter_res_input,
                )
                self.assertEqual(reconstructed_one.dtype, torch.int32)
                self.assertTrue(
                    torch.equal(
                        reconstructed_one,
                        torch.full_like(reconstructed_one, int32_max),
                    )
                )

                depth_half = torch.full_like(depth_one, 0.5)
                reconstructed_half = torch_impl.depth.depth_scatter(
                    motion,
                    depth_half,
                    render_size,
                    (2, 2),
                    quarter_res_input=quarter_res_input,
                )
                self.assertTrue(
                    torch.equal(
                        reconstructed_half,
                        torch.full_like(reconstructed_half, 1 << 30),
                    )
                )

                offscreen_motion = torch.full_like(motion, 100.0)
                holes = torch_impl.depth.depth_scatter(
                    offscreen_motion,
                    depth_half,
                    render_size,
                    (2, 2),
                    quarter_res_input=quarter_res_input,
                )
                self.assertTrue(torch.equal(holes, torch.full_like(holes, int32_max)))
                self.assertFalse(holes.requires_grad)

    def test_reconstruction_bilinear_weight_threshold_is_strict(self) -> None:
        depth = torch.tensor([[[0.5]]], dtype=torch.float32)
        center_row = torch.tensor(0.5, dtype=torch.float32)
        threshold_uv = torch.tensor(0.7, dtype=torch.float32)
        below_uv = torch.nextafter(
            threshold_uv, torch.tensor(float("-inf"), dtype=torch.float32)
        )
        above_uv = torch.nextafter(
            threshold_uv, torch.tensor(float("inf"), dtype=torch.float32)
        )

        def reconstruct(column_uv: torch.Tensor) -> torch.Tensor:
            uv = torch.stack((center_row, column_uv)).reshape(1, 1, 1, 2)
            return torch_impl.depth._reconstruct_previous_depth(  # pylint: disable=protected-access
                depth, uv, (1, 2)
            )

        int_depth = 1 << 30
        self.assertTrue(
            torch.equal(
                reconstruct(below_uv),
                torch.tensor([[[[int_depth, int_depth]]]], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                reconstruct(above_uv),
                torch.tensor(
                    [[[[torch.iinfo(torch.int32).max, int_depth]]]],
                    dtype=torch.int32,
                ),
            )
        )

    def test_rec709_luminance_disocclusion_threshold_is_strict(self) -> None:
        color = torch.ones(1, 3, 1, 3)
        derivative_tm1 = torch.zeros(1, 4, 1, 3)
        derivative_uv = torch.tensor(
            [[[[0.5, 1.0 / 6.0], [0.5, 0.5], [0.5, 5.0 / 6.0]]]]
        )
        threshold = torch.tensor(0.01, dtype=torch.float32)
        disocclusion = torch.stack(
            (
                torch.nextafter(threshold, torch.tensor(float("-inf"))),
                threshold,
                torch.nextafter(threshold, torch.tensor(float("inf"))),
            )
        ).reshape(1, 1, 1, 3)

        _, visible = torch_impl.derivative.calculate_rec709_luminance_derivative(
            color, derivative_tm1, derivative_uv, disocclusion
        )

        self.assertGreater(visible[0, 0, 0, 0].item(), 0.0)
        self.assertGreater(visible[0, 0, 0, 1].item(), 0.0)
        self.assertEqual(visible[0, 0, 0, 2].item(), 0.0)


class TestNSSV1TorchPreprocessCPU(unittest.TestCase):
    """CPU-only contract tests for the public Torch preprocessing route."""

    def test_torch_route_does_not_touch_cuda_or_slang(self) -> None:
        model = _create_model("high", torch.device("cpu"))
        inputs = _create_inputs(model, height=9, width=13)

        with (
            mock.patch.object(
                model,
                "_require_cuda_for_slang_forward",
                side_effect=AssertionError("Torch preprocessing used CUDA guard"),
            ),
            mock.patch.object(
                model,
                "_get_slang",
                side_effect=AssertionError("Torch preprocessing loaded Slang"),
            ),
            mock.patch(
                "ng_model_gym.usecases.nss.model.model_v1.load_slang_module",
                side_effect=AssertionError("Torch preprocessing called Slang loader"),
            ),
        ):
            outputs = model.preprocess(inputs)

        self.assertEqual(len(outputs), 4)
        for output in outputs:
            self.assertEqual(output.device.type, "cpu")

    def test_shapes_dtypes_finiteness_and_zero_fringes(self) -> None:
        cases = (
            ("high", 2, 15, 21, True, 1.5),
            ("mid", 1, 19, 27, True, 2.0),
            ("low", 2, 20, 26, False, 2.0),
        )
        for quality, batch, height, width, luma_derivative, scale in cases:
            with self.subTest(quality=quality, luma_derivative=luma_derivative):
                model = _create_model(
                    quality,
                    torch.device("cpu"),
                    scale=scale,
                    luma_derivative=luma_derivative,
                )
                inputs = _create_inputs(model, batch=batch, height=height, width=width)
                (
                    input_shape,
                    process_shape,
                    _,
                    pad_shape,
                    _,
                ) = model._calculate_dispatch_dims(  # pylint: disable=protected-access
                    inputs
                )
                outputs = model.preprocess(inputs)
                input_tensor, derivative, disocclusion, nearest_offset = outputs

                derivative_spatial = (
                    pad_shape[-2:]
                    if model.preprocess_half_res_input
                    else input_shape[-2:]
                )
                nearest_spatial = (
                    pad_shape[-2:]
                    if model.preprocess_half_res_input
                    else process_shape[-2:]
                )
                self.assertEqual(
                    tuple(input_tensor.shape),
                    (batch, 12, pad_shape[-2], pad_shape[-1]),
                )
                self.assertEqual(
                    tuple(derivative.shape), (batch, 4, *derivative_spatial)
                )
                self.assertEqual(
                    tuple(disocclusion.shape),
                    (batch, 2, process_shape[-2], process_shape[-1]),
                )
                self.assertEqual(
                    tuple(nearest_offset.shape),
                    (
                        batch,
                        model._nearest_depth_offset_channels(),  # pylint: disable=protected-access
                        *nearest_spatial,
                    ),
                )
                for output in outputs:
                    self.assertEqual(output.dtype, torch.float32)
                    self.assertEqual(output.device.type, "cpu")
                    self.assertTrue(output.is_contiguous())
                    self.assertTrue(torch.isfinite(output).all().item())
                self.assertEqual(torch.count_nonzero(disocclusion[:, 1]).item(), 0)

                if model.preprocess_half_res_input:
                    process_height, process_width = process_shape[-2:]
                    self.assertEqual(
                        torch.count_nonzero(
                            derivative[:, :, process_height:, :]
                        ).item(),
                        0,
                    )
                    self.assertEqual(
                        torch.count_nonzero(derivative[:, :, :, process_width:]).item(),
                        0,
                    )
                    self.assertEqual(
                        torch.count_nonzero(
                            nearest_offset[:, :, process_height:, :]
                        ).item(),
                        0,
                    )
                    self.assertEqual(
                        torch.count_nonzero(
                            nearest_offset[:, :, :, process_width:]
                        ).item(),
                        0,
                    )

    def test_all_quality_and_derivative_routes_run_on_cpu(self) -> None:
        for quality in ("high", "mid", "low"):
            for luma_derivative in (True, False):
                with self.subTest(quality=quality, luma_derivative=luma_derivative):
                    model = _create_model(
                        quality,
                        torch.device("cpu"),
                        luma_derivative=luma_derivative,
                    )
                    inputs = _create_inputs(model, height=8, width=16)
                    outputs = model.preprocess(inputs)
                    self.assertTrue(
                        all(output.device.type == "cpu" for output in outputs)
                    )

    def test_equal_depth_ties_and_offset_encoding(self) -> None:
        high_model = _create_model("high", torch.device("cpu"))
        high_inputs = _create_inputs(
            high_model, height=12, width=16, constant_depth=True
        )
        high_offset = high_model.preprocess(high_inputs)[3]
        # High quality uses strict '<': equal depth keeps offset (0, 0).
        expected_high_byte = ((0 + 2) << 3) | (0 + 2)
        high_bytes = torch.round(high_offset * 255.0).to(torch.uint8)
        self.assertTrue(
            torch.equal(
                high_bytes,
                torch.full_like(high_bytes, expected_high_byte),
            )
        )

        for quality in ("mid", "low"):
            with self.subTest(quality=quality):
                model = _create_model(quality, torch.device("cpu"))
                inputs = _create_inputs(model, height=20, width=24, constant_depth=True)
                offset = model.preprocess(inputs)[3]
                offset_bytes = torch.round(offset * 255.0).to(torch.uint8)
                # The interior 4x4 search uses '<=' and the last equal-depth
                # candidate is (+2, +2). Its nibble is 0xf, so two packed
                # nibbles make 0xff in both output channels.
                self.assertTrue(
                    torch.equal(
                        offset_bytes[:, :, 2:6, 2:8],
                        torch.full_like(offset_bytes[:, :, 2:6, 2:8], 255),
                    )
                )

    def test_motion_threshold_is_strict(self) -> None:
        below = torch.nextafter(torch.tensor(0.1), torch.tensor(float("-inf"))).item()
        above = torch.nextafter(torch.tensor(0.1), torch.tensor(float("inf"))).item()
        motion = torch.tensor(
            [[below, 0.0], [0.1, 0.0], [above, 0.0]], dtype=torch.float32
        ).reshape(3, 2, 1, 1)

        thresholded = (
            torch_impl.pipeline._threshold_motion(  # pylint: disable=protected-access
                motion
            )
        )

        self.assertTrue(torch.equal(thresholded[0], torch.zeros_like(thresholded[0])))
        self.assertTrue(torch.equal(thresholded[1], torch.zeros_like(thresholded[1])))
        torch.testing.assert_close(thresholded[2], motion[2], rtol=0.0, atol=0.0)

    def test_preprocess_does_not_mutate_inputs(self) -> None:
        model = _create_model("mid", torch.device("cpu"))
        inputs = _create_inputs(model, height=20, width=26)
        snapshots = {key: value.detach().clone() for key, value in inputs.items()}

        model.preprocess(inputs)

        self.assertEqual(inputs.keys(), snapshots.keys())
        for key, expected in snapshots.items():
            with self.subTest(key=key):
                torch.testing.assert_close(inputs[key], expected, rtol=0.0, atol=0.0)

    def test_only_history_and_feedback_receive_gradients(self) -> None:
        model = _create_model("high", torch.device("cpu"))
        inputs = _create_inputs(
            model,
            height=8,
            width=12,
            requires_grad=True,
            constant_depth=True,
        )
        input_tensor, derivative, disocclusion, nearest_offset = model.preprocess(
            inputs
        )
        history_weights = torch.linspace(
            0.25,
            1.25,
            input_tensor[:, 0:3].numel(),
            device=input_tensor.device,
        ).reshape_as(input_tensor[:, 0:3])
        feedback_weights = torch.linspace(
            -0.75,
            0.5,
            input_tensor[:, 7:11].numel(),
            device=input_tensor.device,
        ).reshape_as(input_tensor[:, 7:11])
        loss = (input_tensor[:, 0:3] * history_weights).sum() + (
            input_tensor[:, 7:11] * feedback_weights
        ).sum()
        keys = tuple(inputs)
        grads = torch.autograd.grad(
            loss,
            tuple(inputs[key] for key in keys),
            allow_unused=True,
        )
        grads_by_key = dict(zip(keys, grads))

        for key in ("history", "temporal_params_tm1"):
            grad = grads_by_key[key]
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all().item())
            self.assertGreater(torch.count_nonzero(grad).item(), 0)
        for key in set(keys) - {"history", "temporal_params_tm1"}:
            grad = grads_by_key[key]
            if grad is not None:
                self.assertEqual(
                    torch.count_nonzero(grad).item(),
                    0,
                    f"unexpected gradient for {key}",
                )

        for auxiliary in (derivative, disocclusion, nearest_offset):
            self.assertFalse(auxiliary.requires_grad)

    def test_one_model_handles_changed_batch_and_spatial_shape(self) -> None:
        model = _create_model("mid", torch.device("cpu"))
        for batch, height, width in ((1, 16, 24), (2, 20, 26), (1, 24, 16)):
            with self.subTest(batch=batch, height=height, width=width):
                inputs = _create_inputs(model, batch=batch, height=height, width=width)
                expected_pad = (
                    model._calculate_dispatch_dims(  # pylint: disable=protected-access
                        inputs
                    )[3]
                )
                input_tensor = model.preprocess(inputs)[0]
                self.assertEqual(
                    tuple(input_tensor.shape),
                    (batch, 12, expected_pad[-2], expected_pad[-1]),
                )


class TestNSSV1TorchPreprocessGolden(unittest.TestCase):
    """Compare CPU Torch preprocessing with hydrated Slang golden tensors."""

    def test_cpu_matches_slang_goldens(self) -> None:
        for quality in ("high", "mid"):
            with self.subTest(quality=quality):
                input_path = _GOLDEN_DIR / (
                    f"nss_v1_{quality}_preprocess_inputs_golden.pt"
                )
                output_path = _GOLDEN_DIR / (
                    f"nss_v1_{quality}_preprocess_output_golden.pt"
                )
                if (
                    not input_path.exists()
                    or not output_path.exists()
                    or input_path.stat().st_size < 1024
                    or output_path.stat().st_size < 1024
                ):
                    self.skipTest("NSS v1 Git LFS golden tensors are not hydrated")

                inputs = torch.load(input_path, map_location="cpu", weights_only=True)
                expected = torch.load(
                    output_path, map_location="cpu", weights_only=True
                )
                model = _create_model(quality, torch.device("cpu"))
                actual = model.preprocess(inputs)

                for tensor, key in zip(
                    actual,
                    (
                        "input_tensor",
                        "derivative",
                        "disocclusion_mask",
                        "nearest_depth_offset",
                    ),
                ):
                    with self.subTest(output=key):
                        torch.testing.assert_close(
                            tensor,
                            expected[key],
                            rtol=_SHADER_RTOL,
                            atol=_SHADER_ATOL,
                        )
                _assert_normalized_bytes_equal(
                    self, actual[3], expected["nearest_depth_offset"]
                )


@unittest.skipUnless(
    torch.cuda.is_available(),
    "CUDA is required for live NSS v1 Slang/Torch parity tests.",
)
class TestNSSV1TorchPreprocessSlangParity(unittest.TestCase):
    """Live same-input comparisons against the authoritative Slang kernels."""

    @classmethod
    def setUpClass(cls) -> None:
        """Compile only parity-test Slang modules without CUDA fast math."""

        super().setUpClass()
        # pylint: disable-next=protected-access
        slang_utils._load_slang_module_cached.cache_clear()
        load_module = slang_utils.slangtorch.loadModule

        def load_module_without_fast_math(*args, **kwargs):
            kwargs["cudaFastMath"] = False
            return load_module(*args, **kwargs)

        cls._slang_load_patcher = mock.patch.object(
            slang_utils.slangtorch,
            "loadModule",
            side_effect=load_module_without_fast_math,
        )
        cls._slang_load_patcher.start()

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove precise-math modules from the process-local module cache."""

        cls._slang_load_patcher.stop()
        # pylint: disable-next=protected-access
        slang_utils._load_slang_module_cached.cache_clear()
        super().tearDownClass()

    def test_torch_cpu_cuda_parity(self) -> None:
        cpu_model = _create_model(
            "high", torch.device("cpu"), backend="torch", scale=1.5
        )
        cuda_model = _create_model(
            "high", torch.device("cuda"), backend="torch", scale=1.5
        )
        cpu_inputs = _create_inputs(
            cpu_model, batch=2, height=15, width=21, constant_depth=True
        )
        cuda_inputs = {key: value.to("cuda") for key, value in cpu_inputs.items()}

        cpu_outputs = cpu_model.preprocess(cpu_inputs)
        cuda_outputs = cuda_model.preprocess(cuda_inputs)
        for cpu_output, cuda_output in zip(cpu_outputs, cuda_outputs):
            torch.testing.assert_close(
                cpu_output,
                cuda_output.cpu(),
                rtol=_SHADER_RTOL,
                atol=_SHADER_ATOL,
            )
        _assert_normalized_bytes_equal(self, cpu_outputs[3], cuda_outputs[3].cpu())

    def test_forward_parity_all_quality_and_derivative_routes(self) -> None:
        device = torch.device("cuda")
        for quality in ("high", "mid", "low"):
            for luma_derivative in (True, False):
                torch_model = _create_model(
                    quality,
                    device,
                    backend="torch",
                    luma_derivative=luma_derivative,
                )
                slang_model = _create_model(
                    quality,
                    device,
                    backend="slang",
                    luma_derivative=luma_derivative,
                )
                with self.subTest(
                    quality=quality,
                    luma_derivative=luma_derivative,
                    inputs="controlled",
                ):
                    _assert_preprocess_parity(
                        self,
                        torch_model,
                        slang_model,
                        _create_inputs(torch_model, device=device),
                        rtol=_PRECISE_SHADER_RTOL,
                        atol=_PRECISE_SHADER_ATOL,
                    )

                for case in _RANDOM_PARITY_CASES:
                    with self.subTest(
                        quality=quality,
                        luma_derivative=luma_derivative,
                        inputs="random",
                        **case,
                    ):
                        _assert_preprocess_parity(
                            self,
                            torch_model,
                            slang_model,
                            _create_inputs(
                                torch_model,
                                **case,
                                device=device,
                                randomize=True,
                            ),
                            rtol=_SHADER_RTOL,
                            atol=_SHADER_ATOL,
                        )

    def test_history_and_feedback_vjp_parity(self) -> None:
        device = torch.device("cuda")
        for quality in ("high", "mid", "low"):
            for luma_derivative in (True, False):
                with self.subTest(quality=quality, luma_derivative=luma_derivative):
                    torch_model = _create_model(
                        quality,
                        device,
                        backend="torch",
                        luma_derivative=luma_derivative,
                    )
                    slang_model = _create_model(
                        quality,
                        device,
                        backend="slang",
                        luma_derivative=luma_derivative,
                    )
                    base_inputs = _create_inputs(
                        torch_model,
                        height=8,
                        width=16,
                        device=device,
                        constant_depth=True,
                    )
                    base_inputs["motion_lr"][:, 0].fill_(0.25)
                    torch_inputs = _clone_inputs(base_inputs)
                    slang_inputs = _clone_inputs(base_inputs)
                    for inputs in (torch_inputs, slang_inputs):
                        inputs["history"].requires_grad_()
                        inputs["temporal_params_tm1"].requires_grad_()

                    torch_output = torch_model.preprocess(torch_inputs)[0]
                    slang_output = slang_model.preprocess(slang_inputs)[0]
                    generator = torch.Generator(device=device)
                    generator.manual_seed(1776)
                    cotangent = torch.rand(
                        torch_output.shape,
                        generator=generator,
                        device=device,
                    )
                    # A full cotangent also checks that non-differentiable
                    # output channels do not leak into either source VJP.
                    torch_grads = torch.autograd.grad(
                        torch_output,
                        (
                            torch_inputs["history"],
                            torch_inputs["temporal_params_tm1"],
                        ),
                        cotangent,
                    )
                    slang_grads = torch.autograd.grad(
                        slang_output,
                        (
                            slang_inputs["history"],
                            slang_inputs["temporal_params_tm1"],
                        ),
                        cotangent,
                    )
                    for torch_grad, slang_grad in zip(torch_grads, slang_grads):
                        self.assertTrue(torch.isfinite(torch_grad).all().item())
                        self.assertTrue(torch.isfinite(slang_grad).all().item())
                        torch.testing.assert_close(
                            torch_grad,
                            slang_grad,
                            rtol=_PRECISE_VJP_RTOL,
                            atol=_PRECISE_VJP_ATOL,
                        )

    def test_two_frame_recurrent_training_parity(self) -> None:
        device = torch.device("cuda")
        # Module initialization uses PyTorch's global CPU generator. Isolate and
        # seed it so recurrent gradient tolerances do not depend on test order or
        # the process's random initial seed.
        model_generator = torch.Generator(device="cpu").manual_seed(1234)
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(model_generator.get_state())
            torch_model = _create_model("high", device, backend="torch")
            slang_model = _create_model("high", device, backend="slang")
        slang_model.autoencoder.load_state_dict(torch_model.autoencoder.state_dict())

        frame_inputs = []
        for frame_index in range(2):
            frame = _create_inputs(
                torch_model,
                height=16,
                width=24,
                device=device,
                constant_depth=True,
            )
            frame["colour_linear"] = frame["colour_linear"] + frame_index * 0.125
            frame["motion_lr"][:, 0].fill_(0.25 + frame_index * 0.125)
            output_height, output_width = torch_model.get_output_spatial_shape(16, 24)
            frame["ground_truth_linear"] = torch.full(
                (1, 3, output_height, output_width),
                0.4 + frame_index * 0.1,
                device=device,
            )
            frame["motion"] = torch.zeros(
                1, 2, output_height, output_width, device=device
            )
            frame["seq"] = torch.ones(1, 1, 1, 1, device=device)
            frame_inputs.append(frame)

        def run_two_frames(model: NSSV1Model):
            model.reset_history_buffers()
            records = []
            for raw_frame in frame_inputs:
                inputs = model.set_buffers(_clone_inputs(raw_frame))
                outputs = model.core_forward(inputs)
                model.update_buffers(inputs, outputs)
                records.append((inputs, outputs))
            return records

        torch_records = run_two_frames(torch_model)
        slang_records = run_two_frames(slang_model)

        for state_key in ("history", "temporal_params_tm1"):
            torch.testing.assert_close(
                torch_records[1][0][state_key],
                slang_records[1][0][state_key],
                rtol=2e-4,
                atol=1e-5,
            )
        for frame_index in range(2):
            for output_key in (
                "output",
                "output_linear",
                "out_filtered",
                "temporal_params",
                "derivative",
                "disocclusion_mask",
            ):
                with self.subTest(frame=frame_index, output=output_key):
                    torch.testing.assert_close(
                        torch_records[frame_index][1][output_key],
                        slang_records[frame_index][1][output_key],
                        rtol=2e-4,
                        atol=1e-5,
                    )

        def recurrent_loss(records) -> torch.Tensor:
            loss = torch.zeros((), device=device)
            for frame_index, (_, outputs) in enumerate(records):
                frame_weight = float(frame_index + 1)
                loss = loss + frame_weight * (
                    outputs["output_linear"].square().mean()
                    + 0.5 * outputs["temporal_params"].square().mean()
                )
            return loss

        torch_loss = recurrent_loss(torch_records)
        slang_loss = recurrent_loss(slang_records)
        torch.testing.assert_close(torch_loss, slang_loss, rtol=2e-4, atol=1e-6)

        def gradient_targets(model: NSSV1Model, records):
            return (
                ("history", records[1][0]["history"]),
                ("temporal_params_tm1", records[1][0]["temporal_params_tm1"]),
                ("conv2d_0.weight", model.autoencoder.conv2d_0.conv2d.weight),
                ("kpn_params.weight", model.autoencoder.kpn_params.conv2d.weight),
                (
                    "temporal_params_out_conv.weight",
                    model.autoencoder.temporal_params_out_conv.conv2d.weight,
                ),
            )

        torch_targets = dict(gradient_targets(torch_model, torch_records))
        slang_targets = dict(gradient_targets(slang_model, slang_records))
        torch_grads = torch.autograd.grad(torch_loss, tuple(torch_targets.values()))
        slang_grads = torch.autograd.grad(slang_loss, tuple(slang_targets.values()))
        for gradient_name, torch_grad, slang_grad in zip(
            torch_targets, torch_grads, slang_grads
        ):
            with self.subTest(gradient=gradient_name):
                self.assertTrue(torch.isfinite(torch_grad).all().item())
                self.assertTrue(torch.isfinite(slang_grad).all().item())
                self.assertGreater(torch.count_nonzero(torch_grad).item(), 0)
                self.assertGreater(torch.count_nonzero(slang_grad).item(), 0)
                torch.testing.assert_close(
                    torch_grad,
                    slang_grad,
                    rtol=1e-4,
                    # Recurrent accumulation receives slightly different upstream
                    # round-off even though the isolated VJP uses tighter tolerances.
                    atol=1.5e-5,
                )
