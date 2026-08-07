# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import math
import unittest
from unittest import mock

import torch

from ng_model_gym.core.data.data_utils import tonemap_forward
from ng_model_gym.usecases.nss.model.model_v1 import NSSV1Model
from ng_model_gym.usecases.nss.model.quality_modes import NSSV1Quality
from ng_model_gym.usecases.nss.model.torch_postprocess.filter import kpn_coordinates
from ng_model_gym.usecases.nss.model.torch_postprocess.pipeline import (
    karis_forward as _karis_forward,
)
from ng_model_gym.usecases.nss.model.torch_postprocess.pipeline import (
    karis_inverse as _karis_inverse,
)
from ng_model_gym.usecases.nss.model.torch_postprocess.pipeline import postprocess_torch
from ng_model_gym.usecases.nss.model.torch_postprocess.pipeline import (
    slang_backward_identity as _slang_backward_identity,
)
from ng_model_gym.usecases.nss.model.torch_postprocess.sampling import EPS as _EPS
from tests.usecases.nss.unit.nss_v1_test_utils import (
    create_nss_v1_test_params,
    load_nss_v1_golden,
    NSS_V1_CORE_OUTPUT_KEYS,
)
from tests.usecases.nss.unit.nss_v1_torch_postprocess_test_utils import (
    _assert_finite_hr_outputs,
    _configure_nearest_offset_probe,
    _configure_temporal_contrast_case,
    _create_core_forward_inputs,
    _filter_color,
    _near_corner_output_pixels,
    _QUALITIES,
    _run_postprocess,
    _set_behavior_sensitive_border_offsets,
    _set_constant_output_motion,
    _SHADER_ATOL,
    _SHADER_RTOL,
    _StageFlags,
    create_nss_v1_postprocess_case,
    NSSV1PostprocessCase,
    postprocess_gradient_probe,
)


def post_process(**arguments):
    """Adapt test dictionaries to the explicit production entrypoint."""

    arguments = dict(arguments)
    settings = arguments.pop("settings")
    arguments.setdefault("filter_kernel_taps", arguments["in_offset_lut"].shape[-1])
    # pylint: disable-next=missing-kwoa
    return postprocess_torch(
        **arguments,
        preprocess_half_res_input=settings.preprocess_half_res_input,
        use_sparse_filter_2x2=settings.use_sparse_filter_2x2,
        use_history_catmull=settings.use_history_catmull,
        packed_nearest_offset_quad=settings.packed_nearest_offset_quad,
        sharp_theta=settings.sharp_theta,
    )


# Account for fast-math cancellation in Slang-generated goldens.
_SLANG_GOLDEN_CANCELLATION_ATOL = 2.5e-4
_SLANG_GOLDEN_CANCELLATION_THRESHOLD = 6.0e-5


def _slang_golden_cancellation_mask(
    case: NSSV1PostprocessCase,
) -> torch.Tensor:
    """Locate fast-math cancellation pixels in Slang goldens."""

    (
        input_shape,
        _,
        hr_shape,
        _,
        _,
    ) = case.model._calculate_dispatch_dims(  # pylint: disable=protected-access
        case.inputs
    )
    (
        offset_lut,
        idx_modulo,
    ) = case.model._generate_offset_lut(  # pylint: disable=protected-access
        case.inputs["jitter"], input_shape, hr_shape
    )
    filtered = _filter_color(
        case.inputs["colour_linear"],
        case.kpn_params,
        offset_lut,
        idx_modulo,
        case.inputs["exposure"],
        hr_shape[-2:],
        tuple(case.temporal_params.shape[-2:]),
        _native_settings(case),
    )
    return (
        torch.abs(filtered.m2 - filtered.m1 * filtered.m1)
        <= _SLANG_GOLDEN_CANCELLATION_THRESHOLD
    )


def _assert_output_linear_matches_golden(
    actual: torch.Tensor,
    expected: torch.Tensor,
    quality: str,
    case: NSSV1PostprocessCase,
) -> None:
    """Compare Slang goldens with a computed cancellation-only mask."""

    if quality == NSSV1Quality.HIGH.value:
        torch.testing.assert_close(
            actual, expected, rtol=_SHADER_RTOL, atol=_SHADER_ATOL
        )
        return

    slang_golden_mask = _slang_golden_cancellation_mask(case)
    torch.testing.assert_close(
        actual[~slang_golden_mask],
        expected[~slang_golden_mask],
        rtol=_SHADER_RTOL,
        atol=_SHADER_ATOL,
    )
    torch.testing.assert_close(
        actual[slang_golden_mask],
        expected[slang_golden_mask],
        rtol=_SHADER_RTOL,
        atol=_SLANG_GOLDEN_CANCELLATION_ATOL,
    )


def _make_golden_postprocess_case(
    quality: str,
) -> tuple[NSSV1PostprocessCase, dict[str, torch.Tensor]]:
    """Adapt a Slang post-process golden to the public model API."""

    device = torch.device("cpu")
    golden_input = load_nss_v1_golden(
        f"nss_v1_{quality}_postprocess_inputs_golden.pt", device
    )
    golden_output = load_nss_v1_golden(
        f"nss_v1_{quality}_postprocess_output_golden.pt", device
    )
    params = create_nss_v1_test_params(quality)
    params.model.processing_backend = "torch"
    model = NSSV1Model(params).eval()
    golden_input["ground_truth_linear"] = torch.zeros_like(golden_input["history"])
    golden_input["reset_event"] = golden_input["reset"]

    (
        input_shape,
        process_shape,
        hr_shape,
        pad_shape,
        _,
    ) = model._calculate_dispatch_dims(  # pylint: disable=protected-access
        golden_input
    )
    derivative_shape = pad_shape if model.preprocess_half_res_input else input_shape
    derivative = torch.zeros(
        input_shape[0], 4, derivative_shape[2], derivative_shape[3]
    )
    disocclusion_mask = torch.zeros(
        input_shape[0], 2, process_shape[2], process_shape[3]
    )
    case = NSSV1PostprocessCase(
        model=model,
        inputs=golden_input,
        kpn_params=golden_input["kpn_params"],
        temporal_params=golden_input["temporal_params"],
        nearest_depth_offset=golden_input["nearest_depth_offset"],
        derivative=derivative,
        disocclusion_mask=disocclusion_mask,
        hr_shape=hr_shape,
    )
    return case, golden_output


def _native_settings(case: NSSV1PostprocessCase) -> _StageFlags:
    """Translate model quality flags into the native kernel's settings."""

    return _StageFlags(
        preprocess_half_res_input=case.model.preprocess_half_res_input,
        use_sparse_filter_2x2=case.model.use_sparse_filter_2x2,
        use_history_catmull=case.model.use_history_catmull,
        packed_nearest_offset_quad=case.model.packed_nearest_offset_quad,
        sharp_theta=case.model.nss_v1_sharp_theta,
    )


def _native_arguments(case: NSSV1PostprocessCase) -> dict[str, object]:
    """Build public native-kernel arguments without calling model postprocess."""

    if "offset_lut" in case.inputs:
        offset_lut = case.inputs["offset_lut"]
        idx_modulo = case.inputs["idx_modulo"]
        reset = case.inputs["reset"]
    else:
        (
            input_shape,
            _,
            _,
            _,
            _,
        ) = case.model._calculate_dispatch_dims(  # pylint: disable=protected-access
            case.inputs
        )
        (
            offset_lut,
            idx_modulo,
        ) = case.model._generate_offset_lut(  # pylint: disable=protected-access
            case.inputs["jitter"], input_shape, case.hr_shape
        )
        reset = 1.0 - (case.inputs["reset_event"] == 0.0).float()
    return {
        "in_color": case.inputs["colour_linear"],
        "in_history": case.inputs["history"],
        "in_kpn_params": case.kpn_params,
        "in_temporal_params": case.temporal_params,
        "in_motion": case.inputs["motion_lr"],
        "in_nearest_depth_off": case.nearest_depth_offset,
        "in_exposure": case.inputs["exposure"],
        "in_offset_lut": offset_lut,
        "in_idx_modulo": idx_modulo,
        "in_reset": reset,
        "output_shape": case.hr_shape,
        "settings": _native_settings(case),
    }


def _run_native_postprocess(
    case: NSSV1PostprocessCase, **overrides: object
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the native module directly, bypassing NSSV1Model.postprocess."""

    arguments = _native_arguments(case)
    arguments.update(overrides)
    return post_process(**arguments)


class TestNSSV1TorchPostprocessKernel(unittest.TestCase):
    """Direct contracts for the public native PyTorch post-process kernel."""

    def test_slang_backward_identity_repeats_wrapped_dispatch_threads(self):
        """Backward must reproduce Slang's modulo-wrapped 256-thread launch."""

        for shape, first_count, high, low in (
            ((2, 3, 8, 12), 64, 2.0, 1.0),
            ((1, 3, 5, 12), 16, 5.0, 4.0),
        ):
            with self.subTest(shape=shape):
                value = torch.arange(math.prod(shape), dtype=torch.float32).reshape(
                    shape
                )
                value.requires_grad_()

                output = _slang_backward_identity(value)
                torch.testing.assert_close(output, value, rtol=0.0, atol=0.0)
                output.sum().backward()

                gradient_nhwc = value.grad.permute(0, 2, 3, 1).reshape(-1, shape[1])
                torch.testing.assert_close(
                    gradient_nhwc[:first_count],
                    torch.full_like(gradient_nhwc[:first_count], high),
                    rtol=0.0,
                    atol=0.0,
                )
                torch.testing.assert_close(
                    gradient_nhwc[first_count:],
                    torch.full_like(gradient_nhwc[first_count:], low),
                    rtol=0.0,
                    atol=0.0,
                )

    def test_history_gradient_matches_compact_slang_dispatch_golden(self):
        """Constant RGB exposes Slang's wrapped-thread backward multiplicity."""

        case = create_nss_v1_postprocess_case(
            "low",
            torch.device("cpu"),
            backend="torch",
            lr_shape=(4, 6),
            reset_value=1.0,
        )
        color_values = torch.tensor([0.25, 0.5, 0.75]).reshape(1, 3, 1, 1)
        case.inputs["motion_lr"].zero_()
        case.inputs["colour_linear"].copy_(color_values)
        case.inputs["history"] = (
            case.inputs["colour_linear"]
            .repeat_interleave(2, -2)
            .repeat_interleave(2, -1)
            .add(0.0001)
            .requires_grad_()
        )
        case.kpn_params = torch.ones_like(case.kpn_params, requires_grad=True)
        case.temporal_params = torch.full_like(
            case.temporal_params, 0.5, requires_grad=True
        )

        output, _ = _run_native_postprocess(case)
        probe = torch.tensor([1.0, 2.0, 4.0]).reshape(1, 3, 1, 1)
        history_gradient = torch.autograd.grad(
            (output * probe).sum(), case.inputs["history"]
        )[0]

        torch.testing.assert_close(
            history_gradient.sum((0, 2, 3)),
            torch.tensor([242.0494080, 484.0988159, 968.1933594]),
            rtol=_SHADER_RTOL,
            atol=_SHADER_ATOL,
        )

    def test_karis_forward_and_inverse_match_hand_computable_values(self):
        """Karis uses the pixel's maximum RGB channel as its denominator."""

        linear = torch.tensor([[[[-2.0]], [[1.0]], [[3.0]]]], dtype=torch.float64)
        mapped = _karis_forward(linear)

        torch.testing.assert_close(
            mapped,
            torch.tensor([[[[0.0]], [[0.25]], [[0.75]]]], dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            _karis_inverse(mapped),
            torch.tensor([[[[0.0]], [[1.0]], [[3.0]]]], dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )

    def test_karis_inverse_clamps_to_shader_hdr_limit(self):
        """Inverse values near one must saturate at Karis(float3(65504))."""

        actual = _karis_inverse(torch.ones((1, 3, 1, 1), dtype=torch.float64))

        torch.testing.assert_close(
            actual,
            torch.full_like(actual, 65504.0),
            rtol=4.0e-12,
            atol=1.0e-8,
        )

    def test_slang_goldens_match_direct_native_kernel(self):
        """All quality modes should reproduce Slang outputs on CPU."""

        for quality in _QUALITIES:
            with self.subTest(quality=quality):
                case, expected = _make_golden_postprocess_case(quality)

                output_linear, filtered_linear = _run_native_postprocess(case)

                expected_output = expected["output_linear"]
                _assert_output_linear_matches_golden(
                    output_linear,
                    expected_output,
                    quality,
                    case,
                )
                if "out_filtered_linear" in expected:
                    torch.testing.assert_close(
                        filtered_linear,
                        expected["out_filtered_linear"],
                        rtol=_SHADER_RTOL,
                        atol=_SHADER_ATOL,
                    )
                filtered = tonemap_forward(
                    filtered_linear * case.inputs["exposure"],
                    mode=case.model.tonemapper,
                )
                torch.testing.assert_close(
                    filtered,
                    expected["out_filtered"],
                    rtol=_SHADER_RTOL,
                    atol=_SHADER_ATOL,
                )

    def test_autograd_matches_shader_diff_annotations_for_each_quality(self):
        """Only history, KPN, and temporal inputs should receive gradients."""

        for quality in _QUALITIES:
            with self.subTest(quality=quality):
                case = create_nss_v1_postprocess_case(
                    quality, torch.device("cpu"), backend="torch", lr_shape=(8, 12)
                )
                arguments = _native_arguments(case)
                differentiable_names = (
                    "in_history",
                    "in_kpn_params",
                    "in_temporal_params",
                )
                no_diff_names = (
                    "in_color",
                    "in_motion",
                    "in_nearest_depth_off",
                    "in_exposure",
                    "in_offset_lut",
                    "in_idx_modulo",
                    "in_reset",
                )
                for name in differentiable_names + no_diff_names:
                    arguments[name] = arguments[name].detach().requires_grad_()

                output, filtered = post_process(**arguments)
                ramp = torch.linspace(0.25, 1.25, output.numel()).reshape_as(output)
                (output * ramp).mean().add((filtered * ramp.flip(-1)).mean()).backward()

                for name in differentiable_names:
                    tensor = arguments[name]
                    self.assertIsNotNone(tensor.grad, name)
                    self.assertEqual(tensor.grad.shape, tensor.shape)
                    self.assertTrue(torch.isfinite(tensor.grad).all().item(), name)
                for name in no_diff_names:
                    self.assertIsNone(arguments[name].grad, name)

    def test_reset_exposure_and_scale_contracts_directly(self):
        """Direct calls should honor gates and exact integer/noninteger shapes."""

        for scale in (1.5, 2.0):
            case = create_nss_v1_postprocess_case(
                "low",
                torch.device("cpu"),
                backend="torch",
                scale=scale,
                lr_shape=(10, 14),
            )
            output, filtered = _run_native_postprocess(case)
            self.assertEqual(output.shape, case.hr_shape)
            self.assertEqual(filtered.shape, case.hr_shape)

        case = create_nss_v1_postprocess_case(
            "mid", torch.device("cpu"), backend="torch", lr_shape=(8, 12)
        )
        _configure_temporal_contrast_case(case)
        arguments = _native_arguments(case)
        reset_on, _ = post_process(**arguments)
        reset_off, filtered_one = post_process(
            **{**arguments, "in_reset": torch.zeros_like(arguments["in_reset"])}
        )
        self.assertGreater(torch.max(torch.abs(reset_on - reset_off)).item(), 0.05)

        doubled_exposure = arguments["in_exposure"] * 2.0
        exposure_output, filtered_two = post_process(
            **{
                **arguments,
                "in_exposure": doubled_exposure,
                "in_reset": torch.zeros_like(arguments["in_reset"]),
            }
        )
        torch.testing.assert_close(
            exposure_output, reset_off, rtol=_SHADER_RTOL, atol=_SHADER_ATOL
        )
        torch.testing.assert_close(
            filtered_two, filtered_one, rtol=_SHADER_RTOL, atol=_SHADER_ATOL
        )

    def test_validation_names_incompatible_inputs(self):
        """Malformed public inputs should fail before obscure gather operations."""

        case = create_nss_v1_postprocess_case(
            "high", torch.device("cpu"), backend="torch", lr_shape=(8, 12)
        )
        valid = _native_arguments(case)
        bad_cases = (
            (
                "in_color",
                valid["in_color"][:, :2],
                "in_color.*at least 3 channels",
            ),
            (
                "in_motion",
                valid["in_motion"][:1],
                "in_motion.*batch",
            ),
            (
                "in_history",
                valid["in_history"][..., :-1, :],
                "in_history.*spatial.*output_shape",
            ),
            (
                "output_shape",
                (2, 4, 16, 24),
                "output_shape.*3 channels",
            ),
            (
                "in_idx_modulo",
                torch.ones((1, 1, 1, 1)),
                "in_idx_modulo.*at least 2 values",
            ),
            (
                "in_offset_lut",
                valid["in_offset_lut"][:1],
                "in_offset_lut.*batch",
            ),
            (
                "filter_kernel_taps",
                valid["in_offset_lut"].shape[-1] + 1,
                "tap count.*filter_kernel_taps",
            ),
        )
        for name, value, message in bad_cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                post_process(**{**valid, name: value})

    def test_validation_requires_float32_inputs_without_mixed_dtypes(self):
        """The float32 shader contract must reject unsafe half and mixed inputs."""

        case = create_nss_v1_postprocess_case(
            "high", torch.device("cpu"), backend="torch", lr_shape=(8, 12)
        )
        valid = _native_arguments(case)
        for name, value in (
            ("in_color", valid["in_color"].to(torch.float16)),
            ("in_history", valid["in_history"].to(torch.bfloat16)),
            ("in_kpn_params", valid["in_kpn_params"].to(torch.float64)),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, rf"{name}.*torch.float32"
            ):
                post_process(**{**valid, name: value})

    def test_full_res_modulo_metadata_uses_batch_zero_like_slang(self):
        """Full-res mode must use the single modulo row read by Slang."""

        case = create_nss_v1_postprocess_case(
            "high", torch.device("cpu"), backend="torch", lr_shape=(8, 12)
        )
        valid = _native_arguments(case)
        shared = valid["in_idx_modulo"].expand(case.hr_shape[0], -1, -1, -1).clone()
        output_shared, filtered_shared = post_process(
            **{**valid, "in_idx_modulo": shared}
        )
        output_single, filtered_single = post_process(**valid)
        torch.testing.assert_close(output_shared, output_single, rtol=0.0, atol=0.0)
        torch.testing.assert_close(filtered_shared, filtered_single, rtol=0.0, atol=0.0)

        shared[1, 0, 0, 0] -= 1.0
        output_distinct, filtered_distinct = post_process(
            **{**valid, "in_idx_modulo": shared}
        )
        torch.testing.assert_close(output_distinct, output_single, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            filtered_distinct, filtered_single, rtol=0.0, atol=0.0
        )

    def test_moderate_forward_backward_streams_spatial_taps(self):
        """A moderate differentiable call should never form a full tap axis."""

        case = create_nss_v1_postprocess_case(
            "high", torch.device("cpu"), backend="torch", lr_shape=(32, 48)
        )
        arguments = _native_arguments(case)
        for name in ("in_history", "in_kpn_params", "in_temporal_params"):
            arguments[name] = arguments[name].detach().requires_grad_()

        with mock.patch(
            "ng_model_gym.usecases.nss.model.torch_postprocess.filter.kpn_coordinates",
            wraps=kpn_coordinates,
        ) as kpn_coordinate_mock:
            output, filtered = post_process(**arguments)
            (output.mean() + filtered.mean()).backward()

        self.assertEqual(kpn_coordinate_mock.call_count, 9)
        self.assertTrue(
            all(call.args[0].ndim == 4 for call in kpn_coordinate_mock.call_args_list)
        )
        self.assertEqual(output.shape, (2, 3, 64, 96))


class TestNSSV1TorchPostprocessGolden(unittest.TestCase):
    """Regression checks against Slang-generated postprocess goldens."""

    def test_slang_goldens_match_torch_backend(self):
        """Every quality's Slang golden should run unchanged on CPU."""

        for quality in _QUALITIES:
            with self.subTest(quality=quality):
                case, expected = _make_golden_postprocess_case(quality)

                actual = _run_postprocess(case)

                _assert_output_linear_matches_golden(
                    actual["output_linear"],
                    expected["output_linear"],
                    quality,
                    case,
                )
                torch.testing.assert_close(
                    actual["out_filtered"],
                    expected["out_filtered"],
                    rtol=_SHADER_RTOL,
                    atol=_SHADER_ATOL,
                )


class TestNSSV1TorchPostprocessCPU(unittest.TestCase):
    """CPU contracts for the native PyTorch postprocess pipeline."""

    def test_core_forward_runs_all_native_stages_without_cuda_or_slang(self):
        """A complete CPU frame must avoid every Slang/CUDA loading entrypoint."""

        params = create_nss_v1_test_params("high")
        params.model.processing_backend = "torch"
        model = NSSV1Model(params).eval()
        inputs = _create_core_forward_inputs(model)
        snapshots = {name: value.clone() for name, value in inputs.items()}

        with mock.patch.object(
            model,
            "_require_cuda_for_slang_forward",
            side_effect=AssertionError("native core_forward entered the CUDA guard"),
        ), mock.patch.object(
            model,
            "_get_slang",
            side_effect=AssertionError("native core_forward requested Slang"),
        ), mock.patch(
            "ng_model_gym.usecases.nss.model.model_v1.load_slang_module",
            side_effect=AssertionError("native core_forward loaded a Slang module"),
        ):
            outputs = model.core_forward(inputs)

        self.assertEqual(
            set(outputs),
            set(NSS_V1_CORE_OUTPUT_KEYS),
        )
        for value in outputs.values():
            self.assertEqual(value.device.type, "cpu")
            self.assertTrue(torch.isfinite(value).all().item())
        for name, snapshot in snapshots.items():
            torch.testing.assert_close(inputs[name], snapshot, rtol=0.0, atol=0.0)

    def test_dynamic_batches_and_odd_nonmultiple_shapes(self):
        """Native postprocess supports runtime batches and unpadded LR extents."""

        for quality in _QUALITIES:
            for batch, lr_shape in ((1, (15, 21)), (2, (19, 27))):
                with self.subTest(quality=quality, batch=batch, lr_shape=lr_shape):
                    case = create_nss_v1_postprocess_case(
                        quality,
                        torch.device("cpu"),
                        backend="torch",
                        batch=batch,
                        scale=1.5,
                        lr_shape=lr_shape,
                    )
                    outputs = _run_postprocess(case)
                    _assert_finite_hr_outputs(self, outputs, case.hr_shape)

    def test_torch_backend_never_loads_slang(self):
        """Selecting torch should avoid both Slang loading and CUDA guards."""

        case = create_nss_v1_postprocess_case(
            "high", torch.device("cpu"), backend="torch"
        )
        with mock.patch.object(
            case.model,
            "_require_cuda_for_slang_forward",
            side_effect=AssertionError("torch backend attempted a Slang CUDA guard"),
        ), mock.patch.object(
            case.model,
            "_get_slang",
            side_effect=AssertionError("torch backend attempted to load Slang"),
        ):
            outputs = _run_postprocess(case)

        _assert_finite_hr_outputs(self, outputs, case.hr_shape)

    def test_slang_backend_retains_cuda_guard(self):
        """Selecting Slang on CPU should fail before attempting to load Slang."""

        case = create_nss_v1_postprocess_case(
            "high", torch.device("cpu"), backend="slang"
        )
        with mock.patch.object(case.model, "_get_slang") as get_slang:
            with self.assertRaisesRegex(
                RuntimeError,
                "NSS-v1 Slang-backed forward requires CUDA",
            ):
                _run_postprocess(case)

        get_slang.assert_not_called()

    def test_backend_routes_return_the_same_public_output_keys(self):
        """Backend selection should not change the public post-process contract."""

        case = create_nss_v1_postprocess_case(
            "high", torch.device("cpu"), backend="torch", lr_shape=(8, 12)
        )
        torch_outputs = _run_postprocess(case)

        case.model.processing_backend = "slang"
        slang = mock.Mock()
        slang.post_process.return_value = (
            torch.zeros(case.hr_shape),
            torch.ones(case.hr_shape),
        )
        with mock.patch.object(
            case.model, "_require_cuda_for_slang_forward"
        ), mock.patch.object(case.model, "_get_slang", return_value=slang):
            slang_outputs = _run_postprocess(case)

        self.assertEqual(torch_outputs.keys(), slang_outputs.keys())

    def test_public_output_packaging_passes_context_through(self):
        """Post-processing should preserve the existing loss-context tensors."""

        case = create_nss_v1_postprocess_case(
            "mid", torch.device("cpu"), backend="torch", lr_shape=(8, 12)
        )

        outputs = _run_postprocess(case)

        self.assertIs(outputs["temporal_params"], case.temporal_params)
        self.assertIs(outputs["derivative"], case.derivative)
        self.assertIs(outputs["disocclusion_mask"], case.disocclusion_mask)

    def test_public_output_packaging_uses_configured_tonemapper(self):
        """Linear products, ground truth, and input color share outer tonemapping."""

        case = create_nss_v1_postprocess_case(
            "low",
            torch.device("cpu"),
            backend="torch",
            exposure=2.0,
            lr_shape=(8, 12),
        )

        outputs = _run_postprocess(case)

        expected_output = tonemap_forward(
            outputs["output_linear"] * case.inputs["exposure"],
            mode=case.model.tonemapper,
        )
        expected_filtered = tonemap_forward(
            _run_native_postprocess(case)[1] * case.inputs["exposure"],
            mode=case.model.tonemapper,
        )
        expected_ground_truth = tonemap_forward(
            case.inputs["ground_truth_linear"] * case.inputs["exposure"],
            mode=case.model.tonemapper,
        )
        expected_input = tonemap_forward(
            case.inputs["colour_linear"] * case.inputs["exposure"],
            mode=case.model.tonemapper,
        )
        for key, expected in (
            ("output", expected_output),
            ("out_filtered", expected_filtered),
            ("ground_truth", expected_ground_truth),
            ("input_color", expected_input),
        ):
            torch.testing.assert_close(outputs[key], expected, rtol=0.0, atol=0.0)

    def test_autograd_reaches_history_kpn_and_temporal_for_each_quality(self):
        """All differentiable shader inputs should receive finite gradients."""

        for quality in _QUALITIES:
            with self.subTest(quality=quality):
                case = create_nss_v1_postprocess_case(
                    quality, torch.device("cpu"), backend="torch"
                )
                case.inputs["history"] = case.inputs["history"].requires_grad_()
                case.kpn_params = case.kpn_params.requires_grad_()
                case.temporal_params = case.temporal_params.requires_grad_()

                outputs = _run_postprocess(case)
                postprocess_gradient_probe(outputs).backward()

                differentiable_inputs = (
                    case.inputs["history"],
                    case.kpn_params,
                    case.temporal_params,
                )
                for tensor in differentiable_inputs:
                    self.assertIsNotNone(tensor.grad)
                    self.assertEqual(tensor.grad.shape, tensor.shape)
                    self.assertTrue(torch.isfinite(tensor.grad).all().item())

    def test_reset_gate_changes_temporal_accumulation(self):
        """Reset one should admit distinct history while reset zero rejects it."""

        for quality in _QUALITIES:
            with self.subTest(quality=quality):
                reset_outputs = []
                for reset_value in (0.0, 1.0):
                    case = create_nss_v1_postprocess_case(
                        quality,
                        torch.device("cpu"),
                        backend="torch",
                        reset_value=reset_value,
                        lr_shape=(8, 12),
                    )
                    _configure_temporal_contrast_case(case)
                    reset_outputs.append(_run_postprocess(case)["output_linear"])

                difference = torch.max(torch.abs(reset_outputs[1] - reset_outputs[0]))
                self.assertGreater(difference.item(), 0.05)

    def test_exposure_is_visible_only_in_tonemapped_outputs(self):
        """Reset-zero linear output stays fixed while returned tone maps respond."""

        for quality in _QUALITIES:
            with self.subTest(quality=quality):
                exposure_outputs = []
                for exposure in (0.5, 2.0):
                    case = create_nss_v1_postprocess_case(
                        quality,
                        torch.device("cpu"),
                        backend="torch",
                        reset_value=0.0,
                        exposure=exposure,
                        lr_shape=(8, 12),
                    )
                    _configure_temporal_contrast_case(case)
                    exposure_outputs.append(_run_postprocess(case))

                torch.testing.assert_close(
                    exposure_outputs[0]["output_linear"],
                    exposure_outputs[1]["output_linear"],
                    rtol=_SHADER_RTOL,
                    atol=_SHADER_ATOL,
                )
                for key in ("output", "out_filtered"):
                    exposure_difference = torch.max(
                        torch.abs(exposure_outputs[1][key] - exposure_outputs[0][key])
                    )
                    self.assertGreater(exposure_difference.item(), 0.01)

    def test_scale_controls_exact_rounded_output_shape(self):
        """Scale 1.5 and 2.0 should produce their exact model-rounded HR sizes."""

        expected_spatial = {1.5: (15, 21), 2.0: (20, 28)}
        for quality in _QUALITIES:
            for scale, spatial_shape in expected_spatial.items():
                with self.subTest(quality=quality, scale=scale):
                    case = create_nss_v1_postprocess_case(
                        quality,
                        torch.device("cpu"),
                        backend="torch",
                        scale=scale,
                        lr_shape=(10, 14),
                    )
                    self.assertEqual(case.hr_shape, (2, 3, *spatial_shape))

                    outputs = _run_postprocess(case)

                    self.assertEqual(outputs["output_linear"].shape, case.hr_shape)
                    self.assertEqual(outputs["out_filtered"].shape, case.hr_shape)

    def test_motion_threshold_is_strictly_greater_than_point_one(self):
        """Lengths at or below 0.1 should zero motion; one above should reproject."""

        for quality in _QUALITIES:
            with self.subTest(quality=quality):
                outputs = []
                for length in (0.099999, 0.1, 0.100001):
                    case = create_nss_v1_postprocess_case(
                        quality,
                        torch.device("cpu"),
                        backend="torch",
                        lr_shape=(8, 12),
                    )
                    _set_constant_output_motion(case, (length, 0.0))
                    outputs.append(_run_postprocess(case)["output_linear"])

                torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=0.0)
                self.assertFalse(torch.equal(outputs[1], outputs[2]))

    def test_nearest_offsets_change_outputs_at_all_lr_borders(self):
        """Border-clamped motion lookups should alter four near-corner outputs."""

        for quality in _QUALITIES:
            with self.subTest(quality=quality):
                control = create_nss_v1_postprocess_case(
                    quality,
                    torch.device("cpu"),
                    backend="torch",
                    lr_shape=(8, 12),
                )
                border = create_nss_v1_postprocess_case(
                    quality,
                    torch.device("cpu"),
                    backend="torch",
                    lr_shape=(8, 12),
                )
                _configure_nearest_offset_probe(control)
                _configure_nearest_offset_probe(border)
                _set_behavior_sensitive_border_offsets(border)

                control_output = _run_postprocess(control)["output_linear"]
                border_output = _run_postprocess(border)["output_linear"]

                for row, column in _near_corner_output_pixels(border):
                    difference = torch.max(
                        torch.abs(
                            border_output[:, :, row, column]
                            - control_output[:, :, row, column]
                        )
                    )
                    self.assertGreater(difference.item(), 1.0e-4)

    def test_history_boundary_gates_one_pixel_beyond_reprojection(self):
        """Inside/exact history contributes while one-pixel-beyond history does not."""

        boundary_cases = {
            "top": ((0.499999, 0.0), (0.5, 0.0), (1.5, 0.0), (0, None)),
            "bottom": (
                (-0.499999, 0.0),
                (-0.5, 0.0),
                (-1.5, 0.0),
                (-1, None),
            ),
            "left": ((0.0, 0.499999), (0.0, 0.5), (0.0, 1.5), (None, 0)),
            "right": (
                (0.0, -0.499999),
                (0.0, -0.5),
                (0.0, -1.5),
                (None, -1),
            ),
        }
        for quality in _QUALITIES:
            for boundary, (inside, exact, beyond, pixel) in boundary_cases.items():
                with self.subTest(quality=quality, boundary=boundary):
                    outputs = []
                    for motion in (inside, exact, beyond):
                        case = create_nss_v1_postprocess_case(
                            quality,
                            torch.device("cpu"),
                            backend="torch",
                            lr_shape=(8, 12),
                        )
                        _configure_temporal_contrast_case(case)
                        _set_constant_output_motion(case, motion)
                        output = _run_postprocess(case)["output_linear"]
                        row = output.shape[-2] // 2 if pixel[0] is None else pixel[0]
                        column = output.shape[-1] // 2 if pixel[1] is None else pixel[1]
                        outputs.append(output[:, :, row, column].mean())

                    self.assertGreater((outputs[0] - outputs[2]).item(), 0.05)
                    self.assertGreater((outputs[1] - outputs[2]).item(), 0.05)

    def test_kpn_epsilon_floor_treats_zero_and_half_epsilon_equally(self):
        """Zero and EPS/2 weights should clamp alike while EPS*2 remains distinct."""

        for quality in _QUALITIES:
            with self.subTest(quality=quality):
                outputs = []
                for selected_weight in (0.0, _EPS / 2.0, _EPS * 2.0):
                    case = create_nss_v1_postprocess_case(
                        quality,
                        torch.device("cpu"),
                        backend="torch",
                        lr_shape=(8, 12),
                    )
                    case.kpn_params.fill_(_EPS * 2.0)
                    case.kpn_params[:, 0].fill_(selected_weight)
                    outputs.append(_run_postprocess(case)["out_filtered"])

                torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=0.0)
                self.assertFalse(torch.equal(outputs[1], outputs[2]))
