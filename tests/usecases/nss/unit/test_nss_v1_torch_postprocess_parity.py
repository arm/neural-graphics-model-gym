# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import unittest
from collections.abc import Callable
from unittest import mock

import torch

from ng_model_gym.core.model.shaders import slang_utils
from ng_model_gym.usecases.nss.model.torch_postprocess.sampling import EPS as _EPS
from tests.usecases.nss.unit.nss_v1_torch_postprocess_test_utils import (
    _configure_controlled_precise_case,
    _QUALITIES,
    _run_postprocess,
    _set_constant_output_motion,
    _set_corner_clamping_offsets,
    _set_high_frequency_history,
    create_nss_v1_postprocess_case,
    NSSV1PostprocessCase,
)

# The precise Slang cache/loader lifecycle intentionally mirrors preprocess parity.
# pylint: disable=duplicate-code


def _make_backend_pair(
    quality: str,
    **case_kwargs,
) -> tuple[NSSV1PostprocessCase, NSSV1PostprocessCase]:
    """Create identical Slang and Torch cases on CUDA."""

    slang_case = create_nss_v1_postprocess_case(
        quality, torch.device("cuda"), backend="slang", **case_kwargs
    )
    torch_case = create_nss_v1_postprocess_case(
        quality, torch.device("cuda"), backend="torch", **case_kwargs
    )
    return slang_case, torch_case


def _make_differentiable(case: NSSV1PostprocessCase) -> tuple[torch.Tensor, ...]:
    """Clone the shader's three differentiable sources as leaves."""

    case.inputs["history"] = case.inputs["history"].detach().clone().requires_grad_()
    case.kpn_params = case.kpn_params.detach().clone().requires_grad_()
    case.temporal_params = case.temporal_params.detach().clone().requires_grad_()
    return case.inputs["history"], case.kpn_params, case.temporal_params


@unittest.skipUnless(
    torch.cuda.is_available(),
    "CUDA is required for live NSS v1 Slang/Torch postprocess parity.",
)
class TestNSSV1TorchPostprocessSlangParity(unittest.TestCase):
    """Live elementwise comparisons against precise authoritative Slang."""

    @classmethod
    def setUpClass(cls) -> None:
        """Compile parity-only Slang modules without CUDA fast math."""

        super().setUpClass()
        # Fast-math mode is absent from the repository loader's cache key.
        # pylint: disable-next=protected-access
        slang_utils._load_slang_module_cached.cache_clear()
        load_module = slang_utils.slangtorch.loadModule
        cls._slang_load_calls = []

        def load_module_without_fast_math(*args, **kwargs):
            kwargs["cudaFastMath"] = False
            cls._slang_load_calls.append(dict(kwargs))
            return load_module(*args, **kwargs)

        cls._slang_load_patcher = mock.patch.object(
            slang_utils.slangtorch,
            "loadModule",
            side_effect=load_module_without_fast_math,
        )
        cls._slang_load_patcher.start()

    @classmethod
    def tearDownClass(cls) -> None:
        """Verify precise compilation and clear parity modules from the cache."""

        try:
            if not cls._slang_load_calls:
                raise AssertionError("No Slang parity module was compiled.")
            if any(
                call.get("cudaFastMath") is not False for call in cls._slang_load_calls
            ):
                raise AssertionError("A parity Slang module used CUDA fast math.")
        finally:
            cls._slang_load_patcher.stop()
            # pylint: disable-next=protected-access
            slang_utils._load_slang_module_cached.cache_clear()
            super().tearDownClass()

    def _assert_backend_parity(
        self,
        quality: str,
        *,
        mutate: Callable[[NSSV1PostprocessCase], None] | None = None,
        compare_vjp: bool = True,
        kpn_atol: float = 1.0e-7,
        forward_rtol: float = 1.0e-5,
        forward_atol: float = 5.0e-6,
        gradient_names: tuple[str, ...] = ("history", "kpn", "temporal"),
        **case_kwargs,
    ) -> None:
        slang_case, torch_case = _make_backend_pair(quality, **case_kwargs)
        if mutate is not None:
            mutate(slang_case)
            mutate(torch_case)

        slang_sources = _make_differentiable(slang_case)
        torch_sources = _make_differentiable(torch_case)
        slang_outputs = _run_postprocess(slang_case)
        torch_outputs = _run_postprocess(torch_case)
        for key in ("output", "output_linear", "out_filtered"):
            torch.testing.assert_close(
                torch_outputs[key],
                slang_outputs[key],
                rtol=forward_rtol,
                atol=forward_atol,
            )

        if not compare_vjp:
            return
        generator = torch.Generator(device="cuda").manual_seed(1984)
        output_cotangent = torch.rand(
            slang_outputs["output_linear"].shape,
            generator=generator,
            device="cuda",
        )
        filtered_cotangent = torch.rand(
            slang_outputs["out_filtered"].shape,
            generator=generator,
            device="cuda",
        )
        # Probe the mean-loss scale used by training while retaining a random,
        # identical cotangent for both backends. This isolates Slang atomic
        # accumulation order from translation errors at tiny gradient entries.
        output_cotangent = output_cotangent / output_cotangent.numel()
        filtered_cotangent = filtered_cotangent / filtered_cotangent.numel()

        def probe(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
            return torch.sum(outputs["output_linear"] * output_cotangent) + torch.sum(
                outputs["out_filtered"] * filtered_cotangent
            )

        slang_gradients = torch.autograd.grad(probe(slang_outputs), slang_sources)
        torch_gradients = torch.autograd.grad(probe(torch_outputs), torch_sources)
        for name, torch_gradient, slang_gradient in zip(
            ("history", "kpn", "temporal"), torch_gradients, slang_gradients
        ):
            if name not in gradient_names:
                continue
            torch.testing.assert_close(
                torch_gradient,
                slang_gradient,
                rtol=1.0e-5,
                atol=kpn_atol if name == "kpn" else 1.0e-7,
            )

    def test_controlled_forward_uses_precise_tolerance(self):
        """Controlled ordinary inputs match each quality at precise tolerance."""

        for quality in _QUALITIES:
            with self.subTest(quality=quality):
                self._assert_backend_parity(
                    quality,
                    mutate=_configure_controlled_precise_case,
                    lr_shape=(16, 24),
                    seed=0,
                    compare_vjp=False,
                    forward_rtol=1.0e-6,
                    forward_atol=1.0e-7,
                )

    def test_randomized_quality_shape_scale_exposure_reset_matrix(self):
        """Random cases cover both batches, scales, exposures, resets, and seeds."""

        cases = (
            {
                "batch": 1,
                "lr_shape": (16, 24),
                "scale": 1.5,
                "exposure": 0.5,
                "reset_value": 0.0,
                "seed": 0,
            },
            {
                "batch": 2,
                "lr_shape": (15, 21),
                "scale": 2.0,
                "exposure": 1.0,
                "reset_value": 1.0,
                "seed": 1,
            },
            {
                "batch": 1,
                "lr_shape": (19, 27),
                "scale": 1.5,
                "exposure": 2.0,
                "reset_value": 0.0,
                "seed": 2,
            },
        )
        for quality in _QUALITIES:
            for case in cases:
                with self.subTest(quality=quality, **case):
                    self._assert_backend_parity(quality, **case)

    def test_history_boundaries_motion_threshold_and_nearest_borders(self):
        """Branch-sensitive reprojection and nearest-offset cases match Slang."""

        motions = (
            (0.099999, 0.0),
            (0.1, 0.0),
            (0.100001, 0.0),
            (0.0707106, 0.0707106),
            (0.070710678, 0.070710678),
            (0.0707110, 0.0707110),
            (0.499999, 0.0),
            (0.5, 0.0),
            (1.5, 0.0),
            (-0.499999, 0.0),
            (-0.5, 0.0),
            (-1.5, 0.0),
            (0.0, 0.499999),
            (0.0, 0.5),
            (0.0, 1.5),
            (0.0, -0.499999),
            (0.0, -0.5),
            (0.0, -1.5),
        )
        for quality in _QUALITIES:
            for motion in motions:
                with self.subTest(quality=quality, motion=motion):
                    self._assert_backend_parity(
                        quality,
                        mutate=lambda case, value=motion: _set_constant_output_motion(
                            case, value
                        ),
                        lr_shape=(8, 12),
                        compare_vjp=False,
                    )
            with self.subTest(quality=quality, nearest_offsets="all_borders"):
                self._assert_backend_parity(
                    quality,
                    mutate=_set_corner_clamping_offsets,
                    lr_shape=(8, 12),
                    compare_vjp=False,
                )

    def test_bilinear_catmull_and_padded_temporal_paths(self):
        """Quality-specific history and temporal-padding paths match Slang."""

        def configure(case: NSSV1PostprocessCase) -> None:
            _set_high_frequency_history(case)
            _set_constant_output_motion(case, (0.375, -0.4375))
            if case.model.preprocess_half_res_input:
                process_height = case.inputs["colour_linear"].shape[-2] // 2
                process_width = case.inputs["colour_linear"].shape[-1] // 2
                case.temporal_params[:, :, process_height:, :] = 0.97
                case.temporal_params[:, :, :, process_width:] = 0.03

        for quality in _QUALITIES:
            with self.subTest(quality=quality):
                self._assert_backend_parity(
                    quality, mutate=configure, lr_shape=(18, 26)
                )

    def test_kpn_epsilon_atomic_case_has_isolated_allowance(self):
        """Only KPN epsilon atomic accumulation receives the measured allowance."""

        def set_weight(case: NSSV1PostprocessCase, value: float) -> None:
            case.kpn_params.fill_(_EPS * 2.0)
            case.kpn_params[:, 0].fill_(value)

        for quality in _QUALITIES:
            for value in (0.0, _EPS / 2.0, _EPS, _EPS * 2.0):
                with self.subTest(quality=quality, value=value):
                    self._assert_backend_parity(
                        quality,
                        mutate=lambda case, weight=value: set_weight(case, weight),
                        lr_shape=(8, 12),
                        kpn_atol=7.0e-2,
                        gradient_names=("kpn",),
                    )

    def test_torch_cpu_cuda_outputs_match(self):
        """The native implementation is device-independent for every quality."""

        for quality in _QUALITIES:
            cpu_case = create_nss_v1_postprocess_case(
                quality,
                torch.device("cpu"),
                backend="torch",
                batch=2,
                scale=1.5,
                lr_shape=(15, 21),
                seed=2,
            )
            cuda_case = create_nss_v1_postprocess_case(
                quality,
                torch.device("cuda"),
                backend="torch",
                batch=2,
                scale=1.5,
                lr_shape=(15, 21),
                seed=2,
            )
            cuda_case.inputs = {
                key: value.to("cuda") for key, value in cpu_case.inputs.items()
            }
            cuda_case.kpn_params = cpu_case.kpn_params.to("cuda")
            cuda_case.temporal_params = cpu_case.temporal_params.to("cuda")
            cuda_case.nearest_depth_offset = cpu_case.nearest_depth_offset.to("cuda")
            actual = _run_postprocess(cuda_case)
            expected = _run_postprocess(cpu_case)
            for key in ("output", "output_linear", "out_filtered"):
                torch.testing.assert_close(
                    actual[key].cpu(), expected[key], rtol=1.0e-5, atol=5.0e-6
                )
