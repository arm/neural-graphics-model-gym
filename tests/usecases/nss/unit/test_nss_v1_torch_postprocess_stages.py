# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import mock

import torch

from ng_model_gym.usecases.nss.model.torch_postprocess.filter import kpn_coordinates
from ng_model_gym.usecases.nss.model.torch_postprocess.filter import (
    map_sparse_2x2_kpn_channels as _map_sparse_2x2_kpn_channels,
)
from ng_model_gym.usecases.nss.model.torch_postprocess.pipeline import (
    rectify_history as _rectify_history,
)
from ng_model_gym.usecases.nss.model.torch_postprocess.sampling import EPS as _EPS
from ng_model_gym.usecases.nss.model.torch_postprocess.sampling import (
    fused_multiply_add,
)
from ng_model_gym.usecases.nss.model.torch_postprocess.sampling import (
    reproject_history_uv as _reproject_history_uv,
)
from ng_model_gym.usecases.nss.model.torch_postprocess.sampling import (
    sample_bilinear as _sample_bilinear,
)
from ng_model_gym.usecases.nss.model.torch_postprocess.sampling import (
    sample_catmull_rom as _sample_catmull_rom,
)
from tests.usecases.nss.unit.nss_v1_torch_postprocess_test_utils import (
    _decode_high_offset_row_col,
    _decode_nearest_offsets,
    _decode_packed_nibble_row_col,
    _filter_color,
    _kpn_coordinates,
    _load_motion,
    _sample_temporal_params,
    _set_corner_clamping_offsets,
    _StageFlags,
    create_nss_v1_postprocess_case,
)


def _bilinear_border_oracle(
    tensor: torch.Tensor, batch: int, uv_yx: torch.Tensor
) -> torch.Tensor:
    """Explicitly bilerp one batch at a normalized UV with border clamping."""

    height, width = tensor.shape[-2:]
    source_y = uv_yx[0] * height - 0.5
    source_x = uv_yx[1] * width - 0.5
    y0 = int(torch.floor(source_y).detach().item())
    x0 = int(torch.floor(source_x).detach().item())
    y1 = y0 + 1
    x1 = x0 + 1
    fraction_y = source_y - y0
    fraction_x = source_x - x0

    y0 = min(max(y0, 0), height - 1)
    y1 = min(max(y1, 0), height - 1)
    x0 = min(max(x0, 0), width - 1)
    x1 = min(max(x1, 0), width - 1)

    top = tensor[batch, :, y0, x0] + fraction_x * (
        tensor[batch, :, y0, x1] - tensor[batch, :, y0, x0]
    )
    bottom = tensor[batch, :, y1, x0] + fraction_x * (
        tensor[batch, :, y1, x1] - tensor[batch, :, y1, x0]
    )
    return top + fraction_y * (bottom - top)


def _catmull_rom_pixel_oracle(
    tensor: torch.Tensor,
    batch: int,
    uv_yx: torch.Tensor,
    spatial_size: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one five-tap Catmull-Rom output pixel explicitly."""

    tc = torch.floor(uv_yx * spatial_size - 0.5) + 0.5
    fraction = uv_yx * spatial_size - tc
    fraction2 = fraction * fraction
    fraction3 = fraction2 * fraction
    weight0 = fraction2 - 0.5 * (fraction3 + fraction)
    weight1 = 1.5 * fraction3 - 2.5 * fraction2 + 1.0
    weight3 = 0.5 * (fraction3 - fraction2)
    weight2 = 1.0 - weight0 - weight1 - weight3
    axis_weights = torch.stack((weight0, weight1 + weight2, weight3))
    axis_positions = torch.stack((tc - 1.0, tc + weight2 / axis_weights[1], tc + 2.0))

    samples = []
    sample_weights = []
    for y_tap in range(3):
        for x_tap in range(3):
            if y_tap != 1 and x_tap != 1:
                continue
            sample_uv = torch.stack(
                (
                    axis_positions[y_tap, 0] / spatial_size[0],
                    axis_positions[x_tap, 1] / spatial_size[1],
                )
            )
            samples.append(_bilinear_border_oracle(tensor, batch, sample_uv))
            sample_weights.append(axis_weights[y_tap, 0] * axis_weights[x_tap, 1])

    stacked_samples = torch.stack(samples)
    color = sum(
        sample * weight for sample, weight in zip(samples, sample_weights)
    ) / sum(sample_weights)
    sample_min = torch.full_like(stacked_samples[0], 65504.0)
    sample_max = torch.full_like(stacked_samples[0], -65504.0)
    for sample in samples:
        sample_min = torch.minimum(sample_min, sample)
        sample_max = torch.maximum(sample_max, sample)
    clamped = torch.maximum(torch.minimum(color, sample_max), sample_min)
    return torch.where(torch.any(color < 0.0), clamped, color)


def _catmull_rom_oracle(tensor: torch.Tensor, uv_yx: torch.Tensor) -> torch.Tensor:
    """Evaluate five-tap Catmull-Rom with explicit scalar output loops."""

    spatial_size = uv_yx.new_tensor(tensor.shape[-2:])
    batches = []
    for batch in range(tensor.shape[0]):
        rows = []
        for row in range(uv_yx.shape[1]):
            pixels = []
            for column in range(uv_yx.shape[2]):
                pixels.append(
                    _catmull_rom_pixel_oracle(
                        tensor,
                        batch,
                        uv_yx[batch, row, column],
                        spatial_size,
                    )
                )
            rows.append(torch.stack(pixels, dim=-1))
        batches.append(torch.stack(rows, dim=-2))
    return torch.stack(batches)


class TestNSSV1TorchSampling(unittest.TestCase):
    """CPU tests for NSS v1 shader-accurate sampling primitives."""

    def test_bilinear_pixel_centers_return_exact_texels(self):
        """Normalized pixel centers should address their exact texels."""

        tensor = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        uv_yx = torch.tensor(
            [[[[0.25, 0.25], [0.25, 0.75]], [[0.75, 0.25], [0.75, 0.75]]]]
        )

        actual = _sample_bilinear(tensor, uv_yx)

        torch.testing.assert_close(actual, tensor, rtol=0.0, atol=0.0)

    def test_bilinear_border_clamping_differs_from_zero_padding(self):
        """A corner UV should clamp fully or blend with three zero texels."""

        tensor = torch.tensor([[[[4.0]]]])
        uv_yx = torch.tensor([[[[0.0, 0.0]]]])

        clamped = _sample_bilinear(tensor, uv_yx, clamp_to_edge=True)
        zero_padded = _sample_bilinear(tensor, uv_yx, clamp_to_edge=False)

        torch.testing.assert_close(clamped, torch.tensor([[[[4.0]]]]))
        torch.testing.assert_close(zero_padded, torch.tensor([[[[1.0]]]]))

    def test_bilinear_exact_one_uv_uses_edge_padding_semantics(self):
        """UV 1 lies half a pixel beyond the last pixel center."""

        tensor = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        uv_yx = torch.tensor([[[[1.0, 1.0]]]])

        clamped = _sample_bilinear(tensor, uv_yx, clamp_to_edge=True)
        zero_padded = _sample_bilinear(tensor, uv_yx, clamp_to_edge=False)

        torch.testing.assert_close(clamped, torch.tensor([[[[4.0]]]]))
        torch.testing.assert_close(zero_padded, torch.tensor([[[[1.0]]]]))

    def test_bilinear_offscreen_uv_clamps_or_returns_zero(self):
        """A UV at 1.25 is fully outside a 2x2 zero-padded texture."""

        tensor = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        uv_yx = torch.tensor([[[[1.25, 1.25]]]])

        clamped = _sample_bilinear(tensor, uv_yx, clamp_to_edge=True)
        zero_padded = _sample_bilinear(tensor, uv_yx, clamp_to_edge=False)

        torch.testing.assert_close(clamped, torch.tensor([[[[4.0]]]]))
        torch.testing.assert_close(zero_padded, torch.tensor([[[[0.0]]]]))

    def test_catmull_rom_negative_overshoot_derings_all_channels(self):
        """One negative channel should clamp every channel to sampled bounds."""

        tensor = torch.tensor([[[[8.0, 0.0, 0.0, 0.0]], [[0.0, 8.0, 8.0, 0.0]]]])
        uv_yx = torch.tensor([[[[0.5, 0.5]]]])

        actual = _sample_catmull_rom(tensor, uv_yx)

        # At the half-pixel position, the independent 1D Catmull-Rom weights
        # are [-1/16, 9/16, 9/16, -1/16]. They produce [-0.5, 9.0].
        # The five bilinear samples span [0, 8] in both channels, so the
        # negative first channel triggers pixel-wide clamping to [0, 8].
        expected = torch.tensor([[[[0.0]], [[8.0]]]])
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_catmull_rom_dering_uses_shader_half_range_bounds(self):
        """Dering bounds should retain the shader's initial half-float range."""

        tensor = torch.tensor(
            [[[[8.0, 0.0, 0.0, 0.0]], [[80000.0, 70000.0, 70000.0, 80000.0]]]]
        )
        uv_yx = torch.tensor([[[[0.5, 0.5]]]])

        actual = _sample_catmull_rom(tensor, uv_yx)

        # The half-pixel Catmull-Rom weights [-1/16, 9/16, 9/16, -1/16]
        # produce [-0.5, 68750]. The negative channel triggers deringing.
        # Slang seeds the sampled bounds at +/-65504, so 68750 is retained.
        expected = torch.tensor([[[[0.0]], [[68750.0]]]])
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_catmull_rom_matches_independent_2d_oracle_with_finite_gradients(self):
        """Asymmetric 2D border samples should match an explicit oracle."""

        values = torch.arange(2 * 3 * 3 * 5, dtype=torch.float64).reshape(2, 3, 3, 5)
        tensor = (torch.sin(values * 0.37) + 2.5).requires_grad_()
        uv_yx = torch.tensor(
            [
                [[[0.08, 0.23], [0.37, 0.94]], [[0.91, 0.12], [0.63, 0.57]]],
                [[[0.14, 0.88], [0.46, 0.07]], [[0.79, 0.71], [0.97, 0.41]]],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )

        actual = _sample_catmull_rom(tensor, uv_yx)
        expected = _catmull_rom_oracle(tensor, uv_yx)

        self.assertEqual(actual.shape, (2, 3, 2, 2))
        torch.testing.assert_close(actual, expected, rtol=1.0e-10, atol=1.0e-10)

        loss_weights = torch.linspace(
            0.7, 1.3, actual.numel(), dtype=actual.dtype
        ).reshape_as(actual)
        actual_gradients = torch.autograd.grad(
            torch.sum(actual * loss_weights), (tensor, uv_yx), retain_graph=True
        )
        expected_gradients = torch.autograd.grad(
            torch.sum(expected * loss_weights), (tensor, uv_yx)
        )
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients
        ):
            self.assertTrue(torch.isfinite(actual_gradient).all().item())
            torch.testing.assert_close(
                actual_gradient, expected_gradient, rtol=1.0e-9, atol=1.0e-9
            )


class TestNSSV1PostprocessFixture(unittest.TestCase):
    """CPU checks for deterministic compact test inputs."""

    def test_fixture_does_not_change_global_rng_state(self):
        """Fixture construction should leave the caller's RNG untouched."""

        state_before = torch.random.get_rng_state()

        create_nss_v1_postprocess_case("high", torch.device("cpu"), backend="torch")

        torch.testing.assert_close(torch.random.get_rng_state(), state_before)

    def test_fixture_is_repeatable_for_same_seed(self):
        """A local generator should reproduce every constructed input tensor."""

        first = create_nss_v1_postprocess_case(
            "mid", torch.device("cpu"), backend="torch", seed=47
        )
        second = create_nss_v1_postprocess_case(
            "mid", torch.device("cpu"), backend="torch", seed=47
        )

        for key in first.inputs:
            torch.testing.assert_close(first.inputs[key], second.inputs[key])
        torch.testing.assert_close(first.kpn_params, second.kpn_params)
        torch.testing.assert_close(first.temporal_params, second.temporal_params)

    def test_fixture_derives_quality_dependent_padded_shapes(self):
        """High and half-resolution modes should use their model-derived extents."""

        expected_shapes = {
            "high": ((2, 36, 4, 6), (2, 4, 16, 24), (2, 1, 16, 24)),
            "mid": ((2, 16, 2, 4), (2, 4, 8, 16), (2, 2, 8, 16)),
            "low": ((2, 16, 2, 4), (2, 4, 8, 16), (2, 2, 8, 16)),
        }
        for quality, expected in expected_shapes.items():
            with self.subTest(quality=quality):
                case = create_nss_v1_postprocess_case(
                    quality, torch.device("cpu"), backend="torch"
                )
                self.assertEqual(case.kpn_params.shape, expected[0])
                self.assertEqual(case.temporal_params.shape, expected[1])
                self.assertEqual(case.nearest_depth_offset.shape, expected[2])
                self.assertEqual(case.hr_shape, (2, 3, 32, 48))

    def test_fixture_rounds_noninteger_scale_per_model_contract(self):
        """Scale 1.5 should preserve batch and produce the rounded HR extent."""

        case = create_nss_v1_postprocess_case(
            "low",
            torch.device("cpu"),
            backend="torch",
            scale=1.5,
            lr_shape=(15, 23),
        )

        self.assertEqual(case.hr_shape, (2, 3, 22, 34))
        self.assertEqual(case.inputs["history"].shape, case.hr_shape)
        self.assertEqual(case.temporal_params.shape[-2:], (8, 16))

    def test_fixture_encodes_integer_nearest_depth_patterns(self):
        """Encoded tensors should contain the shader's exact byte patterns."""

        high = create_nss_v1_postprocess_case(
            "high", torch.device("cpu"), backend="torch"
        )
        expected_high_codes = torch.tensor((0.0, 18.0, 36.0))
        actual_high_codes = torch.unique(
            torch.floor(high.nearest_depth_offset * 255.0 + 0.5)
        )
        torch.testing.assert_close(actual_high_codes, expected_high_codes)

        for quality in ("mid", "low"):
            with self.subTest(quality=quality):
                case = create_nss_v1_postprocess_case(
                    quality, torch.device("cpu"), backend="torch"
                )
                actual_bytes = torch.floor(
                    case.nearest_depth_offset[0, :, 0, 0] * 255.0 + 0.5
                )
                torch.testing.assert_close(actual_bytes, torch.tensor((80.0, 250.0)))

    def test_corner_encodings_decode_to_expected_row_and_column_offsets(self):
        """Off-diagonal border codes should retain the shader's y/x bit order."""

        expected_offsets = ((-2, -2), (-2, 2), (2, -2), (2, 2))
        high = create_nss_v1_postprocess_case(
            "high", torch.device("cpu"), backend="torch", lr_shape=(8, 12)
        )
        _set_corner_clamping_offsets(high)
        high_corners = ((0, 0), (0, 11), (7, 0), (7, 11))
        actual_high = tuple(
            _decode_high_offset_row_col(high.nearest_depth_offset[0, 0, row, column])
            for row, column in high_corners
        )
        self.assertEqual(actual_high, expected_offsets)

        expected_packed_offsets = ((-1, -1), (-1, 2), (2, -1), (2, 2))
        low = create_nss_v1_postprocess_case(
            "low", torch.device("cpu"), backend="torch", lr_shape=(8, 12)
        )
        _set_corner_clamping_offsets(low)
        low_corners = ((0, 0), (0, 5), (3, 0), (3, 5))
        actual_low = tuple(
            _decode_packed_nibble_row_col(low.nearest_depth_offset[0, 0, row, column])
            for row, column in low_corners
        )
        self.assertEqual(actual_low, expected_packed_offsets)


class _StageTestCase(unittest.TestCase):
    """Shared builders for literal shader-stage tests."""

    @staticmethod
    def _settings(**overrides: bool) -> _StageFlags:
        values = {
            "preprocess_half_res_input": False,
            "use_sparse_filter_2x2": False,
            "use_history_catmull": False,
            "packed_nearest_offset_quad": False,
            "sharp_theta": False,
        }
        values.update(overrides)
        return _StageFlags(**values)

    @staticmethod
    def _filter_inputs(
        color: torch.Tensor,
        tap_positions_and_channels: tuple[tuple[float, float, float], ...],
        *,
        valid: tuple[float, ...] | None = None,
        tile_offsets: tuple[tuple[float, float], ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        taps = len(tap_positions_and_channels)
        offset_lut = torch.zeros((1, 6, 1, taps), dtype=color.dtype)
        if valid is None:
            valid = (1.0,) * taps
        if tile_offsets is None:
            tile_offsets = ((0.0, 0.0),) * taps
        offset_lut[0, 0, 0] = torch.tensor(
            [offset[0] for offset in tile_offsets], dtype=color.dtype
        )
        offset_lut[0, 1, 0] = torch.tensor(
            [offset[1] for offset in tile_offsets], dtype=color.dtype
        )
        offset_lut[0, 2, 0] = torch.tensor(valid, dtype=color.dtype)
        offset_lut[0, 3:, 0] = torch.tensor(
            tap_positions_and_channels, dtype=color.dtype
        ).T
        idx_modulo = torch.ones((1, 2, 1, 1), dtype=color.dtype)
        return offset_lut, idx_modulo


class TestNSSV1TorchPostprocessStages(_StageTestCase):
    """Coordinate, motion, and variance contracts."""

    def test_reprojection_preserves_shader_fma_residue_at_exact_boundary(self):
        """A non-power-of-two reciprocal must retain Slang's negative FMA residue."""

        inverse_size = torch.tensor((1.0 / 16.0, 1.0 / 24.0), dtype=torch.float32)
        uv = torch.tensor([[[[0.09375, 0.0625]]]], dtype=torch.float32)
        motion = torch.tensor([[[[1.5, 1.5]]]], dtype=torch.float32)

        actual = _reproject_history_uv(uv, motion, inverse_size)

        expected = torch.tensor([[[[0.0, -1.862645149230957e-9]]]], dtype=torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_variance_cancellation_preserves_shader_fma_rounding(self):
        """Moment subtraction must contract to one float32 rounding."""

        m1 = torch.tensor(0.7099450826644897, dtype=torch.float32)
        m2 = torch.tensor(0.5040221214294434, dtype=torch.float32)

        actual = fused_multiply_add(-m1, m1, m2)
        eager = m2 - m1 * m1

        torch.testing.assert_close(
            actual,
            torch.tensor(1.0102995418037608e-7),
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(torch.equal(actual, eager))

    def test_bilinear_rectification_keeps_variance_subtraction_unfused(self):
        """The low-quality shader specialization must retain eager rounding."""

        m1 = torch.tensor([[[[0.7099450826644897]]]], dtype=torch.float32)
        m2 = torch.tensor([[[[0.5040221214294434]]]], dtype=torch.float32)
        warped = m1 + 1.0
        theta = torch.zeros_like(m1)
        gamma = torch.ones_like(m1)

        actual = _rectify_history(
            m1,
            m2,
            warped,
            theta,
            gamma,
            1.0,
            1.0,
            contract_variance=False,
        )
        eager_variance = torch.maximum(torch.abs(m2 - m1 * m1), m1.new_tensor(_EPS))
        expected = m1 + torch.sqrt(eager_variance)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_load_motion_matches_precise_shader_mapping_at_scale_1_5(self):
        """Direct scale division maps output column three to LR column two."""

        motion = torch.zeros((1, 2, 10, 14), dtype=torch.float32)
        motion[:, 1, :, 1] = 1.0
        motion[:, 1, :, 2] = 2.0
        nearest = torch.full((1, 1, 10, 14), 18.0 / 255.0)

        actual = _load_motion(
            motion,
            nearest,
            (15, 21),
            self._settings(),
        )

        expected_scale_x = torch.tensor(21.0) / torch.tensor(14.0)
        torch.testing.assert_close(
            actual[0, 1, 0, 3], 2.0 * expected_scale_x, rtol=0.0, atol=0.0
        )

    def test_high_nearest_offsets_decode_row_then_column(self):
        """High-quality codes should retain signed row/column orientation."""

        encoded = torch.tensor([[[[0.0, 32 / 255, 4 / 255, 36 / 255]]]])
        input_coordinates = torch.tensor([[[[0, 0], [0, 1], [0, 2], [0, 3]]]])

        actual = _decode_nearest_offsets(
            encoded,
            input_coordinates,
            (1, 4),
            self._settings(),
        )

        expected = torch.tensor([[[[-2, -2], [-2, 2], [2, -2], [2, 2]]]])
        torch.testing.assert_close(actual, expected)

    def test_packed_nearest_offsets_select_all_four_quad_lanes(self):
        """Each input parity lane should select its byte and nibble."""

        encoded = torch.tensor([[[[48 / 255]], [[252 / 255]]]])
        input_coordinates = torch.tensor([[[[0, 0], [0, 1]], [[1, 0], [1, 1]]]])

        actual = _decode_nearest_offsets(
            encoded,
            input_coordinates,
            (2, 2),
            self._settings(
                preprocess_half_res_input=True,
                packed_nearest_offset_quad=True,
            ),
        )

        expected = torch.tensor([[[[-1, -1], [2, -1]], [[-1, 2], [2, 2]]]])
        torch.testing.assert_close(actual, expected)

    def test_half_res_nearest_lookup_avoids_padded_storage_texels(self):
        """Half-res addressing should use logical dimensions with padded stride."""

        encoded = torch.full((1, 1, 8, 8), 18 / 255)
        encoded[0, 0, 4, 6] = 36 / 255
        input_coordinates = torch.tensor([[[[9, 13]]]])

        actual = _decode_nearest_offsets(
            encoded,
            input_coordinates,
            (10, 14),
            self._settings(preprocess_half_res_input=True),
        )

        torch.testing.assert_close(actual, torch.tensor([[[[2, 2]]]]))

    def test_motion_coordinates_clamp_offsets_to_every_input_border(self):
        """Decoded offsets beyond each corner should gather its border texel."""

        motion = torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]]]
        )
        encoded = torch.tensor([[[[0 / 255, 32 / 255], [4 / 255, 36 / 255]]]])

        actual = _load_motion(motion, encoded, (2, 2), self._settings())

        torch.testing.assert_close(actual, motion)

    def test_motion_threshold_is_strict_after_per_axis_scaling(self):
        """Exactly 0.1 should be zero while a larger scaled norm survives."""

        motion = torch.tensor(
            [[[[0.05, 0.0500005]], [[0.0, 0.04]]]], dtype=torch.float64
        )
        zero_offset_code = torch.full((1, 1, 1, 2), 18 / 255, dtype=torch.float64)

        actual = _load_motion(
            motion,
            zero_offset_code,
            (2, 6),
            self._settings(),
        )

        torch.testing.assert_close(
            actual[0, :, 0, 0], torch.zeros(2, dtype=actual.dtype)
        )
        torch.testing.assert_close(
            actual[0, :, 0, 3],
            torch.tensor([0.100001, 0.12], dtype=actual.dtype),
            rtol=0.0,
            atol=1.0e-12,
        )


class TestNSSV1TorchPostprocessFilter(_StageTestCase):
    """KPN addressing, filtering, and moment contracts."""

    def test_filter_lut_taps_clamp_to_all_color_borders(self):
        """Out-of-range LUT taps should gather the four LR edge texels."""

        color = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]).expand(-1, 3, -1, -1)
        tap_data = ((-5, -5, 0), (-5, 5, 1), (5, -5, 2), (5, 5, 3))
        kpn_params = torch.ones((1, 4, 1, 1))
        offset_lut, idx_modulo = self._filter_inputs(color, tap_data)

        actual = _filter_color(
            color,
            kpn_params,
            offset_lut,
            idx_modulo,
            torch.ones(1),
            (1, 1),
            (2, 2),
            self._settings(),
        )

        torch.testing.assert_close(actual.m1, torch.full((1, 3, 1, 1), 2.5))

    def test_sparse_kpn_channel_map_prunes_only_the_outer_ring(self):
        """The 6x6 sparse table should map its central 4x4 to channels 0..15."""

        actual = _map_sparse_2x2_kpn_channels(torch.arange(36).reshape(6, 6))

        expected = torch.full((6, 6), -1)
        expected[1:5, 1:5] = torch.arange(16).reshape(4, 4)
        torch.testing.assert_close(actual, expected)

    def test_sparse_filter_assigns_zero_weight_to_pruned_taps(self):
        """An outer-ring sparse tap should not affect a valid central tap."""

        color = torch.tensor([[[[0.0, 2.0, 10.0]]]]).expand(-1, 3, -1, -1)
        tap_data = ((0, 0, 0), (0, 1, 7))
        kpn_params = torch.ones((1, 16, 1, 1))
        offset_lut, idx_modulo = self._filter_inputs(color, tap_data)

        actual = _filter_color(
            color,
            kpn_params,
            offset_lut,
            idx_modulo,
            torch.ones(1),
            (1, 1),
            (1, 3),
            self._settings(use_sparse_filter_2x2=True),
        )

        torch.testing.assert_close(actual.m1, torch.full((1, 3, 1, 1), 10.0))

    def test_invalid_sparse_nan_weight_cannot_leak(self):
        """A pruned NaN KPN lane must contribute an exact zero weight."""

        color = torch.tensor([[[[0.0, 10.0]]]]).expand(-1, 3, -1, -1)
        kpn_params = torch.tensor([[[[float("nan")]], [[1.0]]]])
        offset_lut, idx_modulo = self._filter_inputs(
            color,
            ((0, 0, 35), (0, 1, 8)),
        )

        actual = _filter_color(
            color,
            kpn_params,
            offset_lut,
            idx_modulo,
            torch.ones(1),
            (1, 1),
            (1, 2),
            self._settings(use_sparse_filter_2x2=True),
        )

        self.assertTrue(torch.isfinite(actual.m1).all().item())
        torch.testing.assert_close(actual.m1, torch.full((1, 3, 1, 1), 10.0))

    def test_kpn_coordinates_distinguish_high_from_low_mid_mapping(self):
        """High quality should use temporal scale while low/mid use input division."""

        lr_coordinates = torch.tensor([[[[[1, 1], [3, 3]]]]])

        high = _kpn_coordinates(
            lr_coordinates,
            (4, 4),
            (2, 2),
            (3, 3),
            self._settings(),
        )
        low_mid = _kpn_coordinates(
            lr_coordinates,
            (4, 4),
            (2, 2),
            (3, 3),
            self._settings(preprocess_half_res_input=True),
        )

        torch.testing.assert_close(high, torch.tensor([[[[[1, 1], [1, 1]]]]]))
        torch.testing.assert_close(low_mid, torch.tensor([[[[[0, 0], [1, 1]]]]]))

    def test_filter_selects_center_sample_and_validity(self):
        """Only an exact zero row/column tap should become the center sample."""

        color = torch.tensor([[[[10.0, 20.0, 30.0]]]]).expand(-1, 3, -1, -1)
        tap_data = ((0, -1, 0), (0, 0, 1), (0, 1, 2))
        kpn_params = torch.ones((1, 3, 1, 1))
        offset_lut, idx_modulo = self._filter_inputs(color, tap_data)

        actual = _filter_color(
            color,
            kpn_params,
            offset_lut,
            idx_modulo,
            torch.ones(1),
            (1, 1),
            (1, 3),
            self._settings(),
        )

        expected = torch.tensor([[[[20.0]], [[20.0]], [[20.0]], [[1.0]]]])
        torch.testing.assert_close(actual.center_sample, expected)

    def test_filter_normalizes_first_and_second_moments(self):
        """Moments should divide weighted color sums by their weight sum."""

        color = torch.tensor([[[[1.0, 3.0]]]]).expand(-1, 3, -1, -1)
        kpn_params = torch.tensor([[[[1.0]], [[3.0]]]])
        offset_lut, idx_modulo = self._filter_inputs(
            color,
            ((0, 0, 0), (0, 0, 1)),
            tile_offsets=((0, -1), (0, 0)),
        )

        actual = _filter_color(
            color,
            kpn_params,
            offset_lut,
            idx_modulo,
            torch.ones(1),
            (1, 1),
            (1, 2),
            self._settings(),
        )

        torch.testing.assert_close(actual.m1, torch.full((1, 3, 1, 1), 2.5))
        torch.testing.assert_close(actual.m2, torch.full((1, 3, 1, 1), 7.0))

    def test_filter_clamps_valid_kpn_weights_to_epsilon(self):
        """Weights below EPS should be floored before moment normalization."""

        color = torch.tensor([[[[1.0, 3.0]]]]).expand(-1, 3, -1, -1)
        kpn_params = torch.tensor([[[[-1.0]], [[2.0 * _EPS]]]])
        offset_lut, idx_modulo = self._filter_inputs(
            color,
            ((0, 0, 0), (0, 0, 1)),
            tile_offsets=((0, -1), (0, 0)),
        )

        actual = _filter_color(
            color,
            kpn_params,
            offset_lut,
            idx_modulo,
            torch.ones(1),
            (1, 1),
            (1, 2),
            self._settings(),
        )

        torch.testing.assert_close(
            actual.m1,
            torch.full((1, 3, 1, 1), 7.0 / 3.0),
            rtol=1.0e-6,
            atol=1.0e-6,
        )

    def test_filter_kpn_gradient_splits_tie_at_epsilon(self):
        """Slang max sends half the derivative to a KPN weight equal to EPS."""

        color = torch.tensor([[[[1.0, 3.0]]]]).expand(-1, 3, -1, -1)
        kpn_params = torch.tensor(
            [[[[float(_EPS)]], [[2.0 * _EPS]]]], requires_grad=True
        )
        offset_lut, idx_modulo = self._filter_inputs(
            color,
            ((0, 0, 0), (0, 0, 1)),
            tile_offsets=((0, -1), (0, 0)),
        )

        actual = _filter_color(
            color,
            kpn_params,
            offset_lut,
            idx_modulo,
            torch.ones(1),
            (1, 1),
            (1, 2),
            self._settings(),
        )
        gradient = torch.autograd.grad(actual.m1[0, 0, 0, 0], kpn_params)[0]

        self.assertAlmostEqual(
            gradient[0, 0, 0, 0].item(),
            -2.0 / (9.0 * _EPS),
            delta=0.25,
        )

    def test_filter_differentiability_matches_shader_inputs(self):
        """Filtering should differentiate KPN weights but not no-diff inputs."""

        color = torch.ones((1, 3, 1, 2), requires_grad=True)
        tap_data = ((0, 0, 0), (0, 1, 1))
        kpn_params = torch.tensor([[[[1.0]], [[2.0]]]], requires_grad=True)
        offset_lut, idx_modulo = self._filter_inputs(color, tap_data)
        offset_lut.requires_grad_()
        exposure = torch.ones(1, requires_grad=True)

        actual = _filter_color(
            color,
            kpn_params,
            offset_lut,
            idx_modulo,
            exposure,
            (1, 1),
            (1, 2),
            self._settings(),
        )
        actual.m1.sum().backward()

        self.assertIsNotNone(kpn_params.grad)
        self.assertIsNone(color.grad)
        self.assertIsNone(offset_lut.grad)
        self.assertIsNone(exposure.grad)

    def test_filter_streams_taps_without_passing_a_tap_axis_to_kpn_mapping(self):
        """KPN coordinates should process one vectorized spatial tap at a time."""

        color = torch.ones((1, 3, 2, 2))
        tap_data = ((-1, 0, 0), (0, 0, 1), (1, 0, 2))
        kpn_params = torch.ones((1, 3, 1, 1))
        offset_lut, idx_modulo = self._filter_inputs(color, tap_data)

        with mock.patch(
            "ng_model_gym.usecases.nss.model.torch_postprocess.filter.kpn_coordinates",
            wraps=kpn_coordinates,
        ) as kpn_coordinate_mock:
            _filter_color(
                color,
                kpn_params,
                offset_lut,
                idx_modulo,
                torch.ones(1),
                (2, 2),
                (2, 2),
                self._settings(),
            )

        self.assertEqual(kpn_coordinate_mock.call_count, len(tap_data))
        self.assertTrue(
            all(call.args[0].ndim == 4 for call in kpn_coordinate_mock.call_args_list)
        )


class TestNSSV1TorchPostprocessTemporal(_StageTestCase):
    """Temporal-parameter sampling and rectification contracts."""

    def test_temporal_sampling_ignores_padding_and_applies_transforms(self):
        """Padded params should sample process texels and transform theta/alpha/gamma."""

        temporal = torch.full((1, 3, 2, 4), 99.0)
        temporal[0, :, 0, 0] = torch.tensor([0.25, 0.2, 0.5])
        temporal[0, :, 0, 1] = torch.tensor([0.75, 0.6, 1.0])

        actual = _sample_temporal_params(
            temporal,
            (1, 2),
            (1, 2),
            self._settings(sharp_theta=True),
        )

        torch.testing.assert_close(
            actual.theta, torch.tensor([[[[0.1, 0.9]]]]), atol=1.0e-6, rtol=0.0
        )
        torch.testing.assert_close(
            actual.alpha, torch.tensor([[[[0.12, 0.26]]]]), atol=1.0e-6, rtol=0.0
        )
        torch.testing.assert_close(
            actual.gamma, torch.tensor([[[[1.0, 2.0]]]]), atol=1.0e-6, rtol=0.0
        )

    def test_unsharpened_temporal_theta_preserves_out_of_range_values(self):
        """The unsharpened shader path should return raw learned theta values."""

        temporal = torch.zeros((1, 3, 1, 2))
        temporal[0, 0, 0] = torch.tensor([-0.25, 1.5])

        actual = _sample_temporal_params(
            temporal,
            (1, 2),
            (1, 2),
            self._settings(sharp_theta=False),
        )

        torch.testing.assert_close(
            actual.theta, torch.tensor([[[[-0.25, 1.5]]]]), rtol=0.0, atol=0.0
        )

    def test_temporal_sampling_uses_per_batch_preprocess_dimensions(self):
        """Half-res metadata must not silently reuse batch zero's UV scale."""

        temporal = torch.zeros((2, 3, 1, 4))
        temporal[:, 0, 0] = torch.tensor([0.0, 0.25, 0.5, 0.75])
        preprocess_size = torch.tensor([[1.0, 2.0], [1.0, 4.0]])

        actual = _sample_temporal_params(
            temporal,
            (1, 2),
            preprocess_size,
            self._settings(),
        )

        torch.testing.assert_close(
            actual.theta[:, 0, 0],
            torch.tensor([[0.0, 0.25], [0.125, 0.625]]),
            rtol=0.0,
            atol=1.0e-7,
        )

    def test_rectification_applies_clamp_reset_and_onscreen_gates(self):
        """Reset and onscreen should gate the clamped and raw history blends."""

        m1 = torch.tensor([[[[10.0]]]])
        m2 = torch.tensor([[[[104.0]]]])
        warped = torch.tensor([[[[20.0]]]])
        theta = torch.tensor([[[[0.25]]]])
        gamma = torch.ones_like(theta)

        reset_off = _rectify_history(m1, m2, warped, theta, gamma, 0.0, 1.0)
        offscreen = _rectify_history(m1, m2, warped, theta, gamma, 1.0, 0.0)
        onscreen = _rectify_history(m1, m2, warped, theta, gamma, 1.0, 1.0)

        torch.testing.assert_close(reset_off, torch.tensor([[[[10.0]]]]))
        torch.testing.assert_close(offscreen, torch.tensor([[[[12.0]]]]))
        torch.testing.assert_close(onscreen, torch.tensor([[[[14.0]]]]))

    def test_rectification_treats_reset_as_no_diff(self):
        """Reset should have no gradient while other inputs remain differentiable."""

        differentiable = tuple(
            torch.tensor([[[[value]]]], requires_grad=True)
            for value in (10.0, 104.0, 20.0, 0.25, 1.0)
        )
        reset = torch.tensor(0.5, requires_grad=True)

        actual = _rectify_history(*differentiable, reset, 1.0)
        gradients = torch.autograd.grad(
            actual.sum(), (*differentiable, reset), allow_unused=True
        )

        self.assertTrue(all(gradient is not None for gradient in gradients[:-1]))
        self.assertIsNone(gradients[-1])
