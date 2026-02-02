# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from ng_model_gym.core.model import (
    compute_luminance,
    generate_lr_to_hr_lut,
    length,
    lerp,
    normalize_mvs,
    normalize_mvs_fixed,
)


class TestComputeLuminosity(unittest.TestCase):
    """Set of unit tests for compute luminosity in PyTorch."""

    def test_single_image(self):
        """Test single image."""
        # Single image - C, H, W - 3, 2, 2
        image = torch.tensor(
            [
                [[1.0, 0.5], [0.3, 0.2]],
                [[0.5, 1.0], [0.7, 0.6]],
                [[0.2, 0.4], [0.9, 0.3]],
            ]
        )

        expected_luminance = torch.tensor([[0.5846, 0.8504], [0.6294, 0.4933]])

        result = compute_luminance(image)

        self.assertTrue(torch.allclose(result, expected_luminance, atol=1e-4))

    def test_batch_of_images(self):
        """Test a batch of images."""
        # Batch of 2 images
        # Manually computed:
        # Example, 2, 2: 0.2*0.2126+0.6*0.7152+0.3*0.0722 = 0.4933
        # Example, 0, 0: 1.0*0.2126+0.5*0.7152+0.2*0.0722 = 0.5846
        images = torch.tensor(
            [
                [
                    [[1.0, 0.5], [0.3, 0.2]],
                    [[0.5, 1.0], [0.7, 0.6]],
                    [[0.2, 0.4], [0.9, 0.3]],
                ],
                [
                    [[0.6, 0.7], [0.8, 0.9]],
                    [[0.5, 0.4], [0.3, 0.2]],
                    [[0.1, 0.2], [0.3, 0.4]],
                ],
            ]
        )

        expected_luminance = torch.tensor(
            [
                [[[0.5846, 0.8504], [0.6294, 0.4933]]],
                [[[0.4924, 0.4493], [0.4063, 0.3633]]],
            ]
        )

        result = compute_luminance(images)

        self.assertTrue(torch.allclose(result, expected_luminance, atol=1e-4))

    def test_invalid_shape(self):
        """Test invalid shape."""
        image = torch.rand(5, 5)
        with self.assertRaises(ValueError):
            compute_luminance(image)


class TestLengthFunction(unittest.TestCase):
    """Set of unit tests for length function in PyTorch."""

    def test_length_positive_values(self):
        """Test length of positive values."""
        vector = torch.tensor([[3.0, 4.0]])
        result = length(vector)
        expected = torch.tensor([[5.0]])
        self.assertTrue(
            torch.allclose(result, expected), f"Expected {expected}, but got {result}"
        )

    def test_length_zero_vector(self):
        """Test length of zero vector."""
        vector = torch.tensor([[0.0, 0.0]])
        result = length(vector)
        expected = torch.tensor([[0.0]])
        self.assertTrue(
            torch.allclose(result, expected), f"Expected {expected}, but got {result}"
        )

    def test_length_negative_values(self):
        """Test length of negative values."""
        vector = torch.tensor([[-3.0, -4.0]])
        result = length(vector)
        expected = torch.tensor([[5.0]])
        self.assertTrue(
            torch.allclose(result, expected), f"Expected {expected}, but got {result}"
        )

    def test_length_batch_vectors(self):
        """Test length of a batch of vector."""
        # Batch vector
        vector = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
        result = length(vector)
        expected = torch.tensor([[5.0], [10.0]])
        self.assertTrue(
            torch.allclose(result, expected), f"Expected {expected}, but got {result}"
        )


class TestLerpFunction(unittest.TestCase):
    """Set of unit tests for linear interpolation function in PyTorch."""

    def test_lerp_zero(self):
        """Test Lerp function with zero starting value."""
        x = torch.tensor([[1.0, 2.0]])
        y = torch.tensor([[3.0, 4.0]])
        a = torch.tensor([0.0])
        result = lerp(x, y, a)
        expected = x
        self.assertTrue(
            torch.allclose(result, expected), f"Expected {expected}, but got {result}"
        )

    def test_lerp_one(self):
        """Test Lerp function with one starting value."""
        x = torch.tensor([[1.0, 2.0]])
        y = torch.tensor([[3.0, 4.0]])
        a = torch.tensor([1.0])
        result = lerp(x, y, a)
        expected = y
        self.assertTrue(
            torch.allclose(result, expected), f"Expected {expected}, but got {result}"
        )

    def test_lerp_less_than_one(self):
        """Test Lerp function with 0.3 starting value."""
        x = torch.tensor([[1.0, 2.0]])
        y = torch.tensor([[3.0, 4.0]])
        a = torch.tensor([0.3])
        result = lerp(x, y, a)
        expected = torch.tensor([[1.6, 2.6]])
        self.assertTrue(
            torch.allclose(result, expected), f"Expected {expected}, but got {result}"
        )


class TestNormalizeMVSFunctions(unittest.TestCase):
    """Set of unit tests for Normalize MVS functions in PyTorch."""

    def test_normalize_mvs_fixed(self):
        """Test Normalize MVS fixed."""
        # Shape of the tensor (1, 2, 1, 1)
        mvs = torch.tensor([[[[270.0]], [[480.0]]]])
        result = normalize_mvs_fixed(mvs)
        expected = torch.tensor([[[[0.5]], [[0.5]]]])
        self.assertEqual(mvs.shape, expected.shape)
        self.assertTrue(
            torch.allclose(result, expected), f"Expected {expected}, but got {result}"
        )

    def test_normalize_mvs(self):
        """Test Normalize MVS."""
        mvs = torch.tensor([[[[300.0, 300.0]], [[400.0, 400.0]]]])
        result = normalize_mvs(mvs)
        expected = torch.tensor([[[[300.0, 300.0]], [[200.0, 200.0]]]])
        self.assertTrue(
            torch.allclose(result, expected), f"Expected {expected}, but got {result}"
        )

    def test_normalize_mvs_batch(self):
        """Test Normalize MVS on batched tensor."""
        mvs = torch.tensor([[[[600.0]], [[800.0]]], [[[1200.0]], [[1600.0]]]])
        result = normalize_mvs(mvs)
        expected = torch.tensor([[[[600.0]], [[800.0]]], [[[1200.0]], [[1600.0]]]])
        self.assertTrue(
            torch.allclose(result, expected), f"Expected {expected}, but got {result}"
        )


class TestLRToHRLookupTable(unittest.TestCase):
    """Tests for LR to HR LUT."""

    def setUp(self):
        """Set up"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = 4
        self.scale = float(2)
        self.jitter = torch.rand(self.batch_size, 2, 1, 1, device=self.device)

        self.SCALES = {
            2.0: ((1, 1), (2, 2), 2),
            1.5: ((2, 2), (3, 3), 3),
            1.3: ((3, 3), (4, 4), 4),
        }

    def test_unsupported_scale(self):
        """Test unsupported scale value."""

        scale = 1.7

        with self.assertRaises(ValueError):
            generate_lr_to_hr_lut(scale, self.jitter)

    def test_output_for_different_scale_values(self):
        """Test offset shape and idx_modulo for valid scale values."""

        for scale, ((H_lr, W_lr), (H_hr, W_hr), idx_mod) in self.SCALES.items():
            jitter = torch.rand(self.batch_size, 2, H_lr, W_lr, device=self.device)

            offset_lut, idx_modulo = generate_lr_to_hr_lut(scale, jitter)

            # shape of offset_lut should be (N, 4, H_hr, W_hr) where N = batch_size
            self.assertEqual(
                offset_lut.shape, torch.Size([self.batch_size, 4, H_hr, W_hr])
            )
            self.assertEqual(idx_modulo, torch.tensor(idx_mod, device=self.device))

    def test_offset_lut_dy_dx(self):
        """Test vertical and horizontal offset values."""

        offset_lut, _ = generate_lr_to_hr_lut(self.scale, self.jitter)

        dy = offset_lut[:, 0]
        dx = offset_lut[:, 1]

        self.assertTrue(torch.all((dy == 0.0) | (dy == 1.0) | (dy == -1.0)))
        self.assertTrue(torch.all((dx == 0.0) | (dx == 1.0) | (dx == -1.0)))

    def test_offset_lut_mask(self):
        """Test offset mask is 0 or 1."""

        offset_lut, _ = generate_lr_to_hr_lut(self.scale, self.jitter)

        mask_values = offset_lut[:, 2]

        self.assertTrue(torch.all((mask_values == 0.0) | (mask_values == 1.0)))

    def test_dy_dx_values_for_mask(self):
        """Test dy and dx are only non zero when mask is 1."""

        offset_lut, _ = generate_lr_to_hr_lut(self.scale, self.jitter)

        mask_true = offset_lut[:, 2] == 1.0
        mask_false = offset_lut[:, 2] == 0.0

        dy = offset_lut[:, 0]
        dx = offset_lut[:, 1]

        expected_dy_dx = torch.tensor([-1.0, 0.0, 1.0], device=self.device)

        self.assertTrue(torch.all(torch.isin(dy[mask_true], expected_dy_dx)))
        self.assertTrue(torch.all(torch.isin(dx[mask_true], expected_dy_dx)))

        zero_dy_dx = torch.tensor([0.0], device=self.device)

        self.assertTrue(torch.all(torch.isin(dy[mask_false], zero_dy_dx)))
        self.assertTrue(torch.all(torch.isin(dx[mask_false], zero_dy_dx)))

    def test_padded_to_zeros(self):
        """Test offset is padded to 4 channels with 0s."""

        offset_lut, _ = generate_lr_to_hr_lut(self.scale, self.jitter)

        padding = offset_lut[:, 3]

        self.assertTrue(torch.all(padding == 0.0))
