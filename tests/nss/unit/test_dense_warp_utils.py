# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from ng_model_gym.core.model.dense_warp_utils import (
    bilinear_oob_zero,
    interpolate_bilinear,
    interpolate_bilinear_w_zero_pad,
)


class TestInterpolateBilinear(unittest.TestCase):
    """Set of unit tests for interpolate_bilinear in PyTorch."""

    def test_interpolate_bilinear_simple(self):
        """Test the interpolate_bilinear."""
        # Shape: (1, 1, 2, 2) -> [batch_size, channels, height, width]
        grid = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        # Shape: (1, 2, 2) -> [batch_size, 2, num_points]
        query_points = torch.tensor([[[0.5, 1.5], [0.5, 0.5]]])
        # Expected output shape: (1, 1, 2) -> [batch_size, channels, num_points]
        expected_output = torch.tensor([[[2.5, 3.5]]])

        output = interpolate_bilinear(grid, query_points)

        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

    def test_interpolate_bilinear_invalid_grid_dim(self):
        """Test interpolate_bilinear with an invalid grid dimension."""
        grid = torch.zeros((2, 2, 1))
        query_points = torch.zeros((2, 1, 2))
        with self.assertRaises(ValueError):
            interpolate_bilinear(grid, query_points)

    def test_interpolate_bilinear_invalid_grid_height(self):
        """Test interpolate_bilinear with an invalid grid height."""
        grid = torch.zeros((1, 1, 1, 2))
        query_points = torch.zeros((2, 1, 2))
        with self.assertRaises(ValueError):
            interpolate_bilinear(grid, query_points)

    def test_interpolate_bilinear_invalid_grid_width(self):
        """Test interpolate_bilinear with an invalid grid width."""
        grid = torch.zeros((1, 1, 2, 1))
        query_points = torch.zeros((2, 1, 2))
        with self.assertRaises(ValueError):
            interpolate_bilinear(grid, query_points)

    def test_interpolate_bilinear_invalid_query_points_dim(self):
        """Test interpolate_bilinear with an invalid query points dimensions."""
        grid = torch.zeros((1, 1, 2, 2))
        query_points = torch.zeros((1, 2))
        with self.assertRaises(ValueError):
            interpolate_bilinear(grid, query_points)

    def test_interpolate_bilinear_invalid_query_points(self):
        """Test interpolate_bilinear with an invalid query points."""
        grid = torch.zeros((1, 1, 2, 2))
        query_points = torch.zeros((1, 1, 1))
        with self.assertRaises(ValueError):
            interpolate_bilinear(grid, query_points)

    def test_interpolate_bilinear_center_point(self):
        """Check interpolation for the center point of the patch."""
        # Shape: (1, 1, 2, 2)
        grid = torch.tensor(
            [[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=torch.float32
        ).permute(0, 3, 1, 2)
        query_points = torch.tensor([[[0.5, 0.5]]], dtype=torch.float32).permute(
            0, 2, 1
        )
        # Bilinear interpolation should be (1+2+3+4)/4 = 2.5
        expected_output = torch.tensor([[[[2.5]]]], dtype=torch.float32)
        output = interpolate_bilinear(grid, query_points)
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

    def test_interpolate_bilinear(self):
        """Test the interpolate_bilinear function."""
        batch_size, height, width = 1, 3, 4
        # Shape: (1, 1, 3, 4)
        grid = torch.tensor(
            [
                [
                    [[0.0], [0.1], [0.2], [0.3]],
                    [[0.4], [0.5], [0.6], [0.7]],
                    [[0.8], [0.9], [0.1], [0.11]],
                ]
            ]
        ).permute(0, 3, 1, 2)
        # Shape: (1, 2, 12)
        query_points = torch.ones((batch_size, 2, height * width))
        # Shape: (1, 12, 1)
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.5],
                        [0.5],
                        [0.5],
                        [0.5],
                        [0.5],
                        [0.5],
                        [0.5],
                        [0.5],
                        [0.5],
                        [0.5],
                        [0.5],
                        [0.5],
                    ]
                ]
            ]
        )
        output = interpolate_bilinear(grid, query_points)
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

    def test_interpolate_bilinear_exact_points(self):
        """Test interpolate_bilinear at exact points."""
        # Shape: (1, 1, 2, 2)
        grid = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]).permute(0, 3, 1, 2)
        # Shape: (1, 2, 2)
        query_points = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]).permute(0, 2, 1)
        # Shape: (1, 2, 1)
        expected_output = torch.tensor([[[1.0, 4.0]]])
        output = interpolate_bilinear(grid, query_points)
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

    def test_interpolate_bilinear_out_of_bounds(self):
        """Test interpolate_bilinear for out of bounds query points."""
        # Shape: (1, 1, 2, 2)
        grid = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]).permute(0, 3, 1, 2)
        # Shape: (1, 2, 2)
        query_points = torch.tensor([[[-2.0, -2.0], [2.0, 2.0]]]).permute(0, 2, 1)
        # Shape: (1, 2, 1)
        expected_output = torch.tensor([[[1.0, 4.0]]])
        output = interpolate_bilinear(grid, query_points)
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))


class TestInterpolateBilinearZeroPad(unittest.TestCase):
    """Set of unit tests for interpolate_bilinear_w_zero_pad in PyTorch."""

    def test_interpolate_bilinear_w_zero_pad(self):
        """Test the interpolate_bilinear_w_zero_pad with zero padding."""
        # Shape: (1, 1, 2, 2) -> [batch_size, channels, height, width]
        grid = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        # Shape: (1, 2, 2) -> [batch_size, num_points, 2]
        query_points = torch.tensor([[[0.5, 0.5], [1.5, 1.5]]])
        # Expected output shape: (1, 1, 2) -> [batch_size, channels, num_points]
        expected_output = torch.tensor([[[2.5, 4.0]]])

        output = interpolate_bilinear_w_zero_pad(grid, query_points, indexing="ij")

        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

    def test_interpolate_bilinear_indexing_modes(self):
        """Test the interpolate_bilinear_w_zero_pad with different indexing modes."""
        # Shape: (1, 1, 2, 2) -> [batch_size, channels, height, width]
        grid = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        query_points = torch.tensor([[[0.5, 1.0], [1.5, 0.0]]])

        output_ij = interpolate_bilinear_w_zero_pad(grid, query_points, indexing="ij")
        output_xy = interpolate_bilinear_w_zero_pad(grid, query_points, indexing="xy")

        self.assertFalse(torch.allclose(output_ij, output_xy, atol=1e-4))

    def test_interpolate_bilinear_w_zero_pad_invalid_grid_dim(self):
        """Test interpolate_bilinear_w_zero_pad with an invalid grid dimension."""
        grid = torch.zeros((2, 2, 1))
        query_points = torch.zeros((2, 1, 2))
        with self.assertRaises(ValueError):
            interpolate_bilinear_w_zero_pad(grid, query_points)

    def test_interpolate_bilinear_w_zero_pad_invalid_grid_height(self):
        """Test interpolate_bilinear_w_zero_pad with an invalid grid height."""
        grid = torch.zeros((1, 1, 1, 2))
        query_points = torch.zeros((2, 1, 2))
        with self.assertRaises(ValueError):
            interpolate_bilinear_w_zero_pad(grid, query_points)

    def test_interpolate_bilinear_w_zero_pad_invalid_grid_width(self):
        """Test interpolate_bilinear_w_zero_pad with an invalid grid width."""
        grid = torch.zeros((1, 1, 2, 1))
        query_points = torch.zeros((2, 1, 2))
        with self.assertRaises(ValueError):
            interpolate_bilinear_w_zero_pad(grid, query_points)

    def test_interpolate_bilinear_w_zero_pad_invalid_query_points_dim(self):
        """Test interpolate_bilinear_w_zero_pad with an invalid query points dimensions."""
        grid = torch.zeros((1, 1, 2, 2))
        query_points = torch.zeros((1, 2))
        with self.assertRaises(ValueError):
            interpolate_bilinear_w_zero_pad(grid, query_points)

    def test_interpolate_bilinear_w_zero_pad_invalid_query_points(self):
        """Test interpolate_bilinear_w_zero_pad with an invalid query points."""
        grid = torch.zeros((1, 1, 2, 2))
        query_points = torch.zeros((1, 1, 1))
        with self.assertRaises(ValueError):
            interpolate_bilinear_w_zero_pad(grid, query_points)

    def test_interpolate_bilinear_w_zero_pad_invalid_indexing(self):
        """Test interpolate_bilinear_w_zero_pad with an invalid indexing argument."""
        grid = torch.zeros((2, 1, 2, 2))
        query_points = torch.zeros((2, 1, 2))
        with self.assertRaises(ValueError):
            interpolate_bilinear_w_zero_pad(grid, query_points, indexing="invalid")


class TestBilinearOobZero(unittest.TestCase):
    """Set of unit tests for bilinear_oob_zero in PyTorch."""

    def test_bilinear_oob_zero(self):
        """Test the bilinear_oob_zero."""
        b, h, w, c, f = (1, 4, 4, 1, 2)
        # Flow tensor with shape [batch_size, 2, height, width]
        flow = torch.ones((b, f, h, w), dtype=torch.float32)
        # Image tensor with shape [batch_size, channels, height, width]
        image = (
            torch.linspace(0, h * w * c, steps=h * w * c, dtype=torch.float32)
            .reshape(b, h, w, c)
            .permute((0, 3, 1, 2))
        )
        # Expected output with shape [batch_size, channels, height, width] - (1, 1, 4, 4)
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0667, 2.1333],
                        [0.0, 4.2667, 5.3333, 6.4000],
                        [0.0, 8.5333, 9.6000, 10.6667],
                    ]
                ]
            ],
            dtype=torch.float32,
        )
        output = bilinear_oob_zero(image, flow)
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))

    def test_bilinear_oob_zero_oob_negative(self):
        """Test the bilinear_oob_zero when the location is out of the image on negative flow."""
        b, h, w, c = (1, 2, 2, 1)
        # Flow tensor with shape [batch_size, 2, height, width]
        flow = torch.tensor(
            [[[[-0.5, -0.5], [-0.5, -0.5]], [[-0.5, -0.5], [-0.5, -0.5]]]]
        )
        # Image tensor with shape [batch_size, channels, height, width]
        image = torch.linspace(
            1, h * w * c, steps=h * w * c, dtype=torch.float32
        ).reshape(b, c, h, w)
        expected_output = torch.tensor(
            [[[[2.5, 3.0], [3.5, 4.0]]]],
            dtype=torch.float32,
        )
        output = bilinear_oob_zero(image, flow)
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6))

    def test_bilinear_oob_zero_oob_positive(self):
        """Test the bilinear_oob_zero when the location is out of the image on positive flow."""
        b, h, w, c = (1, 2, 2, 1)
        # Flow tensor with shape [batch_size, 2, height, width]
        flow = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]]])
        # Image tensor with shape [batch_size, channels, height, width]
        image = torch.linspace(
            1, h * w * c, steps=h * w * c, dtype=torch.float32
        ).reshape(b, c, h, w)
        expected_output = torch.tensor(
            [[[[0.0, 0.0], [0.0, 2.5]]]],
            dtype=torch.float32,
        )
        output = bilinear_oob_zero(image, flow)
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6))

    def test_bilinear_oob_zero_multichannel(self):
        """Test the bilinear_oob_zero with multiple channels."""
        b, h, w, c, f = (1, 3, 3, 3, 2)
        # Flow tensor with shape [batch_size, 2, height, width]
        flow = torch.ones((b, f, h, w), dtype=torch.float32)
        # Image tensor with shape [batch_size, channels, height, width]
        image = torch.linspace(
            1, h * w * c, steps=h * w * c, dtype=torch.float32
        ).reshape(b, c, h, w)
        output = bilinear_oob_zero(image, flow)
        expected_output = torch.tensor(
            [
                [
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 4.0, 5.0]],
                    [[0.0, 0.0, 0.0], [0.0, 10.0, 11.0], [0.0, 13.0, 14.0]],
                    [[0.0, 0.0, 0.0], [0.0, 19.0, 20.0], [0.0, 22.0, 23.0]],
                ]
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6))

    def test_bilinear_oob_zero_multibatch(self):
        """Test the bilinear_oob_zero with multiple batches."""
        b, h, w, c, f = (2, 3, 3, 1, 2)
        # Flow tensor with shape [batch_size, 2, height, width]
        flow = torch.ones((b, f, h, w), dtype=torch.float32)
        # Image tensor with shape [batch_size, channels, height, width]
        image = torch.linspace(
            1, h * w * b, steps=h * w * b, dtype=torch.float32
        ).reshape(b, c, h, w)
        output = bilinear_oob_zero(image, flow)
        expected_output = torch.tensor(
            [
                [[[0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 4.0, 5.0]]],
                [[[0.0, 0.0, 0.0], [0.0, 10.0, 11.0], [0.0, 13.0, 14.0]]],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6))

    def test_bilinear_oob_zero_exact_point(self):
        """Test the bilinear_oob_zero."""
        b, h, w, c, f = (1, 4, 4, 1, 2)
        # Flow tensor with shape [batch_size, 2, height, width]
        flow = torch.ones((b, f, h, w), dtype=torch.float32) * 2
        # Image tensor with shape [batch_size, channels, height, width]
        image = torch.linspace(
            1, h * w * c, steps=h * w * c, dtype=torch.float32
        ).reshape(b, c, h, w)
        output = bilinear_oob_zero(image, flow)
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 2.0],
                        [0.0, 0.0, 5.0, 6.0],
                    ]
                ]
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6))

    def test_bilinear_oob_zero_mix(self):
        """Test the bilinear_oob_zero."""
        b, h, w, c, f = (2, 3, 3, 2, 2)
        # Flow tensor with shape [batch_size, 2, height, width]
        flow = torch.ones((b, f, h, w), dtype=torch.float32)
        # Image tensor with shape [batch_size, channels, height, width]
        image = torch.linspace(
            0, b * h * w * c, steps=b * h * w * c, dtype=torch.float32
        ).reshape(b, c, h, w)
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.0000, 0.0000],
                        [0.0000, 0.0000, 1.0286],
                        [0.0000, 3.0857, 4.1143],
                    ],
                    [
                        [0.0000, 0.0000, 0.0000],
                        [0.0000, 9.2571, 10.2857],
                        [0.0000, 12.3429, 13.3714],
                    ],
                ],
                [
                    [
                        [0.0000, 0.0000, 0.0000],
                        [0.0000, 18.5143, 19.5429],
                        [0.0000, 21.6000, 22.6286],
                    ],
                    [
                        [0.0000, 0.0000, 0.0000],
                        [0.0000, 27.7714, 28.8000],
                        [0.0000, 30.8571, 31.8857],
                    ],
                ],
            ],
            dtype=torch.float32,
        )
        output = bilinear_oob_zero(image, flow)
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
