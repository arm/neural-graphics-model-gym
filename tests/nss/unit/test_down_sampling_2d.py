# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from ng_model_gym.nss.model.layers.down_sampling_2d import DownSampling2D


class TestDownSampling2DLayer(unittest.TestCase):
    """Set of unit tests for DownSampling2D in PyTorch."""

    def setUp(self):
        """Set up."""
        self.layer_nearest = DownSampling2D(
            size=(2, 2), interpolation="nearest", antialias=False
        )
        self.layer_bilinear = DownSampling2D(
            size=(2, 2), interpolation="bilinear", antialias=True
        )
        self.layer_bicubic = DownSampling2D(
            size=(2, 2), interpolation="bicubic", antialias=False
        )

    def test_invalid_interpolation_method(self):
        """Test we raise an exception for an invalid interpolation method."""
        with self.assertRaises(ValueError):
            DownSampling2D(size=(2, 2), interpolation="invalid_method")

    def test_output_shape_nearest(self):
        """Test that we produce the correct output shape."""
        x = torch.randn(1, 3, 32, 32)
        y = self.layer_nearest(x)
        expected_shape = (1, 3, 16, 16)
        self.assertEqual(y.shape, expected_shape)

    def test_output_shape_bilinear(self):
        """Test that we produce the correct output shape."""
        x = torch.randn(1, 3, 64, 64)
        y = self.layer_bilinear(x)
        expected_shape = (1, 3, 32, 32)
        self.assertEqual(y.shape, expected_shape)

    def test_output_shape_bicubic(self):
        """Test that we produce the correct output shape."""
        x = torch.randn(1, 3, 40, 40)
        y = self.layer_bicubic(x)
        expected_shape = (1, 3, 20, 20)
        self.assertEqual(y.shape, expected_shape)

    def test_non_square_input(self):
        """Test that the layer works for non-square inputs."""
        x = torch.randn(1, 3, 30, 40)
        y = self.layer_nearest(x)
        expected_shape = (1, 3, 15, 20)
        self.assertEqual(y.shape, expected_shape)

    def test_zero_size_input(self):
        """Test the layer raises an error if we have dimensions."""
        x = torch.randn(1, 3, 0, 16)
        with self.assertRaises(RuntimeError):
            self.layer_nearest(x)

    def test_output_dtype(self):
        """Test that the output dtype matches the input dtype."""
        x = torch.randn(1, 3, 32, 32, dtype=torch.float32)
        y = self.layer_nearest(x)
        self.assertEqual(y.dtype, x.dtype)

    def test_antialias_flag_bilinear(self):
        """Test that the antialias flag is set for bilinear interpolation."""
        layer_with_antialias = DownSampling2D(
            size=(2, 2), interpolation="bilinear", antialias=True
        )
        x = torch.randn(1, 3, 64, 64)
        y = layer_with_antialias(x)
        expected_shape = (1, 3, 32, 32)
        self.assertEqual(y.shape, expected_shape)

    def test_antialias_flag_nearest(self):
        """Test that we don't produce any errors with the antialias
        for the nearest interpolation."""
        layer_with_antialias = DownSampling2D(
            size=(2, 2), interpolation="nearest", antialias=True
        )
        x = torch.randn(1, 3, 64, 64)
        y = layer_with_antialias(x)
        expected_shape = (1, 3, 32, 32)
        self.assertEqual(y.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
