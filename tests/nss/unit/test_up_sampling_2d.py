# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from ng_model_gym.nss.model.layers.up_sampling_2d import UpSampling2D


class TestUpSampling2D(unittest.TestCase):
    """Set of unit tests for UpSampling2D in PyTorch."""

    def setUp(self):
        """Setup"""
        self.layer_nearest = UpSampling2D(size=(2, 2), interpolation="nearest")
        self.layer_bilinear = UpSampling2D(size=(2, 2), interpolation="bilinear")
        self.layer_bicubic = UpSampling2D(size=(2, 2), interpolation="bicubic")

    def test_invalid_interpolation_method(self):
        """Test if the layer raises an exception for invalid interpolation method"""
        with self.assertRaises(ValueError):
            UpSampling2D(size=(2, 2), interpolation="invalid_method")

    def test_output_shape_nearest(self):
        """Test that we produce the correct output shape"""
        x = torch.randn(1, 3, 16, 16)
        y = self.layer_nearest(x)
        # by 2x
        expected_shape = (1, 3, 32, 32)
        self.assertEqual(y.shape, expected_shape)

    def test_output_shape_bilinear(self):
        """Test that we produce the correct output shape"""
        x = torch.randn(1, 3, 16, 16)
        y = self.layer_bilinear(x)
        expected_shape = (1, 3, 32, 32)
        self.assertEqual(y.shape, expected_shape)

    def test_output_shape_bicubic(self):
        """Test that we produce the correct output shape"""
        x = torch.randn(1, 3, 16, 16)
        y = self.layer_bicubic(x)
        expected_shape = (1, 3, 32, 32)
        self.assertEqual(y.shape, expected_shape)

    def test_non_square_input(self):
        """Test that we can have the non-square inputs"""
        x = torch.randn(1, 3, 20, 30)
        y = self.layer_nearest(x)
        expected_shape = (1, 3, 40, 60)
        self.assertEqual(y.shape, expected_shape)

    def test_input_with_zero_dim(self):
        """Test what if we have 0 dimension"""
        x = torch.randn(1, 3, 0, 16)
        with self.assertRaises(RuntimeError):
            self.layer_nearest(x)

    def test_output_dtype(self):
        """Test that the output type is the same as the input type"""
        x = torch.randn(1, 3, 16, 16, dtype=torch.float32)
        y = self.layer_nearest(x)
        self.assertEqual(y.dtype, x.dtype)

    def test_training_nearest_mode(self):
        """Test the training mode."""
        x = torch.randn(1, 3, 16, 16)
        self.layer_nearest.train()
        y = self.layer_nearest(x)
        expected_shape = (1, 3, 32, 32)
        self.assertEqual(y.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
