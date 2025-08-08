# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from ng_model_gym.nss.model.layers.dense_warp import DenseWarp


class TestDenseWarp(unittest.TestCase):
    """Set of unit tests for DenseWarp layer that is defined in dense_warp.py"""

    valid_interpolations = ["bilinear_oob_zero", "bilinear"]

    def test_initialization(self):
        """Test that the DenseWarp layer initializes with valid interpolation methods."""
        for interp in TestDenseWarp.valid_interpolations:
            layer = DenseWarp(interpolation=interp)
            self.assertEqual(layer.interpolation, interp)

    def test_invalid_initialization(self):
        """Test that an error is raised if invalid interpolation method is provided."""
        with self.assertRaises(ValueError):
            DenseWarp(interpolation="invalid_method")

    def test_forward_call_shape(self):
        """Test the DenseWarp 'forward' method returns correct tensor with correct shape."""
        batch_size, channels, height, width = 1, 3, 10, 10
        frame = torch.rand((batch_size, channels, height, width))
        flow_vectors = torch.rand((batch_size, 2, height, width))
        for interp in TestDenseWarp.valid_interpolations:
            layer = DenseWarp(interpolation=interp)
            warped_frame = layer([frame, flow_vectors])
            self.assertEqual(warped_frame.shape, (batch_size, channels, height, width))

    def test_forward_value_oob_zero(self):
        """Test the DenseWarp 'forward' method returns correct tensor
        with correct shape and values."""
        batch_size, height, width, channels = 1, 3, 4, 1
        # Shape: (1, 1, 3, 4)
        frame = torch.tensor(
            [
                [
                    [[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7], [0.8, 0.9, 0.1, 0.11]],
                ]
            ]
        )
        flow_vectors = torch.ones((batch_size, 2, height, width))
        # Shape: (1, 1, 3, 4)
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.0000, 0.0000, 0.0000],
                        [0.0000, 0.0000, 0.1000, 0.2000],
                        [0.0000, 0.4000, 0.5000, 0.6000],
                    ]
                ]
            ]
        )

        layer = DenseWarp(interpolation="bilinear_oob_zero")
        warped_frame = layer([frame, flow_vectors])
        self.assertEqual(warped_frame.shape, (batch_size, channels, height, width))

        self.assertTrue(torch.allclose(warped_frame, expected_output, atol=1e-4))

    def test_forward_value_bilinear(self):
        """Test the DenseWarp 'forward' method returns correct tensor
        with correct shape and values."""
        batch_size, height, width, channels = 1, 3, 4, 1
        # Shape: (1, 1, 3, 4)
        frame = torch.tensor(
            [
                [
                    [[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7], [0.8, 0.9, 0.1, 0.11]],
                ]
            ]
        )
        flow_vectors = torch.ones((batch_size, 2, height, width))
        # Shape: (1, 1, 3, 4)
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.0000, 0.1000, 0.2000],
                        [0.0000, 0.0000, 0.1000, 0.2000],
                        [0.4000, 0.4000, 0.5000, 0.6000],
                    ]
                ]
            ]
        )

        layer = DenseWarp(interpolation="bilinear")
        warped_frame = layer([frame, flow_vectors])
        self.assertEqual(warped_frame.shape, (batch_size, channels, height, width))

        self.assertTrue(torch.allclose(warped_frame, expected_output, atol=1e-4))

    def test_forward_value_nearest(self):
        """Test the DenseWarp 'forward' method returns correct tensor
        with correct shape and values."""
        batch_size, height, width, channels = 1, 3, 4, 2
        # Shape: (1, 2, 3, 4)
        frame = torch.tensor(
            [
                [
                    [[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7], [0.8, 0.9, 0.1, 0.11]],
                    [[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7], [0.8, 0.9, 0.1, 0.11]],
                ]
            ]
        )
        flow_vectors = torch.ones((batch_size, 2, height, width))
        # Shape: (1, 2, 3, 4)
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0000, 0.0000, 0.1000, 0.2000],
                        [0.0000, 0.0000, 0.1000, 0.2000],
                        [0.4000, 0.4000, 0.5000, 0.6000],
                    ],
                    [
                        [0.0000, 0.0000, 0.1000, 0.2000],
                        [0.0000, 0.0000, 0.1000, 0.2000],
                        [0.4000, 0.4000, 0.5000, 0.6000],
                    ],
                ]
            ]
        )

        layer = DenseWarp(interpolation="nearest")
        warped_frame = layer([frame, flow_vectors])
        self.assertEqual(warped_frame.shape, (batch_size, channels, height, width))

        self.assertTrue(torch.allclose(warped_frame, expected_output, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
