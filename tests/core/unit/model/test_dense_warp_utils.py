# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from ng_model_gym.core.model import bilinear_oob_zero


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
            [[[[2.5, 1.5], [1.75, 1.0]]]],
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
            [[[[0.25, 0.75], [1.0, 2.5]]]],
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

    def test_bilinear_oob_zero_preserves_mixed_precision_dtype(self):
        """Test bilinear_oob_zero with mixed precision tensors."""
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                image = torch.arange(1, 10, dtype=dtype).reshape(1, 1, 3, 3)
                flow = torch.zeros((1, 2, 3, 3), dtype=dtype)

                output = bilinear_oob_zero(image, flow)

                self.assertEqual(output.dtype, dtype)
                torch.testing.assert_close(
                    output.to(torch.float32),
                    image.to(torch.float32),
                    atol=2e-2,
                    rtol=0.0,
                )

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
