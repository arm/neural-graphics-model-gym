# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from ng_model_gym.nss.model.layers.zero_upsampling import ZeroUpsample


class TestZeroUpsample(unittest.TestCase):
    """Set of unit tests for ZeroUSample in PyTorch."""

    def setUp(self):
        """Set up."""
        self.layer = ZeroUpsample(scale=(2.0, 2.0))

    def test_no_jitter(self):
        """Test with no jitter"""
        b, _, _, _ = 1, 1, 2, 2  # Tensor shape [b, c, ih, iw]
        ten_in = torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.float32)
        # No jitter
        jitter = torch.zeros(b, 2)

        # Expected output is a 4x4 tensor without jitter
        expected_output = torch.tensor(
            [[[[0, 0, 0, 0], [0, 1, 0, 2], [0, 0, 0, 0], [0, 3, 0, 4]]]],
            dtype=torch.float32,
        )

        output = self.layer((ten_in, jitter))

        # Check the output
        self.assertTrue(
            torch.equal(output, expected_output),
            "Output does not match the expected no-jitter result",
        )

    def test_with_jitter(self):
        """Test with a jitter value of 0.2 for both x and y."""
        # Tensor shape [b, c, ih, iw] = [1, 1, 2, 2]
        ten_in = torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.float32)
        # Small jitter
        jitter = torch.tensor([[-0.2, -0.2]])

        # Expected output is still the same because of floor operation
        expected_output = torch.tensor(
            [[[[1, 0, 2, 0], [0, 0, 0, 0], [3, 0, 4, 0], [0, 0, 0, 0]]]],
            dtype=torch.float32,
        )

        output = self.layer((ten_in, jitter))

        self.assertTrue(
            torch.equal(output, expected_output),
            "Output does not match the expected jittered result",
        )

    def test_different_jitter_per_batch(self):
        """Test with different jitter values for each batch."""
        # Tensor shape [b, c, ih, iw] = [2, 1, 2, 2]
        ten_in = torch.tensor(
            [[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]], dtype=torch.float32
        )
        jitter = torch.tensor([[0.9, 0.2], [0.0, 0.0]])

        expected_output = torch.tensor(
            [
                [[[0, 0, 0, 0], [0, 0, 0, 2], [0, 1, 0, 0], [0, 3, 0, 4]]],
                [[[0, 0, 0, 0], [0, 0, 0, 6], [0, 5, 0, 0], [0, 7, 0, 8]]],
            ],
            dtype=torch.float32,
        )

        output = self.layer((ten_in, jitter))

        self.assertTrue(
            torch.equal(output, expected_output),
            "Output does not match expected result with different jitter per batch",
        )


if __name__ == "__main__":
    unittest.main()
