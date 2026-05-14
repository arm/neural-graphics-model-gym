# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from ng_model_gym.usecases.nss.model.model_blocks_v1 import (
    AutoEncoderV1,
    get_kpn_prune_indices,
)


class TestNSSV1AutoEncoder(unittest.TestCase):
    """Tests for the NSS v1 AutoEncoderV1 block."""

    def test_output_shapes(self):
        """Test AutoEncoderV1 output shapes for the high-quality KPN size."""

        autoencoder = AutoEncoderV1(kpn_size=(6, 6))
        x = torch.randn(2, 12, 128, 128)

        kpn_params, temporal_params = autoencoder(x)

        self.assertEqual(kpn_params.shape, (2, 36, 32, 32))
        self.assertEqual(temporal_params.shape, (2, 4, 128, 128))

    def test_get_kpn_prune_indices_returns_centered_column_major_indices(self):
        """Test centered pruning indices preserve column-major tap ordering."""

        indices = get_kpn_prune_indices(source_size=(6, 6), target_size=(4, 4))

        self.assertEqual(
            indices,
            (
                7,
                8,
                9,
                10,
                13,
                14,
                15,
                16,
                19,
                20,
                21,
                22,
                25,
                26,
                27,
                28,
            ),
        )

    def test_get_kpn_prune_indices_rejects_unsupported_mode(self):
        """Test only centered KPN pruning is supported."""

        with self.assertRaisesRegex(ValueError, "Unsupported KPN prune mode"):
            get_kpn_prune_indices(
                source_size=(6, 6),
                target_size=(4, 4),
                mode="top_left",
            )
