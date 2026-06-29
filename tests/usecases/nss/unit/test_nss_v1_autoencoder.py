# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from ng_model_gym.usecases.nss.model.model_blocks_v1 import (
    AutoEncoderV1,
    get_kpn_prune_indices,
)
from tests.base_gpu_test import BaseGPUMemoryTest

_RTOL = 1e-6
_ATOL = 1e-6


class TestNSSV1AutoEncoder(BaseGPUMemoryTest):
    """Tests for the NSS v1 AutoEncoderV1 block."""

    def test_output_shapes_for_high_quality_kpn_size(self):
        """Test AutoEncoderV1 output shapes for the high-quality KPN size."""

        autoencoder = AutoEncoderV1(kpn_size=(6, 6))
        x = torch.randn(2, 12, 128, 128)

        kpn_params, temporal_params = autoencoder(x)

        self.assertEqual(kpn_params.shape, (2, 36, 32, 32))
        self.assertEqual(temporal_params.shape, (2, 4, 128, 128))

    def test_output_shapes_for_low_mid_quality_kpn_size(self):
        """Test AutoEncoderV1 output shapes for the low/mid-quality KPN size."""

        autoencoder = AutoEncoderV1(kpn_size=(4, 4))
        x = torch.randn(2, 12, 128, 128)

        kpn_params, temporal_params = autoencoder(x)

        self.assertEqual(kpn_params.shape, (2, 16, 32, 32))
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


class TestNSSV1AutoEncoderGolden(unittest.TestCase):
    """Test NSS v1 autoencoder implementation against known inputs and outputs."""

    def test_high_forward_pass(self):
        """Test high-quality autoencoder forward pass."""

        self._run_autoencoder_golden("high", kpn_size=(6, 6))

    def test_mid_forward_pass(self):
        """Test mid-quality autoencoder forward pass."""

        self._run_autoencoder_golden("mid", kpn_size=(4, 4))

    def _run_autoencoder_golden(self, quality, kpn_size):
        device = torch.device("cpu")

        autoencoder_input = torch.load(
            "tests/usecases/nss/unit/data/nss_v1_golden_values/"
            + f"nss_v1_{quality}_autoencoder_input_golden.pt",
            map_location=device,
            weights_only=True,
        )

        autoencoder_output = torch.load(
            "tests/usecases/nss/unit/data/nss_v1_golden_values/"
            + f"nss_v1_{quality}_autoencoder_output_golden.pt",
            map_location=device,
            weights_only=True,
        )

        autoencoder = AutoEncoderV1(kpn_size=kpn_size)
        autoencoder.load_state_dict(autoencoder_input["autoencoder_state"])
        autoencoder.to(device)
        autoencoder.eval()

        with torch.no_grad():
            kpn_params, temporal_params = autoencoder(autoencoder_input["input_tensor"])

        torch.testing.assert_close(
            kpn_params,
            autoencoder_output["kpn_params"],
            rtol=_RTOL,
            atol=_ATOL,
        )
        torch.testing.assert_close(
            temporal_params,
            autoencoder_output["temporal_params"],
            rtol=_RTOL,
            atol=_ATOL,
        )
