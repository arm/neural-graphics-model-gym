# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch
from torch import nn

from ng_model_gym.usecases.nss.model.model_blocks_v1 import (
    AutoEncoderV1,
    get_kpn_prune_indices,
)

_RTOL = 1e-4
_ATOL = 1e-4


class TestNSSV1AutoEncoder(unittest.TestCase):
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

    def test_forward_pass_consistency_high_quality(self):
        """Test if the AutoEncoderV1 produces stable values for high-quality KPN size."""

        autoencoder = AutoEncoderV1(kpn_size=(6, 6))
        x = torch.randn(2, 12, 128, 128)

        kpn_params1, temporal_params1 = autoencoder(x)
        kpn_params2, temporal_params2 = autoencoder(x)

        torch.testing.assert_close(kpn_params1, kpn_params2, rtol=_RTOL, atol=_ATOL)
        torch.testing.assert_close(
            temporal_params1, temporal_params2, rtol=_RTOL, atol=_ATOL
        )

    def test_forward_pass_consistency_low_mid_quality(self):
        """Test if the AutoEncoderV1 produces stable values for low/mid-quality KPN size."""

        autoencoder = AutoEncoderV1(kpn_size=(4, 4))
        x = torch.randn(2, 12, 128, 128)

        kpn_params1, temporal_params1 = autoencoder(x)
        kpn_params2, temporal_params2 = autoencoder(x)

        torch.testing.assert_close(kpn_params1, kpn_params2, rtol=_RTOL, atol=_ATOL)
        torch.testing.assert_close(
            temporal_params1, temporal_params2, rtol=_RTOL, atol=_ATOL
        )

    def test_output_values_for_known_input(self):
        """Test AutoEncoderV1 output values for a known input tensor."""
        autoencoder = AutoEncoderV1(kpn_size=(4, 4))

        def init_weights(module):
            """Initializes every nn.Conv2d weight and bias to 0.1"""
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.weight, 0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.1)

        autoencoder.apply(init_weights)
        x = torch.ones((1, 12, 128, 128))

        with torch.no_grad():
            kpn_params, temporal_params = autoencoder(x)

        self.assertEqual(kpn_params.shape, (1, 16, 32, 32))
        self.assertEqual(temporal_params.shape, (1, 4, 128, 128))
        self.assertTrue(torch.isfinite(kpn_params).all().item())
        self.assertTrue(torch.isfinite(temporal_params).all().item())

        self.assertAlmostEqual(kpn_params.flatten()[0].item(), 1.0, places=6)
        self.assertAlmostEqual(temporal_params.flatten()[0].item(), 1.0, places=6)

    def test_model_training(self):
        """Test we can do backward pass"""
        autoencoder = AutoEncoderV1(kpn_size=(4, 4))
        input_tensor = torch.ones(2, 12, 128, 128)

        kpn_params, temporal_params = autoencoder(input_tensor)
        loss = kpn_params.mean() + temporal_params.mean()
        loss.backward()

        # Check that gradients are here
        for param in autoencoder.parameters():
            self.assertIsNotNone(param.grad)

    def test_zero_input(self):
        """Test AutoEncoderV1 output for a zero input tensor."""
        autoencoder = AutoEncoderV1(kpn_size=(4, 4))
        zero_input = torch.zeros(2, 12, 128, 128)

        kpn_params, temporal_params = autoencoder(zero_input)

        # Should not have NaN or Inf in any output
        self.assertFalse(torch.isinf(kpn_params).any())
        self.assertFalse(torch.isnan(kpn_params).any())
        self.assertFalse(torch.isinf(temporal_params).any())
        self.assertFalse(torch.isnan(temporal_params).any())

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
