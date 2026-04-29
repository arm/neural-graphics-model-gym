# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path

import torch

from ng_model_gym.usecases.nfru.model.nfru_v1_ne import NFRUAutoEncoder

_GOLDEN_ROOT = Path(__file__).resolve().parent / "data" / "nfru_v1_golden_values"


@unittest.skip("NFRU CI/assets disabled for now")
class TestNFRUAutoEncoder(unittest.TestCase):
    """Golden-value regression tests for the NFRU v1 autoencoder."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.auto_encoder = NFRUAutoEncoder().to(self.device)
        state_dict = torch.load(
            _GOLDEN_ROOT / "autoencoder_state_golden.pt",
            map_location=self.device,
            weights_only=False,
        )
        self.auto_encoder.load_state_dict(state_dict)
        self.auto_encoder.eval()
        self.autoencoder_input = torch.load(
            _GOLDEN_ROOT / "autoencoder_input_golden.pt",
            map_location=self.device,
            weights_only=False,
        ).to(self.device)
        self.expected_output = torch.load(
            _GOLDEN_ROOT / "autoencoder_output_golden.pt",
            map_location=self.device,
            weights_only=False,
        ).to(self.device)
        self.autoencoder_state_train = torch.load(
            _GOLDEN_ROOT / "autoencoder_state_train_golden.pt",
            map_location=self.device,
            weights_only=False,
        )
        self.autoencoder_input_train = torch.load(
            _GOLDEN_ROOT / "autoencoder_input_train_golden.pt",
            map_location=self.device,
            weights_only=False,
        ).to(self.device)
        self.expected_output_train = torch.load(
            _GOLDEN_ROOT / "autoencoder_output_train_golden.pt",
            map_location=self.device,
            weights_only=False,
        ).to(self.device)

    def test_forward_matches_golden_values(self) -> None:
        """Ensure the autoencoder reproduces the expected motion parameters."""
        torch.manual_seed(1)
        with torch.no_grad():
            output = self.auto_encoder(self.autoencoder_input.clone())

        self.assertEqual(output.shape, self.expected_output.shape)
        self.assertTrue(torch.isfinite(output).all().item())
        torch.testing.assert_close(output, self.expected_output, rtol=1e-4, atol=1e-4)

    def test_forward_matches_training_golden_values(self) -> None:
        """Validate the training capture reproduces the recorded motion parameters."""
        auto_encoder = NFRUAutoEncoder().to(self.device)
        auto_encoder.load_state_dict(self.autoencoder_state_train)
        auto_encoder.train()

        torch.manual_seed(5)
        output = auto_encoder(self.autoencoder_input_train.clone())

        self.assertEqual(output.shape, self.expected_output_train.shape)
        self.assertTrue(torch.isfinite(output).all().item())
        torch.testing.assert_close(
            output, self.expected_output_train, rtol=1e-3, atol=5e-3
        )

    def test_forward_is_deterministic_with_fixed_seed(self) -> None:
        """Running forward twice with the same RNG seed yields identical tensors."""
        torch.manual_seed(123)
        with torch.no_grad():
            first = self.auto_encoder(self.autoencoder_input.clone())

        torch.manual_seed(123)
        with torch.no_grad():
            second = self.auto_encoder(self.autoencoder_input.clone())

        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_backward_pass_propagates_gradients(self) -> None:
        """A toy loss produces finite gradients on parameters and inputs."""
        self.auto_encoder.train()
        self.auto_encoder.zero_grad(set_to_none=True)

        train_input = self.autoencoder_input.clone().detach().requires_grad_(True)

        torch.manual_seed(7)

        output = self.auto_encoder(train_input)
        loss = output.square().mean()
        loss.backward()

        weight_grad = self.auto_encoder.output_conv_mv.weight.grad
        self.assertIsNotNone(weight_grad)
        self.assertGreater(weight_grad.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(weight_grad).all().item())

        self.assertIsNotNone(train_input.grad)
        self.assertTrue(torch.isfinite(train_input.grad).all().item())

    def test_forward_outputs_are_finite(self) -> None:
        """Forward outputs contain no NaN or Inf values under golden inputs."""
        torch.manual_seed(99)
        with torch.no_grad():
            output = self.auto_encoder(self.autoencoder_input.clone())

        self.assertTrue(torch.isfinite(output).all().item())
