# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Dict
from unittest.mock import Mock

import torch
from torch import nn

from ng_model_gym.core.config.config_model import ConfigModel
from ng_model_gym.core.model import BaseNGModel, create_model
from ng_model_gym.core.utils.enum_definitions import TrainEvalMode
from ng_model_gym.usecases.nfru.model.nfru_v1 import m
from ng_model_gym.usecases.nfru.model.nfru_v1_ne import NFRUAutoEncoder
from tests.base_gpu_test import BaseGPUMemoryTest
from tests.testing_utils import create_simple_params

_GOLDEN_ROOT = Path(__file__).resolve().parent / "data" / "nfru_v1_golden_values"
# Slang-backed full-model outputs can drift slightly across GPU environments.
# Keep the end-to-end rendered-output check, but allow a small absolute margin
# for cross-GPU/runtime variation in the full pipeline replay.
_OUTPUT_ATOL = 2e-3
_OUTPUT_RTOL = 1e-3
_SWAP_COEFF_ATOL = 2e-3
_SWAP_COEFF_RTOL = 5e-3


@unittest.skip("NFRU CI/assets disabled for now")
class TestNFRUV1Model(BaseGPUMemoryTest):
    """Regression tests for the NFRU v1 model using recorded golden data."""

    def setUp(self) -> None:
        super().setUp()
        self.assertTrue(
            torch.cuda.is_available(),
            "CUDA is required for NFRU v1 slang-backed golden tests.",
        )
        self.device = torch.device("cuda")
        self._reset_rng()

        autoencoder_state = torch.load(
            _GOLDEN_ROOT / "autoencoder_state_golden.pt",
            map_location=self.device,
            weights_only=False,
        )
        self.autoencoder_state = self._state_to_device(autoencoder_state)

        self._build_model(batch_size=1, autoencoder_state=self.autoencoder_state)

        self.forward_reference = torch.load(
            _GOLDEN_ROOT / "forward_pass_inputs_golden.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.forward_reference_dynamic_flow = torch.load(
            _GOLDEN_ROOT / "forward_pass_inputs_dynamic_flow_golden.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.expected_outputs = torch.load(
            _GOLDEN_ROOT / "forward_pass_output_golden.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.forward_reference_scale3 = torch.load(
            _GOLDEN_ROOT / "forward_pass_inputs_scale3_golden.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.expected_outputs_scale3 = torch.load(
            _GOLDEN_ROOT / "forward_pass_output_scale3_golden.pt",
            map_location="cpu",
            weights_only=False,
        )

    def _build_model(
        self, batch_size: int, autoencoder_state: Dict[str, torch.Tensor]
    ) -> None:
        """Create a fresh NFRU model instance configured for the given batch size."""
        params = create_simple_params(usecase="nfru")
        params_dict = params.model_dump(mode="json")

        params_dict["dataset"]["health_check"] = False
        params_dict["train"]["batch_size"] = batch_size

        params = ConfigModel.model_validate(params_dict)
        params.model_train_eval_mode = TrainEvalMode.FP32

        model = create_model(params, self.device)
        if not isinstance(model, BaseNGModel):
            raise TypeError("Failed to create a BaseNGModel instance for NFRU")

        auto_encoder = model.get_neural_network()
        auto_encoder.load_state_dict(autoencoder_state)
        model.eval()
        model.on_evaluation_start()
        self.model = model

    def _configure_network(self, reference: Dict[str, torch.Tensor]) -> None:
        """Apply golden configuration to the model and move inputs to the test device."""
        network = self.model.network
        quant_params = reference["quant_params"]
        network.quant_params = {
            "max_val": float(quant_params["max_val"]),
            "bits_exp": int(quant_params["bits_exp"]),
            "bits_x": int(quant_params["bits_x"]),
            "bits_y": int(quant_params["bits_y"]),
        }
        network.flow_method = reference["flow_method"]
        network.scale_factor = int(reference["scale_factor"])
        network.shader_accurate = bool(reference["shader_accurate"])
        network.new_dynamic_mask = bool(reference.get("new_dynamic_mask", False))

    def _prepare_inputs(
        self, reference: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        self._configure_network(reference)
        inputs = reference["inputs"]
        return {
            key: tensor.to(self.device) if isinstance(tensor, torch.Tensor) else tensor
            for key, tensor in inputs.items()
        }

    def _reset_rng(self) -> None:
        """Reset CPU and CUDA random number generator state before stochastic forward passes."""
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

    def _assert_output_shapes(self, outputs: Dict[str, torch.Tensor]) -> None:
        expected = self.expected_outputs

        self.assertIn("output", outputs)
        self.assertIn("coeffs", outputs)
        self.assertIn("output_mfg", outputs)

        self.assertEqual(outputs["output"].shape, expected["output"].shape)
        self.assertEqual(outputs["coeffs"].shape, expected["coeffs"].shape)
        self.assertIsInstance(outputs["output_mfg"], list)
        self.assertEqual(len(outputs["output_mfg"]), len(expected["output_mfg"]))
        for generated, exp in zip(outputs["output_mfg"], expected["output_mfg"]):
            self.assertIsInstance(generated, torch.Tensor)
            self.assertEqual(generated.shape, exp.shape)

    def test_forward_pass_output_shapes(self) -> None:
        """Ensure the NFRU v1 model produces tensors of the expected shape."""
        inputs = self._prepare_inputs(self.forward_reference)
        self._reset_rng()
        with torch.no_grad():
            outputs = self.model(inputs)

        self._assert_output_shapes(outputs)

    def test_forward_pass_matches_golden_values(self) -> None:
        """Compare the model forward pass with recorded output tensors."""
        inputs = self._prepare_inputs(self.forward_reference)
        self._reset_rng()

        with torch.no_grad():
            outputs = self.model(inputs)

        expected_output = self.expected_outputs["output"].to(self.device)
        torch.testing.assert_close(
            outputs["output"],
            expected_output,
            rtol=_OUTPUT_RTOL,
            atol=_OUTPUT_ATOL,
        )

        # The standalone autoencoder golden tests define coefficient exactness on
        # CPU. In the full slang-backed GPU path, tiny cross-device differences in
        # the intermediate coeff tensor are expected, so only require finiteness
        # and let the rendered outputs remain the regression oracle here.
        self.assertEqual(outputs["coeffs"].shape, self.expected_outputs["coeffs"].shape)
        self.assertTrue(torch.isfinite(outputs["coeffs"]).all())

        expected_mfg = [
            frame.to(self.device) for frame in self.expected_outputs["output_mfg"]
        ]
        self.assertEqual(len(outputs["output_mfg"]), len(expected_mfg))
        for generated, expected in zip(outputs["output_mfg"], expected_mfg):
            torch.testing.assert_close(
                generated,
                expected,
                rtol=_OUTPUT_RTOL,
                atol=_OUTPUT_ATOL,
            )

    def test_forward_pass_scale_factor_three_matches_golden(self) -> None:
        """Regress a 3x interpolation run to exercise the multi-frame loop."""
        inputs = self._prepare_inputs(self.forward_reference_scale3)

        self._reset_rng()
        with torch.no_grad():
            outputs = self.model(inputs)

        expected = self.expected_outputs_scale3
        torch.testing.assert_close(
            outputs["output"],
            expected["output"].to(self.device),
            rtol=_OUTPUT_RTOL,
            atol=_OUTPUT_ATOL,
        )
        self.assertEqual(outputs["coeffs"].shape, expected["coeffs"].shape)
        self.assertTrue(torch.isfinite(outputs["coeffs"]).all())

        expected_mfg = [frame.to(self.device) for frame in expected["output_mfg"]]
        self.assertEqual(len(outputs["output_mfg"]), len(expected_mfg))
        self.assertEqual(
            len(outputs["output_mfg"]), self.model.network.scale_factor - 1
        )
        for generated, exp in zip(outputs["output_mfg"], expected_mfg):
            torch.testing.assert_close(
                generated,
                exp,
                rtol=_OUTPUT_RTOL,
                atol=_OUTPUT_ATOL,
            )

    def test_set_neural_network_swaps_autoencoder(self) -> None:
        """Ensure set_neural_network injects the provided module into the forward path."""
        inputs = self._prepare_inputs(self.forward_reference)

        class RecordingAutoEncoder(nn.Module):
            """Wrap an autoencoder to collect call metadata during replacement tests."""

            def __init__(self, inner: nn.Module):
                """Store the wrapped module and initialise tracking state."""
                super().__init__()
                self.inner = inner
                self.calls = 0
                self.last_input_shape = None

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Record invocation details then delegate to the wrapped module."""
                self.calls += 1
                self.last_input_shape = tuple(x.shape)
                return self.inner(x)

        self._reset_rng()
        baseline_outputs = self.model(inputs)

        replacement_inner = NFRUAutoEncoder().to(self.device)
        replacement_inner.load_state_dict(self.model.get_neural_network().state_dict())
        replacement_inner.eval()
        recording_wrapper = RecordingAutoEncoder(replacement_inner)
        recording_wrapper.eval()
        recording_wrapper.to(self.device)

        self.model.set_neural_network(recording_wrapper)

        self._reset_rng()
        swapped_outputs = self.model(inputs)

        self.assertGreaterEqual(recording_wrapper.calls, 1)
        self.assertIsNotNone(recording_wrapper.last_input_shape)
        torch.testing.assert_close(
            swapped_outputs["output"],
            baseline_outputs["output"],
            rtol=_OUTPUT_RTOL,
            atol=_OUTPUT_ATOL,
        )
        torch.testing.assert_close(
            swapped_outputs["coeffs"],
            baseline_outputs["coeffs"],
            rtol=_SWAP_COEFF_RTOL,
            atol=_SWAP_COEFF_ATOL,
        )

    def test_on_after_batch_transfer_applies_colour_pipeline(self) -> None:
        """Colour-correct ground truth tensors and preserve device/dtype."""
        batch_inputs = {"MotionMat": torch.zeros(1, 2, 4, 4, device=self.device)}
        ground_truth = torch.rand(1, 3, 4, 4, device=self.device, dtype=torch.float32)

        self.model.eval()
        expected = self.model.network.colour_pipeline(
            ground_truth, batch_inputs, time_index="m1"
        )
        if not isinstance(expected, torch.Tensor):
            expected = torch.from_numpy(expected)
        expected = expected.to(self.device, dtype=torch.float32)

        processed_inputs, processed_gt = self.model.on_after_batch_transfer(
            (batch_inputs, ground_truth.clone())
        )

        self.assertIs(processed_inputs, batch_inputs)
        torch.testing.assert_close(processed_gt, expected)
        self.assertEqual(processed_gt.device, ground_truth.device)
        self.assertEqual(processed_gt.dtype, torch.float32)

    def test_on_after_batch_transfer_resamples_random_effects_in_train(self) -> None:
        """Train-mode batch transfer should resample random colour effects once."""
        batch_inputs = {"MotionMat": torch.zeros(1, 2, 4, 4, device=self.device)}
        ground_truth = torch.rand(1, 3, 4, 4, device=self.device, dtype=torch.float32)

        self.model.train()
        self.model.on_train_epoch_start()
        colour_pipeline = self.model.network.colour_pipeline
        calls = {"count": 0}

        def _record_resample() -> None:
            calls["count"] += 1

        if not hasattr(colour_pipeline, "resample_effects"):
            self.fail("Train pipeline should support per-batch effect resampling")
        colour_pipeline.resample_effects = _record_resample

        _, processed_gt = self.model.on_after_batch_transfer(
            (batch_inputs, ground_truth.clone())
        )

        self.assertEqual(calls["count"], 1)
        self.assertEqual(processed_gt.device, ground_truth.device)
        self.assertEqual(processed_gt.dtype, torch.float32)

    def test_dynamic_mask_selector_uses_reference_default(self) -> None:
        """Dynamic mask selector should use the v1 reference default."""
        network = self.model.network
        self.assertIs(network._get_dynamic_mask_fn(), m.calculate_previous_dynamic_mask)

    def test_model_hooks_switch_colour_pipeline_by_split(self) -> None:
        """Explicit lifecycle hooks should select the colour pipeline by split."""
        network = self.model.network
        train_pipeline = network.available_colour_pipeline["train"]
        validation_pipeline = network.available_colour_pipeline["validation"]
        test_pipeline = network.available_colour_pipeline["test"]

        self.model.train()
        self.model.on_train_epoch_start()
        self.assertIs(network.colour_pipeline, train_pipeline)

        self.model.eval()
        self.model.on_validation_start()
        self.assertIs(network.colour_pipeline, validation_pipeline)

        self.model.on_evaluation_start()
        self.assertIs(network.colour_pipeline, test_pipeline)

    def test_model_requires_colour_preprocessing(self) -> None:
        """NFRU v1 should reject configs without explicit colour preprocessing."""
        params = create_simple_params(usecase="nfru")
        params.dataset.colour_preprocessing = None
        with self.assertRaisesRegex(ValueError, "dataset.colour_preprocessing"):
            create_model(params, self.device)

    def test_model_requires_all_colour_preprocessing_splits(self) -> None:
        """NFRU v1 should reject configs missing validation/test/train splits."""
        params = create_simple_params(usecase="nfru")
        if params.dataset.colour_preprocessing is None:
            self.fail("Expected colour_preprocessing config in simple NFRU params")
        params.dataset.colour_preprocessing.validation = None

        with self.assertRaisesRegex(
            ValueError, "Missing or invalid splits: validation"
        ):
            create_model(params, self.device)

    def test_set_colour_pipeline_requires_exact_split_name(self) -> None:
        """Unexpected split names should fail instead of falling back silently."""
        with self.assertRaisesRegex(ValueError, "preview"):
            self.model.network.set_colour_pipeline("preview")

    def test_forward_pass_rejects_scale_factor_one(self) -> None:
        """Runtime guard should reject scale factors with no interpolation steps."""
        inputs = self._prepare_inputs(self.forward_reference)
        self.model.network.scale_factor = 1

        with self.assertRaisesRegex(ValueError, "no interpolation timesteps"):
            self.model(inputs)

    def test_forward_pass_uses_precomputed_v311_flow_when_available(self) -> None:
        """Supplying blockmatch-v311 flow should bypass dynamic recomputation."""
        inputs = self._prepare_inputs(self.forward_reference)
        network = self.model.network
        network.flow_method = "blockmatch_v311"
        flow_key = "flow_m1_f30_p1@blockmatch_v311"
        if flow_key not in inputs:
            self.fail(
                "Forward reference is missing the NFRU v1 blockmatch-v311 flow tensor."
            )
        network.dynamic_flow_model.forward = Mock(
            side_effect=AssertionError("dynamic flow should not be recomputed")
        )

        self._reset_rng()

        with torch.no_grad():
            outputs = self.model(inputs)

        network.dynamic_flow_model.forward.assert_not_called()
        self.assertIn("output", outputs)
        self.assertIn("coeffs", outputs)

    def test_forward_pass_computes_v311_flow_when_missing(self) -> None:
        """Missing blockmatch-v311 flow should be recomputed by the dynamic flow path."""
        inputs = self._prepare_inputs(self.forward_reference_dynamic_flow)
        network = self.model.network
        network.flow_method = "blockmatch_v311"
        flow_key = "flow_m1_f30_p1@blockmatch_v311"
        self.assertNotIn(flow_key, inputs)

        expected_flow = self.forward_reference["inputs"][flow_key].to(self.device)
        dynamic_flow_forward = network.dynamic_flow_model.forward
        network.dynamic_flow_model.forward = Mock(wraps=dynamic_flow_forward)

        self._reset_rng()

        with torch.no_grad():
            outputs = self.model(inputs)

        network.dynamic_flow_model.forward.assert_called_once()
        self.assertIn(flow_key, inputs)
        flow_diff = (inputs[flow_key] - expected_flow).abs()
        self.assertLess(flow_diff.mean().item(), 5e-4)
        self.assertLessEqual(int((flow_diff > 0.1).sum().item()), 256)
        self.assertLessEqual(flow_diff.max().item(), 8.0)

        self._assert_output_shapes(outputs)
        self.assertTrue(torch.isfinite(outputs["output"]).all())
        self.assertTrue(torch.isfinite(outputs["coeffs"]).all())

    def test_network_reuses_cached_preprocessing_modules(self) -> None:
        """Cache stateless preprocessing modules on network construction."""
        network = self.model.network
        self.assertIsInstance(network.flow_downsampler, nn.Module)
        self.assertIsInstance(network.flow_upsampler, nn.Module)
        self.assertIsInstance(network.coeff_softmax, nn.Softmax)

    def _state_to_device(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Move all tensor entries in a state dict to the test device."""
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in state_dict.items()
        }
