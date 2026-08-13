# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
"""Portable CPU tests for the NFRU native Torch processing stages."""

# pylint: disable=missing-function-docstring,too-many-lines

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from ng_model_gym.core.model import create_model
from ng_model_gym.core.model.shaders import slang_utils
from ng_model_gym.core.model.shaders.slang_utils import SlangOutput
from ng_model_gym.core.utils.enum_definitions import TrainEvalMode
from ng_model_gym.usecases.nfru.model.nfru_v1 import NFRUv1Core
from ng_model_gym.usecases.nfru.model.torch_processing import (
    postprocess_torch,
    preprocess_torch,
    previous_dynamic_mask_torch,
    warp_flow_torch,
    warp_mv_torch,
)
from ng_model_gym.usecases.nfru.model.torch_processing.motion import (
    _fill_motion,
    _warp_flow_packed_torch,
    _warp_mv_packed_torch,
)
from ng_model_gym.usecases.nfru.model.torch_processing.preprocess import _hash_to_float
from ng_model_gym.usecases.nfru.model.torch_processing.quantization import (
    decode_motion,
    pack_depth_motion,
)
from ng_model_gym.usecases.nfru.model.torch_processing.sampling import (
    bilinear_sample,
    gather_pixels,
    ordered_nearest_depth,
)
from tests.testing_utils import create_simple_params

_GOLDEN_ROOT = Path(__file__).resolve().parent / "data" / "nfru_v1_golden_values"


def _small_forward_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    """Crop the immutable full-forward fixture into a fast complete-frame case."""

    reference = torch.load(
        _GOLDEN_ROOT / "forward_pass_inputs_golden.pt",
        map_location="cpu",
        weights_only=False,
    )["inputs"]
    result = {}
    for name, value in reference.items():
        if not isinstance(value, torch.Tensor):
            result[name] = value
        elif value.ndim == 4 and value.shape[-2:] == (1080, 1920):
            result[name] = value[:, :, :64, :64].clone().to(device)
        elif value.ndim == 4 and value.shape[-2:] == (540, 960):
            result[name] = value[:, :, :32, :32].clone().to(device)
        else:
            result[name] = value.clone().to(device)
    return result


class TestNFRUTorchPrimitives(unittest.TestCase):
    """Exercise shader-sensitive primitives without CUDA."""

    def test_bilinear_sampling_uses_pixel_centers_and_edge_clamping(self):
        source = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        centers = torch.tensor(
            [[[[0.25, 0.25], [0.25, 0.75]], [[0.75, 0.25], [0.75, 0.75]]]]
        )
        torch.testing.assert_close(
            bilinear_sample(source, centers), source, rtol=0.0, atol=0.0
        )
        edge = bilinear_sample(source, torch.tensor([[[[1.0, 1.0]]]]))
        torch.testing.assert_close(edge, torch.tensor([[[[4.0]]]]))

    def test_integer_sampling_distinguishes_zero_oob_and_edge_clamping(self):
        source = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        pixels = torch.tensor([[[[-1, 0], [0, 2], [1, 1]]]])

        zero_oob = gather_pixels(source, pixels, clamp_to_edge=False)
        clamped = gather_pixels(source, pixels, clamp_to_edge=True)

        self.assertTrue(torch.equal(zero_oob, torch.tensor([[[[0.0, 0.0, 4.0]]]])))
        self.assertTrue(torch.equal(clamped, torch.tensor([[[[1.0, 2.0, 4.0]]]])))

    def test_ordered_nearest_depth_retains_first_equal_tap(self):
        depth = torch.full((1, 1, 3, 3), 9.0)
        depth[0, 0, 2, 1] = 1.0
        depth[0, 0, 1, 2] = 1.0
        pixels = torch.tensor([[[[1, 1]]]])

        nearest, offset = ordered_nearest_depth(depth, pixels)

        self.assertEqual(nearest.item(), 1.0)
        self.assertTrue(torch.equal(offset, torch.tensor([[[[1, 0]]]])))

    def test_oob_hash_has_exact_shader_bits(self):
        values = torch.stack(
            [_hash_to_float(torch.tensor([[[0]]]), channel, 17) for channel in range(3)]
        )

        self.assertEqual(
            [
                hex(value)
                for value in values.contiguous().view(torch.int32).flatten().tolist()
            ],
            [hex(0x3F5C029A), hex(0x3F3612F6), hex(0x3F5D2A62)],
        )

    def test_pack_decode_round_trip_preserves_quantized_motion(self):
        motion = torch.tensor([[[[12.5, -9.25], [1.5, 2.25]]]])
        depth = torch.tensor([[[1.0, 0.5]]])
        packed = pack_depth_motion(motion, depth)
        decoded = decode_motion(packed)

        self.assertEqual(packed.dtype, torch.int32)
        self.assertTrue(torch.all(packed >= 0).item())
        torch.testing.assert_close(decoded, motion, rtol=0.0, atol=0.15)

    def test_identity_camera_and_zero_rendered_motion_are_static(self):
        depth = torch.full((1, 1, 3, 5), 0.5)
        motion = torch.zeros(1, 2, 3, 5)
        transform = torch.eye(4).unsqueeze(0)

        result = previous_dynamic_mask_torch(depth, motion, transform, 0.01, 0.001)

        self.assertTrue(torch.equal(result, torch.zeros_like(result)))

    def test_flow_and_mv_zero_motion_keep_shapes_and_discrete_masks(self):
        depth = torch.full((1, 1, 4, 6), 0.5)
        flow = torch.zeros(1, 2, 2, 3)
        motion = torch.zeros(1, 2, 4, 6)
        mask = torch.zeros(1, 1, 4, 6)
        transform = torch.eye(4).unsqueeze(0)

        filled_flow = warp_flow_torch(depth, flow, 0.5)
        filled_mv, next_mask, holes_t, holes_tm1 = warp_mv_torch(
            depth,
            depth,
            motion,
            mask,
            transform,
            transform,
            0.5,
            0.01,
            0.001,
        )

        self.assertEqual(filled_flow.shape, flow.shape)
        self.assertEqual(filled_mv.shape, motion.shape)
        self.assertTrue(torch.equal(filled_flow, torch.zeros_like(filled_flow)))
        self.assertTrue(torch.equal(filled_mv, torch.zeros_like(filled_mv)))
        self.assertTrue(torch.equal(next_mask, torch.zeros_like(next_mask)))
        self.assertTrue(torch.all(holes_t == 1.0).item())
        self.assertTrue(torch.all(holes_tm1 == 1.0).item())


class TestNFRUTorchPrePost(unittest.TestCase):
    """Check stage contracts and the intentionally narrow autograd path."""

    def setUp(self):
        self.batch = 1
        self.depth_shape = (4, 6)
        self.flow_shape = (2, 3)
        self.rgb_shape = (8, 12)
        self.depth = torch.full((1, 1, *self.depth_shape), 0.5)
        self.flow = torch.zeros(1, 2, *self.flow_shape)
        self.motion = torch.zeros(1, 2, *self.depth_shape)
        self.holes = torch.ones(1, 1, *self.depth_shape)
        self.rgb_m1 = torch.rand(1, 3, *self.rgb_shape)
        self.rgb_p1 = torch.rand(1, 3, *self.rgb_shape)
        self.transform = torch.eye(4).unsqueeze(0)
        self.depth_params = torch.tensor([0.0, 1.0, 1.0, 1.0]).reshape(1, 4, 1, 1)

    def test_seeded_preprocess_is_repeatable_detached_and_16_channel(self):
        arguments = (
            self.flow,
            self.motion,
            self.rgb_m1,
            self.rgb_p1,
            self.depth,
            self.depth,
            self.holes,
            self.holes,
            self.transform,
            self.transform,
            self.depth_params,
            0.5,
            17,
        )
        first = preprocess_torch(*arguments)
        second = preprocess_torch(*arguments)

        self.assertEqual(first.shape, (1, 16, *self.flow_shape))
        self.assertEqual(first.dtype, torch.float32)
        self.assertFalse(first.requires_grad)
        self.assertTrue(torch.isfinite(first).all().item())
        self.assertTrue(torch.equal(first, second))

    def test_preprocess_detaches_every_input_and_preserves_values(self):
        tensors = [
            value.detach().clone().requires_grad_()
            for value in (
                self.flow,
                self.motion,
                self.rgb_m1,
                self.rgb_p1,
                self.depth,
                self.depth,
                self.holes,
                self.holes,
                self.transform,
                self.transform,
                self.depth_params,
            )
        ]
        originals = [value.detach().clone() for value in tensors]

        result = preprocess_torch(*tensors, 0.5, 17)

        self.assertFalse(result.requires_grad)
        for value, original in zip(tensors, originals):
            self.assertIsNone(value.grad)
            self.assertTrue(torch.equal(value, original))

    def test_postprocess_only_backpropagates_to_learned_params(self):
        flow = self.flow.requires_grad_()
        motion = self.motion.requires_grad_()
        rgb_m1 = self.rgb_m1.requires_grad_()
        rgb_p1 = self.rgb_p1.requires_grad_()
        learnt = torch.randn(1, 4, *self.flow_shape, requires_grad=True)
        originals = [
            value.detach().clone() for value in (flow, motion, rgb_m1, rgb_p1, learnt)
        ]

        output = postprocess_torch(flow, motion, rgb_m1, rgb_p1, learnt, timestep=0.5)
        output.square().mean().backward()

        self.assertEqual(output.shape, (1, 3, *self.rgb_shape))
        self.assertIsNotNone(learnt.grad)
        self.assertTrue(torch.isfinite(learnt.grad).all().item())
        self.assertIsNone(flow.grad)
        self.assertIsNone(motion.grad)
        self.assertIsNone(rgb_m1.grad)
        self.assertIsNone(rgb_p1.grad)
        for value, original in zip((flow, motion, rgb_m1, rgb_p1, learnt), originals):
            self.assertTrue(torch.equal(value, original))

    def test_postprocess_extreme_logits_are_finite_and_stable(self):
        learnt = torch.empty(1, 4, *self.flow_shape)
        learnt[:, 0] = 1000.0
        learnt[:, 1] = -1000.0
        learnt[:, 2] = 500.0
        learnt[:, 3] = -500.0

        output = postprocess_torch(
            self.flow,
            self.motion,
            self.rgb_m1,
            self.rgb_p1,
            learnt,
            timestep=0.5,
        )

        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output, self.rgb_m1, rtol=1e-5, atol=2e-7)

    def test_preprocess_cpu_matches_existing_slang_golden(self):
        inputs = torch.load(
            _GOLDEN_ROOT / "preprocess_inputs_golden.pt",
            map_location="cpu",
            weights_only=True,
        )
        expected = torch.load(
            _GOLDEN_ROOT / "preprocess_output_golden.pt",
            map_location="cpu",
            weights_only=True,
        )["network_in"]

        actual = preprocess_torch(
            inputs["flow_t_f30_xx"],
            inputs["mv_t_f30_m1"],
            inputs["rgb_m1"],
            inputs["rgb_p1"],
            inputs["depth_m1"],
            inputs["depth_p1"],
            inputs["depth_p1_warp_t"],
            inputs["depth_p1_warp_p1"],
            inputs["motion_mat_m1p1"],
            inputs["motion_mat_p1m1"],
            inputs["depth_params"],
            float(inputs["timestep"]),
            int(inputs["random_seed"].item()),
        )

        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_postprocess_cpu_matches_existing_slang_golden(self):
        inputs = torch.load(
            _GOLDEN_ROOT / "postprocess_inputs_golden.pt",
            map_location="cpu",
            weights_only=True,
        )
        expected = torch.load(
            _GOLDEN_ROOT / "postprocess_output_golden.pt",
            map_location="cpu",
            weights_only=True,
        )["output"]

        actual = postprocess_torch(
            inputs["flow_t_f30_xx"],
            inputs["mv_t_f30_m1"],
            inputs["rgb_m1"],
            inputs["rgb_p1"],
            inputs["learnt_params"],
            float(inputs["timestep"]),
        )

        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_preprocess_rejects_non_tensor_depth_params_by_name(self):
        with self.assertRaisesRegex(TypeError, "depth_params must be a torch.Tensor"):
            preprocess_torch(
                self.flow,
                self.motion,
                self.rgb_m1,
                self.rgb_p1,
                self.depth,
                self.depth,
                self.holes,
                self.holes,
                self.transform,
                self.transform,
                [0.0, 1.0, 1.0, 1.0],
                0.5,
                17,
            )

    def test_preprocess_validation_names_each_invalid_contract(self):
        valid = {
            "warped_flow": self.flow,
            "warped_mv": self.motion,
            "rgb_m1": self.rgb_m1,
            "rgb_p1": self.rgb_p1,
            "depth_m1": self.depth,
            "depth_p1": self.depth,
            "holes_t": self.holes,
            "holes_tm1": self.holes,
            "motion_mat_m1p1": self.transform,
            "motion_mat_p1m1": self.transform,
            "depth_params": self.depth_params,
            "timestep": 0.5,
            "random_seed": 17,
        }
        cases = (
            ("type", "warped_flow", [0.0], TypeError, "warped_flow"),
            ("rank", "warped_flow", self.flow[:, :, 0], ValueError, "warped_flow"),
            (
                "channels",
                "warped_flow",
                self.flow[:, :1],
                ValueError,
                "warped_flow",
            ),
            ("batch", "rgb_m1", self.rgb_m1.repeat(2, 1, 1, 1), ValueError, "rgb_m1"),
            (
                "spatial",
                "holes_t",
                self.holes[:, :, :-1],
                ValueError,
                "hole surfaces",
            ),
            ("dtype", "depth_m1", self.depth.double(), TypeError, "depth_m1"),
            (
                "matrix shape",
                "motion_mat_m1p1",
                torch.eye(3).unsqueeze(0),
                ValueError,
                "motion_mat_m1p1",
            ),
            (
                "metadata shape",
                "depth_params",
                self.depth_params[:, :, 0],
                ValueError,
                "depth_params",
            ),
            ("finite scalar", "timestep", float("inf"), ValueError, "timestep"),
            ("seed type", "random_seed", True, TypeError, "random_seed"),
        )
        for label, name, invalid, error, message in cases:
            with self.subTest(label=label):
                arguments = dict(valid)
                arguments[name] = invalid
                with self.assertRaisesRegex(error, message):
                    preprocess_torch(**arguments)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA validates mixed devices.")
    def test_preprocess_validation_names_mixed_device_input(self):
        with self.assertRaisesRegex(ValueError, "rgb_p1"):
            preprocess_torch(
                self.flow,
                self.motion,
                self.rgb_m1,
                self.rgb_p1.cuda(),
                self.depth,
                self.depth,
                self.holes,
                self.holes,
                self.transform,
                self.transform,
                self.depth_params,
                0.5,
                17,
            )

    def test_motion_stages_require_boolean_runtime_accurate(self):
        calls = (
            lambda: previous_dynamic_mask_torch(
                self.depth,
                self.motion,
                self.transform,
                0.01,
                0.001,
                runtime_accurate=1,
            ),
            lambda: warp_mv_torch(
                self.depth,
                self.depth,
                self.motion,
                torch.zeros_like(self.depth),
                self.transform,
                self.transform,
                0.5,
                0.01,
                0.001,
                runtime_accurate="false",
            ),
        )
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    TypeError, "runtime_accurate must be a bool"
                ):
                    call()


class TestNFRUTorchRouting(unittest.TestCase):
    """Regress device ownership and backend isolation at model level."""

    @staticmethod
    def _network():
        params = create_simple_params(usecase="nfru-v1")
        params.model.processing_backend = "torch"
        params.model_train_eval_mode = TrainEvalMode.FP32
        return create_model(params, torch.device("cpu")).network

    @staticmethod
    def _stage_cases():
        tensor = torch.zeros(1, 1, 1, 1)
        matrix = torch.eye(4).unsqueeze(0)
        preprocess_args = (tensor,) * 11 + (0.5, 17)
        postprocess_args = (tensor,) * 5 + (0.5,)
        return (
            (
                "previous_dynamic_mask",
                "previous_dynamic_mask_torch",
                "_previous_dynamic_mask_slang",
                (tensor, tensor, matrix),
            ),
            (
                "warp_mv",
                "warp_mv_torch",
                "_warp_mv_slang",
                (tensor,) * 6 + (0.5, 1, [1, 1]),
            ),
            (
                "warp_flow",
                "warp_flow_torch",
                "_warp_flow_slang",
                (tensor, tensor, 0.5, 1, [1, 1]),
            ),
            (
                "preprocess",
                "preprocess_torch",
                "_preprocess_slang",
                preprocess_args,
            ),
            (
                "postprocess",
                "postprocess_torch",
                "_postprocess_slang",
                postprocess_args,
            ),
        )

    def test_cpu_model_device_is_parameter_derived_without_cuda_query(self):
        params = create_simple_params(usecase="nfru-v1")
        params.model.processing_backend = "torch"
        params.model_train_eval_mode = TrainEvalMode.FP32

        with patch("torch.cuda.is_available", side_effect=AssertionError):
            model = create_model(params, torch.device("cpu"))

        self.assertEqual(model.device, torch.device("cpu"))
        self.assertEqual(model.network.device, torch.device("cpu"))

    def test_slang_cpu_guard_does_not_load_module(self):
        params = create_simple_params(usecase="nfru-v1")
        params.model.processing_backend = "slang"
        params.model_train_eval_mode = TrainEvalMode.FP32
        model = create_model(params, torch.device("cpu"))
        model.network._get_slang = Mock()

        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            model.network._require_cuda_for_slang_forward(
                {"rgb_linear_m1": torch.zeros(1, 3, 2, 2)}
            )

        model.network._get_slang.assert_not_called()

    def test_torch_stage_does_not_load_slang(self):
        params = create_simple_params(usecase="nfru-v1")
        params.model.processing_backend = "torch"
        params.model_train_eval_mode = TrainEvalMode.FP32
        model = create_model(params, torch.device("cpu"))
        model.network._get_slang = Mock(side_effect=AssertionError("loaded Slang"))
        depth = torch.full((1, 1, 2, 3), 0.5)
        motion = torch.zeros(1, 2, 2, 3)

        result = model.network.previous_dynamic_mask(
            depth, motion, torch.eye(4).unsqueeze(0)
        )

        self.assertEqual(result.device.type, "cpu")
        model.network._get_slang.assert_not_called()

    def test_all_stages_dispatch_to_only_the_selected_backend(self):
        network = self._network()
        module = "ng_model_gym.usecases.nfru.model.nfru_v1"

        for method_name, torch_name, slang_name, arguments in self._stage_cases():
            for backend in ("torch", "slang"):
                with self.subTest(stage=method_name, backend=backend):
                    expected = object()
                    network.processing_backend = backend
                    with patch(
                        f"{module}.{torch_name}", return_value=expected
                    ) as torch_impl, patch.object(
                        network, slang_name, return_value=expected
                    ) as slang_impl:
                        actual = getattr(network, method_name)(*arguments)

                    self.assertIs(actual, expected)
                    if backend == "torch":
                        torch_impl.assert_called_once()
                        slang_impl.assert_not_called()
                    else:
                        torch_impl.assert_not_called()
                        slang_impl.assert_called_once()

    def test_complete_cpu_torch_forward_never_touches_cuda_or_slang(self):
        params = create_simple_params(usecase="nfru-v1")
        params.model.processing_backend = "torch"
        params.model_train_eval_mode = TrainEvalMode.FP32
        model = create_model(params, torch.device("cpu"))
        model.eval()
        model.on_evaluation_start()
        model.network._next_preprocess_seed = Mock(return_value=17)
        inputs = _small_forward_inputs(torch.device("cpu"))

        with patch.object(
            model.network,
            "_require_cuda_for_slang_forward",
            side_effect=AssertionError("CUDA guard reached"),
        ), patch.object(
            model.network,
            "_get_slang",
            side_effect=AssertionError("Slang reached"),
        ), patch(
            "ng_model_gym.usecases.nfru.model.nfru_v1.load_slang_module",
            side_effect=AssertionError("Slang loaded"),
        ):
            with torch.no_grad():
                outputs = model(inputs)

        self.assertEqual(set(outputs), {"output", "coeffs", "output_mfg"})
        self.assertEqual(outputs["output"].shape, (1, 3, 64, 64))
        self.assertEqual(outputs["coeffs"].shape, (1, 4, 16, 16))
        self.assertEqual(len(outputs["output_mfg"]), 1)

    def test_cpu_torch_full_backward_reaches_trainable_parameters(self):
        params = create_simple_params(usecase="nfru-v1")
        params.model.processing_backend = "torch"
        params.model_train_eval_mode = TrainEvalMode.FP32
        model = create_model(params, torch.device("cpu"))
        model.on_evaluation_start()
        model.network._next_preprocess_seed = Mock(return_value=17)

        loss = model(_small_forward_inputs(torch.device("cpu")))["output"].mean()
        loss.backward()

        gradients = [
            parameter.grad
            for parameter in model.get_neural_network().parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(
            sum(gradient.abs().sum().item() for gradient in gradients), 0
        )

    def test_controlled_cpu_full_forward_matches_precise_slang_golden(self):
        torch.manual_seed(3)
        params = create_simple_params(usecase="nfru-v1")
        params.model.processing_backend = "torch"
        params.model_train_eval_mode = TrainEvalMode.FP32
        model = create_model(params, torch.device("cpu"))
        model.get_neural_network().load_state_dict(
            torch.load(
                _GOLDEN_ROOT / "autoencoder_state_golden.pt",
                map_location="cpu",
                weights_only=False,
            )
        )
        model.eval()
        model.on_evaluation_start()
        model.network._next_preprocess_seed = Mock(return_value=17)

        with torch.no_grad():
            outputs = model(_small_forward_inputs(torch.device("cpu")))

        output_coordinates = (
            (0, 0),
            (0, 63),
            (7, 13),
            (16, 31),
            (24, 24),
            (32, 48),
            (47, 11),
            (63, 63),
        )
        actual_output = torch.cat(
            [outputs["output"][0, :, row, column] for row, column in output_coordinates]
        )
        expected_output = torch.tensor(
            [
                0.1107043922,
                0.2074896097,
                0.1652634144,
                0.1075379401,
                0.2001839727,
                0.1627181321,
                0.1086709946,
                0.2026483715,
                0.1604052782,
                0.1370553076,
                0.2498794645,
                0.2222083956,
                0.2311285585,
                0.3643483818,
                0.3369982243,
                0.0919770300,
                0.1767938584,
                0.1352192760,
                0.1478743851,
                0.2258733809,
                0.2116565406,
                0.3174799085,
                0.3646405041,
                0.2672485411,
            ]
        )
        coefficient_coordinates = ((0, 0), (3, 7), (8, 8), (15, 15))
        actual_coefficients = torch.cat(
            [
                outputs["coeffs"][0, :, row, column]
                for row, column in coefficient_coordinates
            ]
        )
        expected_coefficients = torch.tensor(
            [
                0.5127666593,
                0.4817948043,
                0.0029791314,
                0.0024594436,
                0.4802976251,
                0.5196996331,
                0.0000012350,
                0.0000014934,
                0.4514693618,
                0.5485293865,
                0.0000005526,
                0.0000006880,
                0.3346528709,
                0.6033993959,
                0.0334260352,
                0.0285217296,
            ]
        )
        torch.testing.assert_close(actual_output, expected_output, rtol=1e-3, atol=2e-3)
        torch.testing.assert_close(
            actual_coefficients, expected_coefficients, rtol=1e-3, atol=2e-3
        )

    def test_to_immediately_updates_both_public_device_properties(self):
        params = create_simple_params(usecase="nfru-v1")
        params.model.processing_backend = "torch"
        params.model_train_eval_mode = TrainEvalMode.FP32
        model = create_model(params, torch.device("cpu"))

        targets = [torch.device("cpu")]
        if torch.cuda.is_available():
            targets.insert(0, torch.device("cuda"))
        for target in targets:
            with self.subTest(target=target):
                model.to(target)
                parameter_device = next(model.get_neural_network().parameters()).device
                self.assertEqual(parameter_device.type, target.type)
                self.assertEqual(model.device, parameter_device)
                self.assertEqual(model.network.device, parameter_device)

    def test_real_cpu_slang_forward_fails_before_module_loading(self):
        params = create_simple_params(usecase="nfru-v1")
        params.model.processing_backend = "slang"
        params.model_train_eval_mode = TrainEvalMode.FP32
        model = create_model(params, torch.device("cpu"))
        model.network._get_slang = Mock(side_effect=AssertionError("loaded Slang"))

        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            model(_small_forward_inputs(torch.device("cpu")))

        model.network._get_slang.assert_not_called()

    def test_each_slang_stage_guards_inputs_before_module_loading(self):
        network = Mock(spec=NFRUv1Core)
        tensor = torch.zeros(1, 1, 1, 1)
        matrix = torch.eye(4).unsqueeze(0)
        cases = (
            (
                "_previous_dynamic_mask_slang",
                (tensor, tensor, matrix),
                {"depth", "rendered_mv", "previous_transform"},
            ),
            (
                "_warp_mv_slang",
                (tensor,) * 6 + (0.5, 1, [1, 1]),
                {
                    "depth_m1",
                    "depth_p1",
                    "mv_p1_f30_m1",
                    "dynamic_mask",
                    "motion_mat_tm1",
                    "motion_mat_tp1",
                },
            ),
            (
                "_warp_flow_slang",
                (tensor, tensor, 0.5, 1, [1, 1]),
                {"depth", "mv"},
            ),
            (
                "_preprocess_slang",
                (tensor,) * 11 + (0.5, 17),
                {
                    "flow_t_f30_xx",
                    "mv_t_f30_m1",
                    "rgb_m1",
                    "rgb_p1",
                    "depth_m1",
                    "depth_p1",
                    "depth_p1_warp_t",
                    "depth_p1_warp_p1",
                    "motion_mat_m1p1",
                    "motion_mat_p1m1",
                    "depth_params",
                },
            ),
            (
                "_postprocess_slang",
                (tensor,) * 5 + (0.5,),
                {
                    "flow_t_f30_xx",
                    "mv_t_f30_m1",
                    "rgb_m1",
                    "rgb_p1",
                    "learnt_params",
                },
            ),
        )

        for method_name, arguments, expected_names in cases:
            with self.subTest(stage=method_name):
                network.reset_mock()
                network._require_cuda_for_slang_forward.side_effect = RuntimeError(
                    "requires CUDA"
                )
                network._get_slang.side_effect = AssertionError("loaded Slang")

                with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                    getattr(NFRUv1Core, method_name)(network, *arguments)

                guarded = network._require_cuda_for_slang_forward.call_args.args[0]
                self.assertEqual(set(guarded), expected_names)
                network._get_slang.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA exercises per-input guards.")
    def test_each_preprocess_cpu_argument_fails_before_loading_slang(self):
        params = create_simple_params(usecase="nfru-v1")
        params.model.processing_backend = "slang"
        params.model_train_eval_mode = TrainEvalMode.FP32
        network = create_model(params, torch.device("cuda")).network
        network._get_slang = Mock(side_effect=AssertionError("loaded Slang"))
        arguments = [
            torch.zeros(1, channels, height, width, device="cuda")
            for channels, height, width in (
                (2, 2, 3),
                (2, 4, 6),
                (3, 8, 12),
                (3, 8, 12),
                (1, 4, 6),
                (1, 4, 6),
                (1, 4, 6),
                (1, 4, 6),
            )
        ]
        arguments.extend(
            (
                torch.eye(4, device="cuda").unsqueeze(0),
                torch.eye(4, device="cuda").unsqueeze(0),
                torch.ones(1, 4, 1, 1, device="cuda"),
            )
        )

        for index in range(len(arguments)):
            with self.subTest(argument=index):
                mixed = list(arguments)
                mixed[index] = mixed[index].cpu()
                with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                    network.preprocess(*mixed, timestep=0.5, random_seed=17)

        network._get_slang.assert_not_called()


@unittest.skipUnless(
    torch.cuda.is_available(), "CUDA is required for live NFRU Slang/Torch parity."
)
class TestNFRUTorchLiveParity(unittest.TestCase):
    """Compare same-input Torch stages against precise authoritative Slang."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        slang_utils._load_slang_module_cached.cache_clear()
        load_module = slang_utils.slangtorch.loadModule
        cls.load_calls = []

        def load_precise(*args, **kwargs):
            kwargs["cudaFastMath"] = False
            cls.load_calls.append(dict(kwargs))
            return load_module(*args, **kwargs)

        cls.slang_patcher = patch.object(
            slang_utils.slangtorch, "loadModule", side_effect=load_precise
        )
        cls.slang_patcher.start()

    @classmethod
    def tearDownClass(cls):
        try:
            if not cls.load_calls:
                raise AssertionError("No precise Slang parity module was compiled.")
            if any(call.get("cudaFastMath") is not False for call in cls.load_calls):
                raise AssertionError("A parity Slang module used CUDA fast math.")
        finally:
            cls.slang_patcher.stop()
            slang_utils._load_slang_module_cached.cache_clear()
            super().tearDownClass()

    @staticmethod
    def _model(backend: str, *, device: str = "cuda", runtime_accurate=False):
        torch.manual_seed(3)
        torch.cuda.manual_seed(3)
        params = create_simple_params(usecase="nfru-v1")
        params.model.processing_backend = backend
        params.model.dynamic_mask_is_runtime_accurate = runtime_accurate
        params.model_train_eval_mode = TrainEvalMode.FP32
        model = create_model(params, torch.device(device))
        model.eval()
        model.on_evaluation_start()
        model.network._next_preprocess_seed = Mock(return_value=17)
        return model

    def test_fill_mv_matches_slang_for_empty_edges_and_equal_maxima(self):
        batch, height, width = 2, 4, 5
        packed = torch.zeros(batch, 1, height, width, dtype=torch.int32, device="cuda")
        codes = pack_depth_motion(
            torch.tensor(((3.0, -2.0), (-4.0, 1.0), (1.0, 5.0)), device="cuda"),
            torch.tensor((0.25, 0.9, 0.6), device="cuda"),
        )
        packed[0, 0, 0, 0] = codes[0]
        packed[0, 0, 1, 0] = codes[1]
        packed[0, 0, 0, 1] = codes[1]
        packed[0, 0, height - 1, width - 1] = codes[2]

        network = self._model("slang").network
        slang_output = network._get_slang().fill_mv(
            in_packed_mv=packed,
            out_constructors={
                "out_motion": SlangOutput(
                    shape=(batch, 2, height, width), device="cuda"
                )
            },
            dispatch_size=[batch, height, width],
        )

        torch.testing.assert_close(
            _fill_motion(packed), slang_output, rtol=0.0, atol=0.0
        )

    def test_private_packed_helpers_and_discrete_surfaces_match_exactly(self):
        generator = torch.Generator().manual_seed(1984)
        batch, height, width = 2, 6, 8
        current_depth = (
            0.1 + 0.8 * torch.rand(batch, 1, height, width, generator=generator)
        ).cuda()
        previous_depth = (
            0.1 + 0.8 * torch.rand(batch, 1, height, width, generator=generator)
        ).cuda()
        motion = (
            (torch.rand(batch, 2, height, width, generator=generator) - 0.5) * 0.3
        ).cuda()
        previous_mask = torch.ones(batch, 1, height, width, device="cuda")
        previous_mask[0, 0, 0, 0] = 2.0
        previous_to_current = torch.eye(4).repeat(batch, 1, 1).cuda()
        current_to_previous = torch.eye(4).repeat(batch, 1, 1).cuda()
        previous_to_current[:, 0, 3] = 0.01
        current_to_previous[:, 1, 3] = -0.02

        for runtime_accurate in (False, True):
            with self.subTest(runtime_accurate=runtime_accurate):
                network = self._model(
                    "slang", runtime_accurate=runtime_accurate
                ).network
                slang = network._get_slang()
                slang_outputs = slang.warp_mv(
                    in_depth=current_depth,
                    in_depth_m1=previous_depth,
                    in_motion=motion,
                    in_dynamic_mask=previous_mask,
                    in_motion_mat_m1p1=previous_to_current,
                    in_motion_mat_p1m1=current_to_previous,
                    in_timestep=0.25,
                    in_dynamic_mask_is_runtime_accurate=runtime_accurate,
                    in_mv_similarity_threshold=network.mv_similarity_threshold,
                    in_mv_similarity_noise_threshold=network.mv_similarity_noise_threshold,
                    out_constructors={
                        "out_packed_mv": SlangOutput(
                            shape=(batch, 1, height, width),
                            dtype=torch.int32,
                            device="cuda",
                        ),
                        "out_dynamic_mask": SlangOutput(
                            shape=(batch, 1, height, width), device="cuda"
                        ),
                        "out_holes_t": SlangOutput(
                            shape=(batch, 1, height, width), device="cuda"
                        ),
                        "out_holes_tm1": SlangOutput(
                            shape=(batch, 1, height, width), device="cuda"
                        ),
                    },
                    dispatch_size=[batch, height, width],
                )
                torch_outputs = _warp_mv_packed_torch(
                    current_depth,
                    previous_depth,
                    motion,
                    previous_mask,
                    previous_to_current,
                    current_to_previous,
                    0.25,
                    network.mv_similarity_threshold,
                    network.mv_similarity_noise_threshold,
                    runtime_accurate,
                )
                for name, actual, expected in zip(
                    ("packed", "next_mask", "holes_t", "holes_tm1"),
                    torch_outputs,
                    slang_outputs,
                ):
                    with self.subTest(output=name):
                        self.assertTrue(
                            torch.equal(actual, expected),
                            f"{name}: {(actual != expected).sum().item()} mismatches",
                        )

                mask_motion = torch.full_like(motion, 0.5)
                mask_transform = torch.eye(4).repeat(batch, 1, 1).cuda()
                slang_previous = network.previous_dynamic_mask(
                    previous_depth, mask_motion, mask_transform
                )
                torch_previous = previous_dynamic_mask_torch(
                    previous_depth,
                    mask_motion,
                    mask_transform,
                    network.mv_similarity_threshold,
                    network.mv_similarity_noise_threshold,
                    runtime_accurate,
                )
                self.assertTrue(torch.equal(torch_previous, slang_previous))

        flow = motion[:, :, :3, :4]
        for timestep in (0.25, 0.75):
            with self.subTest(timestep=timestep):
                network = self._model("slang").network
                slang_packed = network._get_slang().warp_flow(
                    in_depth=current_depth,
                    in_motion=flow,
                    in_timestep=timestep,
                    out_constructors={
                        "out_packed_mv": SlangOutput(
                            shape=(batch, 1, 3, 4),
                            dtype=torch.int32,
                            device="cuda",
                        )
                    },
                    dispatch_size=[batch, 3, 4],
                )
                self.assertTrue(
                    torch.equal(
                        _warp_flow_packed_torch(current_depth, flow, timestep),
                        slang_packed,
                    )
                )

    def test_preprocess_postprocess_and_learned_vjp_match(self):
        generator = torch.Generator().manual_seed(99)

        def random(shape, scale=1.0):
            return (torch.rand(*shape, generator=generator) * scale).cuda()

        batch = 2
        flow = random((batch, 2, 3, 5), 0.8) - 0.4
        motion = random((batch, 2, 6, 10), 0.8) - 0.4
        rgb_m1 = random((batch, 3, 12, 20))
        rgb_p1 = random((batch, 3, 12, 20))
        depth_m1 = random((batch, 1, 6, 10), 0.8) + 0.1
        depth_p1 = random((batch, 1, 6, 10), 0.8) + 0.1
        holes_t = (random((batch, 1, 6, 10)) > 0.5).float()
        holes_tm1 = (random((batch, 1, 6, 10)) > 0.5).float()
        matrix_a = torch.eye(4).repeat(batch, 1, 1).cuda()
        matrix_b = matrix_a.clone()
        matrix_a[:, 0, 3] = 0.02
        matrix_b[:, 1, 3] = -0.01
        depth_params = (
            torch.tensor([-1.0, 1.0, 1.0, 1.0])
            .repeat(batch, 1)
            .reshape(batch, 4, 1, 1)
            .cuda()
        )
        network = self._model("slang").network

        slang_pre = network._preprocess_slang(
            flow,
            motion,
            rgb_m1,
            rgb_p1,
            depth_m1,
            depth_p1,
            holes_t,
            holes_tm1,
            matrix_a,
            matrix_b,
            depth_params,
            0.4,
            17,
        )
        torch_pre = preprocess_torch(
            flow,
            motion,
            rgb_m1,
            rgb_p1,
            depth_m1,
            depth_p1,
            holes_t,
            holes_tm1,
            matrix_a,
            matrix_b,
            depth_params,
            0.4,
            17,
        )
        torch.testing.assert_close(torch_pre, slang_pre, rtol=1e-5, atol=5e-6)

        learnt_slang = (random((1, 4, 3, 5), 2.0) - 1.0).requires_grad_()
        learnt_torch = learnt_slang.detach().clone().requires_grad_()
        slang_post = network._postprocess_slang(
            flow[:1], motion[:1], rgb_m1[:1], rgb_p1[:1], learnt_slang, 0.4
        )
        torch_post = postprocess_torch(
            flow[:1], motion[:1], rgb_m1[:1], rgb_p1[:1], learnt_torch, 0.4
        )
        torch.testing.assert_close(torch_post, slang_post, rtol=1e-5, atol=5e-6)
        cotangent = torch.zeros_like(torch_post)
        cotangent[0, 0, 5, 7] = 1.0
        (slang_post * cotangent).sum().backward()
        (torch_post * cotangent).sum().backward()
        torch.testing.assert_close(
            learnt_torch.grad, learnt_slang.grad, rtol=1e-5, atol=1e-7
        )

    def test_controlled_full_forward_and_cpu_cuda_parity(self):
        slang_model = self._model("slang")
        torch_cuda_model = self._model("torch")
        torch_cuda_model.get_neural_network().load_state_dict(
            slang_model.get_neural_network().state_dict()
        )
        inputs = _small_forward_inputs(torch.device("cuda"))
        with torch.no_grad():
            slang_outputs = slang_model(
                {name: value.clone() for name, value in inputs.items()}
            )
            torch_outputs = torch_cuda_model(
                {name: value.clone() for name, value in inputs.items()}
            )

        self.assertEqual(set(torch_outputs), set(slang_outputs))
        for name in ("output", "coeffs"):
            self.assertEqual(torch_outputs[name].shape, slang_outputs[name].shape)
            torch.testing.assert_close(
                torch_outputs[name], slang_outputs[name], rtol=1e-5, atol=5e-6
            )
        for actual, expected in zip(
            torch_outputs["output_mfg"], slang_outputs["output_mfg"]
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=5e-6)

        torch_cpu_model = self._model("torch", device="cpu")
        torch_cpu_model.get_neural_network().load_state_dict(
            torch_cuda_model.get_neural_network().state_dict()
        )
        cpu_inputs = {
            name: value.cpu() if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()
        }
        with torch.no_grad():
            cpu_outputs = torch_cpu_model(cpu_inputs)
        for name in ("output", "coeffs"):
            torch.testing.assert_close(
                cpu_outputs[name],
                torch_outputs[name].cpu(),
                rtol=1e-5,
                atol=5e-6,
            )

        slang_model.zero_grad(set_to_none=True)
        torch_cuda_model.zero_grad(set_to_none=True)
        slang_gradient_output = slang_model(
            {name: value.clone() for name, value in inputs.items()}
        )["output"]
        torch_gradient_output = torch_cuda_model(
            {name: value.clone() for name, value in inputs.items()}
        )["output"]
        cotangent = torch.zeros_like(slang_gradient_output)
        cotangent[0, 0, 24, 24] = 1.0
        (slang_gradient_output * cotangent).sum().backward()
        (torch_gradient_output * cotangent).sum().backward()
        slang_parameters = dict(slang_model.get_neural_network().named_parameters())
        torch_parameters = dict(
            torch_cuda_model.get_neural_network().named_parameters()
        )
        compared = 0
        for name in slang_parameters:
            slang_gradient = slang_parameters[name].grad
            torch_gradient = torch_parameters[name].grad
            if slang_gradient is None or torch_gradient is None:
                continue
            torch.testing.assert_close(
                torch_gradient, slang_gradient, rtol=1e-4, atol=1e-5
            )
            compared += 1
            if compared == 2:
                break
        self.assertEqual(compared, 2)
