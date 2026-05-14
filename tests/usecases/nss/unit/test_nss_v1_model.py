# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from ng_model_gym.core.model import BaseNGModel, create_model
from ng_model_gym.core.utils.enum_definitions import TrainEvalMode
from ng_model_gym.usecases.nss.model.model_blocks_v1 import AutoEncoderV1
from tests.testing_utils import create_simple_params


class TestNSSV1Model(unittest.TestCase):
    """Tests for NSS v1 model registration and public guardrails."""

    def _data_creator_helper(self, lr_h, lr_w, hr_h, hr_w, recurrence=None):
        """Create NSS v1 recurrent tensors for a forward pass."""

        if recurrence is None:
            recurrence = self.recurrence

        render_size = torch.zeros(self.batch, recurrence, 2, 1, 1)
        render_size[:, :, 0, :, :] = lr_h
        render_size[:, :, 1, :, :] = lr_w

        data = {
            "colour_linear": torch.rand(self.batch, recurrence, 3, lr_h, lr_w),
            "depth": torch.rand(self.batch, recurrence, 1, lr_h, lr_w),
            "depth_params": torch.rand(self.batch, recurrence, 4, lr_h, lr_w),
            "ground_truth_linear": torch.rand(self.batch, recurrence, 3, hr_h, hr_w),
            "jitter": torch.zeros(self.batch, recurrence, 2, 1, 1),
            "motion": torch.zeros(self.batch, recurrence, 2, hr_h, hr_w),
            "motion_lr": torch.zeros(self.batch, recurrence, 2, lr_h, lr_w),
            "render_size": render_size,
            "seq": torch.ones(self.batch, recurrence, 1, 1, 1),
            "exposure": torch.ones(self.batch, recurrence, 1, 1, 1),
        }
        return {key: tensor.to(self.device) for key, tensor in data.items()}

    def _stub_core_forward(self, model):
        """Replace Slang-backed core_forward with a CPU-safe recorder."""

        captured_inputs = []

        def core_forward(inputs):
            captured_inputs.append(
                {
                    key: value.detach().clone()
                    for key, value in inputs.items()
                    if isinstance(value, torch.Tensor)
                }
            )
            output_value = float(len(captured_inputs))
            output_linear = torch.full_like(inputs["history"], output_value)
            return {
                "output": output_linear,
                "output_linear": output_linear,
                "out_filtered": output_linear,
                "temporal_params": torch.full_like(
                    inputs["temporal_params_tm1"],
                    output_value,
                ),
                "derivative": torch.full_like(
                    inputs["derivative_tm1"],
                    output_value,
                ),
            }

        model.core_forward = core_forward
        return captured_inputs

    def setUp(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = create_simple_params(usecase="nss_v1")
        self.params.model_train_eval_mode = TrainEvalMode.FP32
        self.params.train.batch_size = 2
        self.params.model.recurrent_samples = 4
        self.batch = self.params.train.batch_size
        self.recurrence = self.params.model.recurrent_samples

    def test_create_nss_v1_model(self) -> None:
        """NSS v1 creates a BaseNGModel with AutoEncoderV1."""

        model = create_model(self.params, self.device)

        self.assertIsInstance(model, BaseNGModel)
        self.assertIsInstance(model.get_neural_network(), AutoEncoderV1)
        self.assertTrue(model.shader_accurate)

    def test_device_reports_autoencoder_device(self) -> None:
        """NSS v1 device tracks the trainable network device."""

        model = create_model(self.params, self.device)

        self.assertEqual(
            model.device,
            next(model.get_neural_network().parameters()).device,
        )

        model.get_neural_network().to(torch.device("meta"))

        self.assertEqual(
            model.device,
            next(model.get_neural_network().parameters()).device,
        )

    def test_qat_is_not_supported_yet(self) -> None:
        """NSS v1 raises clearly for QAT until follow-up work lands."""

        self.params.model_train_eval_mode = TrainEvalMode.QAT_INT8

        with self.assertRaisesRegex(NotImplementedError, "NSS-v1 QAT"):
            create_model(self.params, self.device)

    def test_dynamic_export_shape_is_not_supported_yet(self) -> None:
        """NSS v1 raises clearly for export until follow-up work lands."""

        model = create_model(self.params, self.device)

        with self.assertRaisesRegex(NotImplementedError, "NSS-v1 export"):
            model.define_dynamic_export_model_input()

    def test_core_forward_requires_cuda_for_slang_path(self) -> None:
        """Real NSS v1 Slang-backed forward fails clearly on CPU."""

        model = create_model(self.params, torch.device("cpu"))
        one_frame = {
            key: tensor[:, 0, :, :, :].cpu()
            for key, tensor in self._data_creator_helper(
                128, 128, 256, 256, recurrence=1
            ).items()
        }
        one_frame = model.set_buffers(one_frame)

        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            model.core_forward(one_frame)

    def test_non_multiple_lr_shapes_pad_temporal_state_only(self) -> None:
        """NSS v1 pads autoencoder input/state but keeps raw derivative shapes."""

        model = create_model(self.params, self.device)
        one_frame = {
            key: tensor[:, 0, :, :, :]
            for key, tensor in self._data_creator_helper(130, 132, 260, 264).items()
        }
        one_frame = model.set_buffers(one_frame)

        dispatch_dims = model._calculate_dispatch_dims(one_frame)
        input_shape, process_shape, hr_shape, pad_shape, depth_shape = dispatch_dims

        self.assertEqual(input_shape, (self.batch, 3, 130, 132))
        self.assertEqual(process_shape, (self.batch, 3, 130, 132))
        self.assertEqual(hr_shape, (self.batch, 3, 260, 264))
        self.assertEqual(pad_shape, (self.batch, 3, 136, 136))
        self.assertEqual(depth_shape, (self.batch, 1, 65, 66))
        self.assertEqual(
            one_frame["temporal_params_tm1"].shape,
            (self.batch, 4, 136, 136),
        )
        self.assertEqual(
            one_frame["derivative_tm1"].shape,
            (self.batch, 2, 130, 132),
        )

    def test_forward_uses_available_time_dimension(self) -> None:
        """NSS v1 does not index past the input recurrent dimension."""

        self.params.model.recurrent_samples = 16
        model = create_model(self.params, self.device)
        captured_inputs = self._stub_core_forward(model)
        data = self._data_creator_helper(128, 128, 256, 256, recurrence=1)

        model_out = model(data)

        self.assertEqual(len(captured_inputs), 1)
        self.assertEqual(model_out["output_linear"].shape[1], 1)

    def test_split_inputs_over_time_matches_direct_time_indexing(self) -> None:
        """Pre-split per-frame inputs match the previous direct indexing."""

        model = create_model(self.params, self.device)
        data = self._data_creator_helper(128, 128, 256, 256, recurrence=3)

        frames = model._split_inputs_over_time(data, sequence_length=3)

        self.assertEqual(len(frames), 3)
        for t, frame in enumerate(frames):
            direct_frame = model._get_input_data_at_t(data, t=t)
            self.assertEqual(frame.keys(), direct_frame.keys())
            for key, value in frame.items():
                torch.testing.assert_close(
                    value,
                    direct_frame[key],
                    rtol=0,
                    atol=0,
                    msg=key,
                )

    def test_split_inputs_over_time_reuses_non_recurrent_metadata(self) -> None:
        """Non-recurrent tensors are passed through to every per-frame input."""

        model = create_model(self.params, self.device)
        data = {
            "colour_linear": torch.arange(
                self.batch * 2 * 3 * 4 * 4,
                device=self.device,
                dtype=torch.float32,
            ).reshape(self.batch, 2, 3, 4, 4),
            "metadata": torch.arange(
                self.batch * 2,
                device=self.device,
                dtype=torch.float32,
            ).reshape(self.batch, 2),
        }

        frames = model._split_inputs_over_time(data, sequence_length=2)

        self.assertEqual(len(frames), 2)
        self.assertIs(frames[0]["metadata"], data["metadata"])
        self.assertIs(frames[1]["metadata"], data["metadata"])
        torch.testing.assert_close(
            frames[0]["colour_linear"],
            data["colour_linear"][:, 0],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            frames[1]["colour_linear"],
            data["colour_linear"][:, 1],
            rtol=0,
            atol=0,
        )

    def test_forward_exposes_loss_context_keys(self) -> None:
        """NSS v1 recurrent output includes loss context tensors."""

        model = create_model(self.params, self.device)
        self._stub_core_forward(model)
        data = self._data_creator_helper(128, 128, 256, 256, recurrence=2)
        data["motion"][:, 0, ...] = 5.0
        data["motion"][:, 1, ...] = 7.0

        model_out = model(data)

        self.assertIn("motion", model_out)
        self.assertIn("reset_event", model_out)
        self.assertIs(model_out["motion"], data["motion"])
        self.assertEqual(model_out["motion"].shape, data["motion"].shape)
        self.assertEqual(model_out["reset_event"].shape, data["seq"].shape)
        torch.testing.assert_close(
            model_out["motion"][:, 0, ...],
            torch.full_like(model_out["motion"][:, 0, ...], 5.0),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model_out["motion"][:, 1, ...],
            torch.full_like(model_out["motion"][:, 1, ...], 7.0),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model_out["reset_event"][:, 0, ...],
            torch.zeros_like(model_out["reset_event"][:, 0, ...]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model_out["reset_event"][:, 1, ...],
            torch.ones_like(model_out["reset_event"][:, 1, ...]),
            rtol=0,
            atol=0,
        )

    def test_forward_overwrites_loss_context_keys(self) -> None:
        """NSS v1 recurrent output ignores core-owned loss context tensors."""

        model = create_model(self.params, self.device)

        def core_forward(inputs):
            output_linear = torch.zeros_like(inputs["history"])
            return {
                "output": output_linear,
                "output_linear": output_linear,
                "out_filtered": output_linear,
                "temporal_params": torch.zeros_like(inputs["temporal_params_tm1"]),
                "derivative": torch.zeros_like(inputs["derivative_tm1"]),
                "motion": torch.full_like(inputs["motion"], -99.0),
                "reset_event": torch.full_like(inputs["reset_event"], -42.0),
            }

        model.core_forward = core_forward
        data = self._data_creator_helper(128, 128, 256, 256, recurrence=2)
        data["motion"][:, 0, ...] = 5.0
        data["motion"][:, 1, ...] = 7.0

        model_out = model(data)

        self.assertIs(model_out["motion"], data["motion"])
        torch.testing.assert_close(
            model_out["motion"][:, 0, ...],
            torch.full_like(model_out["motion"][:, 0, ...], 5.0),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model_out["motion"][:, 1, ...],
            torch.full_like(model_out["motion"][:, 1, ...], 7.0),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model_out["reset_event"][:, 0, ...],
            torch.zeros_like(model_out["reset_event"][:, 0, ...]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model_out["reset_event"][:, 1, ...],
            torch.ones_like(model_out["reset_event"][:, 1, ...]),
            rtol=0,
            atol=0,
        )

    def test_forward_preserves_history_across_same_sequence_calls(self) -> None:
        """NSS v1 keeps recurrent history across forward calls with matching seq."""

        model = create_model(self.params, self.device)
        captured_inputs = self._stub_core_forward(model)
        data = self._data_creator_helper(128, 128, 256, 256, recurrence=1)

        model(data)
        model(data)

        self.assertTrue(torch.all(captured_inputs[0]["history"] == 0))
        self.assertTrue(torch.all(captured_inputs[1]["history"] == 1))

    def test_forward_resets_history_when_sequence_changes(self) -> None:
        """NSS v1 resets recurrent history when seq changes."""

        model = create_model(self.params, self.device)
        captured_inputs = self._stub_core_forward(model)
        data = self._data_creator_helper(128, 128, 256, 256, recurrence=1)
        next_sequence_data = self._data_creator_helper(
            128,
            128,
            256,
            256,
            recurrence=1,
        )
        next_sequence_data["seq"] = torch.full_like(next_sequence_data["seq"], 2.0)

        model(data)
        model(next_sequence_data)

        self.assertTrue(torch.all(captured_inputs[0]["history"] == 0))
        self.assertTrue(torch.all(captured_inputs[1]["history"] == 0))

    def test_forward_zeroes_motion_on_initial_and_reset_frames(self) -> None:
        """NSS v1 zeroes motion inputs when recurrent state is reset."""

        model = create_model(self.params, self.device)
        captured_inputs = self._stub_core_forward(model)
        data = self._data_creator_helper(128, 128, 256, 256, recurrence=1)
        data["motion"].fill_(7.0)
        data["motion_lr"].fill_(3.0)
        next_sequence_data = self._data_creator_helper(
            128,
            128,
            256,
            256,
            recurrence=1,
        )
        next_sequence_data["motion"].fill_(7.0)
        next_sequence_data["motion_lr"].fill_(3.0)
        next_sequence_data["seq"] = torch.full_like(next_sequence_data["seq"], 2.0)

        model(data)
        model(data)
        model(next_sequence_data)

        self.assertTrue(torch.all(captured_inputs[0]["motion"] == 0))
        self.assertTrue(torch.all(captured_inputs[0]["motion_lr"] == 0))
        self.assertTrue(torch.all(captured_inputs[1]["motion"] == 7))
        self.assertTrue(torch.all(captured_inputs[1]["motion_lr"] == 3))
        self.assertTrue(torch.all(captured_inputs[2]["motion"] == 0))
        self.assertTrue(torch.all(captured_inputs[2]["motion_lr"] == 0))

    def test_gt_history_augmentation_uses_linear_initial_history_during_training(
        self,
    ) -> None:
        """GT history augmentation seeds first-frame history from linear GT."""

        self.params.model.gt_history_augmentation = True
        self.params.model.gt_history_augmentation_chance = 100.0
        model = create_model(self.params, self.device)
        model.train()
        captured_inputs = self._stub_core_forward(model)
        data = self._data_creator_helper(128, 128, 256, 256, recurrence=2)
        data["ground_truth_linear"].fill_(0.25)
        y_true = torch.full(
            (self.batch, self.recurrence, 3, 256, 256),
            0.75,
            device=self.device,
        )
        model.y_true = y_true

        model(data)

        torch.testing.assert_close(
            captured_inputs[0]["history"],
            data["ground_truth_linear"][:, 0, ...],
            rtol=0,
            atol=0,
        )
        self.assertTrue(torch.all(captured_inputs[0]["reset_event"] == 1.0))
        self.assertTrue(torch.all(captured_inputs[1]["history"] == 1.0))

    def test_gt_history_augmentation_chance_zero_preserves_reset_history(self) -> None:
        """Chance zero leaves the reset history untouched."""

        self.params.model.gt_history_augmentation = True
        self.params.model.gt_history_augmentation_chance = 0.0
        model = create_model(self.params, self.device)
        model.train()
        captured_inputs = self._stub_core_forward(model)
        data = self._data_creator_helper(128, 128, 256, 256, recurrence=1)
        model.y_true = torch.ones(
            self.batch,
            self.recurrence,
            3,
            256,
            256,
            device=self.device,
        )

        model(data)

        self.assertTrue(torch.all(captured_inputs[0]["history"] == 0.0))
        self.assertTrue(torch.all(captured_inputs[0]["reset_event"] == 0.0))

    def test_gt_history_augmentation_is_training_only(self) -> None:
        """GT history augmentation is disabled in eval mode."""

        self.params.model.gt_history_augmentation = True
        self.params.model.gt_history_augmentation_chance = 100.0
        model = create_model(self.params, self.device)
        model.eval()
        captured_inputs = self._stub_core_forward(model)
        data = self._data_creator_helper(128, 128, 256, 256, recurrence=1)
        model.y_true = torch.ones(
            self.batch,
            self.recurrence,
            3,
            256,
            256,
            device=self.device,
        )

        model(data)

        self.assertTrue(torch.all(captured_inputs[0]["history"] == 0.0))

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "NSS v1 forward requires CUDA because the forward path is Slang-backed.",
    )
    def test_shape_nss_v1_model_forward_pass(self) -> None:
        """Test NSS v1 recurrent high-quality forward output shapes."""

        model = create_model(self.params, self.device)
        data = self._data_creator_helper(128, 128, 256, 256)

        with torch.no_grad():
            model.train()
            model_out = model(data)

        self.assertEqual(
            model_out["output_linear"].shape,
            (self.batch, self.recurrence, 3, 256, 256),
        )
        self.assertEqual(
            model_out["output"].shape,
            (self.batch, self.recurrence, 3, 256, 256),
        )
        self.assertEqual(
            model_out["out_filtered"].shape,
            (self.batch, self.recurrence, 3, 256, 256),
        )
        self.assertEqual(
            model_out["temporal_params"].shape,
            (self.batch, self.recurrence, 4, 128, 128),
        )
        self.assertEqual(
            model_out["derivative"].shape,
            (self.batch, self.recurrence, 2, 128, 128),
        )
