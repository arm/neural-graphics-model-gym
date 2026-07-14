# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import MethodType
from unittest.mock import Mock

import torch
from torch import nn, optim

from ng_model_gym.core.loss import LossV1
from ng_model_gym.core.optimizers import LARS
from ng_model_gym.core.trainer import get_loss_fn, get_optimizer_type, Trainer
from ng_model_gym.core.utils.enum_definitions import (
    LossFn,
    OptimizerType,
    TrainEvalMode,
)
from tests.testing_utils import create_simple_params

# pylint: disable=abstract-method, duplicate-code, unsubscriptable-object


class TinyModel(nn.Module):
    """Simple test model"""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self.layer.weight.data = torch.full((1, 1), 0.5)
        self.layer.bias.data = torch.full((1, 1), 0.4)

    def forward(self, _):
        """Forward pass returns a dict containing a mock"""
        return {"output": torch.zeros(1, 1)}

    def on_train_epoch_start(self) -> None:
        """hook for trainer tests"""
        return None

    def on_train_epoch_end(self) -> None:
        """hook for trainer tests"""
        return None

    def on_train_batch_start(self) -> None:
        """hook for trainer tests"""
        return None

    def on_before_batch_transfer(self, batch):
        """hook for trainer tests"""
        return batch

    def on_after_batch_transfer(self, batch):
        """hook for trainer tests"""
        return batch

    def on_train_batch_end(self) -> None:
        """hook for trainer tests"""
        return None

    def on_train_end(self) -> None:
        """hook for trainer tests"""
        return None

    def on_validation_start(self) -> None:
        """hook for trainer tests"""
        return None

    def on_validation_end(self) -> None:
        """hook for trainer tests"""
        return None


class TinyModelWithWeightsLoadHook(TinyModel):
    """Tiny model that records weights-only checkpoint hook calls."""

    def __init__(self):
        super().__init__()
        self.weights_seen_state_dict = None

    def prepare_checkpoint_state_dict_for_weights_load(self, state_dict):
        """Record weights-only hook input and return loadable weights."""
        self.weights_seen_state_dict = state_dict
        prepared = self.state_dict()
        prepared["layer.weight"] = torch.full_like(prepared["layer.weight"], 0.7)
        prepared["layer.bias"] = torch.full_like(prepared["layer.bias"], 0.8)
        return prepared


class TestTrainerMethods(unittest.TestCase):
    """Tests for Trainer class"""

    def setUp(self):
        """Setup before each test"""
        torch.manual_seed(1)
        logging.disable()
        self.mock_trainer = Mock(spec=Trainer)
        self.mock_trainer.model = TinyModel()
        self.mock_trainer.optimizer = optim.Adam(
            self.mock_trainer.model.parameters(), lr=0.001
        )

        # --- Config ---
        self.mock_trainer.training_mode_params = Mock()
        self.mock_trainer.training_mode_params.number_of_epochs = 10
        self.mock_trainer.starting_epoch = 1
        self.mock_trainer.device = torch.device("cpu")
        self.mock_trainer.train_metrics = []
        self.mock_trainer.val_metrics = []
        self.mock_trainer.is_feedback = False

        # --- Model ---
        self.mock_trainer.model.reset_history_buffers = Mock()
        self.mock_trainer.model.detach_buffers = Mock()

        # --- Dataloader ---
        self.mock_input = torch.zeros((1, 1))
        self.mock_ground_truth = torch.zeros((1, 1))
        self.mock_trainer.train_dataloader = [
            ({"x": self.mock_input.clone()}, self.mock_ground_truth.clone())
            for _ in range(10)
        ]

        # --- Loss / Criterion ---
        mock_loss = Mock()
        mock_loss.backward = Mock()
        mock_loss.item = Mock(return_value=0.1)
        self.mock_trainer.criterion = Mock(return_value=mock_loss)
        self.mock_trainer._training_model = self.mock_trainer.model
        self.mock_trainer._training_loss = self.mock_trainer.criterion
        self.mock_trainer._train_step = MethodType(
            Trainer._train_step, self.mock_trainer
        )

        # --- Other ---
        self.mock_trainer.lr_schedule = Mock()
        self.mock_trainer.lr_schedule.state_dict.return_value = {}
        self.mock_trainer.lr_schedule.load_state_dict = Mock()
        self.mock_trainer.lr_schedule.step = Mock()
        self.mock_trainer._save_checkpoint = Mock()
        self.mock_trainer.validate = Mock()

    def tearDown(self):
        """Re-enable logging"""
        logging.disable(logging.NOTSET)

    def test_train_calls_validate_every_3_epochs(self):
        """Ensure Trainer.train() only triggers validate() on configured epochs"""
        self.mock_trainer.params = Mock()
        self.mock_trainer.params.train.perform_validate = True
        self.mock_trainer.params.train.validate_frequency = 3

        # Bind the real method to the mock
        # pylint: disable=E1120
        self.mock_trainer._should_validate = Trainer._should_validate.__get__(
            self.mock_trainer, Trainer
        )  # pylint: enable=E1120

        # Run the real training loop, passing mock as 'self'
        Trainer.train(self.mock_trainer)

        called_epochs = [c.args[0] for c in self.mock_trainer.validate.call_args_list]
        self.assertEqual(called_epochs, [3, 6, 9])

    def test_train_calls_validate_on_specific_epochs(self):
        """Ensure Trainer.train() only triggers validate() on configured epochs"""
        self.mock_trainer.params = Mock()
        self.mock_trainer.params.train.perform_validate = True
        self.mock_trainer.params.train.validate_frequency = [1, 2, 5, 9]

        # Bind the real method to the mock
        # pylint: disable=E1120
        self.mock_trainer._should_validate = Trainer._should_validate.__get__(
            self.mock_trainer, Trainer
        )  # pylint: enable=E1120

        # Run the real training loop, passing mock as 'self'
        Trainer.train(self.mock_trainer)

        called_epochs = [c.args[0] for c in self.mock_trainer.validate.call_args_list]
        self.assertEqual(called_epochs, [1, 2, 5, 9])

    def test_lifecycle_hooks(self):
        """Test trainer lifecycle hooks are called"""
        trainer = self.mock_trainer
        trainer.model.on_train_epoch_start = Mock()
        trainer.model.on_train_batch_start = Mock()
        trainer.model.on_train_batch_end = Mock()
        trainer.model.on_train_epoch_end = Mock()
        trainer.model.on_train_end = Mock()
        trainer.model.on_validation_start = Mock()
        trainer.model.on_validation_end = Mock()
        trainer.model.on_before_batch_transfer = Mock(
            spec=TinyModel.on_before_batch_transfer
        )
        trainer.model.on_before_batch_transfer.side_effect = lambda batch: batch
        trainer.model.on_after_batch_transfer = Mock(
            spec=TinyModel.on_after_batch_transfer
        )
        trainer.model.on_after_batch_transfer.side_effect = lambda batch: batch

        trainer.training_mode_params.number_of_epochs = 2
        trainer.starting_epoch = 1

        trainer.params = Mock()
        trainer.params.train.perform_validate = True
        trainer.params.train.validate_frequency = 1

        mock_input = torch.zeros((1, 1))
        mock_ground_truth = torch.zeros((1, 1))
        trainer.train_dataloader = [
            ({"x": mock_input}, mock_ground_truth) for _ in range(3)
        ]
        trainer.val_dataloader = [
            ({"x": mock_input}, mock_ground_truth) for _ in range(2)
        ]

        # Get the actual validate functions from Trainer
        # pylint: disable=no-value-for-parameter, assignment-from-no-return
        trainer._should_validate = Trainer._should_validate.__get__(trainer, Trainer)
        trainer.validate = Trainer.validate.__get__(trainer, Trainer)
        # pylint: enable=no-value-for-parameter, assignment-from-no-return

        Trainer.train(trainer)

        self.assertEqual(trainer.model.on_train_epoch_start.call_count, 2)
        self.assertEqual(trainer.model.on_before_batch_transfer.call_count, 10)
        self.assertEqual(trainer.model.on_after_batch_transfer.call_count, 10)
        self.assertEqual(trainer.model.on_train_batch_start.call_count, 6)
        self.assertEqual(trainer.model.on_train_batch_start.call_count, 6)
        self.assertEqual(trainer.model.on_train_batch_end.call_count, 6)
        self.assertEqual(trainer.model.on_train_epoch_end.call_count, 2)
        self.assertEqual(trainer.model.on_train_end.call_count, 1)
        self.assertEqual(trainer.model.on_validation_start.call_count, 2)
        self.assertEqual(trainer.model.on_validation_end.call_count, 2)

    def test_save_checkpoint_overwrites_best_ckpt_correctly(self):
        """Test that best checkpoint is overwritten only when loss improves"""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self.mock_trainer
            trainer.model_save_path = temp_dir
            trainer.params = Mock()
            trainer.params.train.perform_validate = True
            trainer.avg_val_loss = 0.25  # lower is better

            # First save should store best
            Trainer._save_checkpoint(trainer, current_epoch=1)

            best_ckpt_path = Path(temp_dir, "best-validated-ckpt.pt")
            best_meta_path = Path(temp_dir, "best-validated-ckpt.meta.json")

            self.assertTrue(best_ckpt_path.exists(), "Best checkpoint was not saved")
            self.assertTrue(
                best_meta_path.exists(), "Best checkpoint metadata was not saved"
            )

            with open(best_meta_path, "r", encoding="utf-8") as f:
                from_json = json.load(f)
            self.assertEqual(from_json["val_loss"], 0.25)

            # Simulate worse next epoch
            trainer.avg_val_loss = 0.30
            Trainer._save_checkpoint(trainer, current_epoch=2)

            with open(best_meta_path, "r", encoding="utf-8") as f:
                from_json2 = json.load(f)

            self.assertEqual(
                from_json2["val_loss"], 0.25, "Metadata was overwritten by worse loss"
            )
            self.assertEqual(
                from_json2["epoch"], 1, "Epoch should not change on worse loss"
            )

            # Simulate better epoch
            trainer.avg_val_loss = 0.10
            Trainer._save_checkpoint(trainer, current_epoch=3)

            with open(best_meta_path, "r", encoding="utf-8") as f:
                from_json3 = json.load(f)

            self.assertEqual(
                from_json3["val_loss"], 0.10, "Metadata was not updated on better loss"
            )
            self.assertEqual(
                from_json3["epoch"], 3, "Epoch not updated correctly for new best"
            )

    def test_restoring_model_weights(self):
        """Test model weights are restored correctly"""
        self.mock_trainer.params = Mock()
        self.mock_trainer.params.train.perform_validate = False

        # Create temp dir for saving checkpoints
        with tempfile.TemporaryDirectory() as temp_dir:
            time_stamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            model_save_path = Path(temp_dir, "checkpoints", time_stamp)
            model_save_path.mkdir(exist_ok=True, parents=True)

            # Fake a training step (update model/optimizer state dict values)
            with torch.no_grad():
                self.mock_trainer.model.layer.weight.data = torch.full((1, 1), 0.9)
                self.mock_trainer.model.layer.bias.data = torch.full((1, 1), 0.1)
            self.mock_trainer.optimizer.state["state"]["mock_data"] = 0.1234

            self.mock_trainer.model_save_path = model_save_path
            # Create checkpoint ckpt-10.pt
            Trainer._save_checkpoint(self.mock_trainer, current_epoch=10)

            # Fake a new training run (e.g. re-run training after interruption)
            mock_resume_trainer = Mock(spec=Trainer)
            mock_resume_trainer.model = TinyModel()
            mock_resume_trainer.optimizer = optim.Adam(
                self.mock_trainer.model.parameters(), lr=0.001
            )
            mock_resume_trainer.lr_schedule = Mock()
            mock_resume_trainer.lr_schedule.state_dict.return_value = {}
            mock_resume_trainer.lr_schedule.load_state_dict = Mock()
            mock_resume_trainer.lr_schedule.step = Mock()

            mock_resume_trainer.params = create_simple_params(usecase="nss-v1")
            mock_resume_trainer.params.train.resume = Path(
                model_save_path, "ckpt-10.pt"
            )
            mock_resume_trainer.params.train.fp32.number_of_epochs = 15
            mock_resume_trainer.params.train.fp32.checkpoints.dir = (
                model_save_path.parent
            )

            mock_resume_trainer.params.model_train_eval_mode = TrainEvalMode.FP32
            mock_resume_trainer.training_mode_params = (
                mock_resume_trainer.params.train.fp32
            )

            Trainer._restore_model_weights(mock_resume_trainer)

            new_model_state_dict = mock_resume_trainer.model.state_dict()
            new_optimizer_state_dict = mock_resume_trainer.optimizer.state_dict()

            # Check updated params are present in saved dict
            self.assertTrue(
                torch.equal(new_model_state_dict["layer.weight"], torch.tensor([[0.9]]))
            )
            self.assertTrue(
                torch.equal(new_model_state_dict["layer.bias"], torch.tensor([[0.1]]))
            )
            self.assertEqual(
                new_optimizer_state_dict["state"]["state"]["mock_data"], 0.1234
            )

            # Check epoch after resuming from saved checkpoint is one after
            self.assertEqual(mock_resume_trainer.starting_epoch, 11)

    def test_finetune_calls_weights_only_prepare_hook(self):
        """Finetune should call optional weights-only preparation before load."""
        model = TinyModelWithWeightsLoadHook()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir, "finetune.pt")
            finetune_state = {
                "layer.weight": torch.full((1, 1), -1.0),
                "layer.bias": torch.full_like(model.state_dict()["layer.bias"], -2.0),
            }
            torch.save({"model_state_dict": finetune_state}, checkpoint_path)

            trainer = Mock(spec=Trainer)
            trainer.model = model
            trainer.params = create_simple_params(usecase="nss-v1")
            trainer.params.train.resume = None
            trainer.params.train.finetune = checkpoint_path
            trainer.params.model_train_eval_mode = TrainEvalMode.FP32
            trainer.training_mode_params = trainer.params.train.fp32

            Trainer._restore_model_weights(trainer)

        self.assertTrue(
            torch.equal(
                model.weights_seen_state_dict["layer.weight"],
                finetune_state["layer.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                model.weights_seen_state_dict["layer.bias"],
                finetune_state["layer.bias"],
            )
        )
        self.assertTrue(
            torch.equal(model.layer.weight.detach(), torch.full((1, 1), 0.7))
        )
        self.assertTrue(
            torch.equal(
                model.layer.bias.detach(),
                torch.full_like(model.layer.bias.detach(), 0.8),
            )
        )

    def test_qat_finetune_rejects_qat_checkpoint_before_prepare_hook(self):
        """QAT finetune should reject QAT/int8 checkpoints before model hooks."""
        marker_keys = (
            "layer.activation_post_process_0.scale",
            "layer.fake_quant_enabled",
            "layer._param_constant0",
        )

        for marker_key in marker_keys:
            with self.subTest(marker_key=marker_key):
                model = TinyModelWithWeightsLoadHook()

                with tempfile.TemporaryDirectory() as temp_dir:
                    checkpoint_path = Path(temp_dir, "qat-finetune.pt")
                    qat_state = {
                        "layer.weight": torch.full((1, 1), -1.0),
                        "layer.bias": torch.full_like(
                            model.state_dict()["layer.bias"],
                            -2.0,
                        ),
                        marker_key: torch.ones(1),
                    }
                    torch.save({"model_state_dict": qat_state}, checkpoint_path)

                    trainer = Mock(spec=Trainer)
                    trainer.model = model
                    trainer.params = create_simple_params(usecase="nss-v1")
                    trainer.params.train.resume = None
                    trainer.params.train.finetune = checkpoint_path
                    trainer.params.model_train_eval_mode = TrainEvalMode.QAT_INT8
                    trainer.training_mode_params = trainer.params.train.qat

                    with self.assertRaisesRegex(
                        ValueError,
                        "QAT finetune only supports FP32 pretrained weights",
                    ):
                        Trainer._restore_model_weights(trainer)

                self.assertIsNone(model.weights_seen_state_dict)

    def test_qat_finetune_fp32_checkpoint_calls_weights_only_prepare_hook(self):
        """QAT finetune should still allow FP32-shaped pretrained weights."""
        model = TinyModelWithWeightsLoadHook()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir, "fp32-finetune.pt")
            fp32_state = {
                "layer.weight": torch.full((1, 1), -1.0),
                "layer.bias": torch.full_like(model.state_dict()["layer.bias"], -2.0),
            }
            torch.save({"model_state_dict": fp32_state}, checkpoint_path)

            trainer = Mock(spec=Trainer)
            trainer.model = model
            trainer.params = create_simple_params(usecase="nss-v1")
            trainer.params.train.resume = None
            trainer.params.train.finetune = checkpoint_path
            trainer.params.model_train_eval_mode = TrainEvalMode.QAT_INT8
            trainer.training_mode_params = trainer.params.train.qat

            Trainer._restore_model_weights(trainer)

        self.assertTrue(
            torch.equal(
                model.weights_seen_state_dict["layer.weight"],
                fp32_state["layer.weight"],
            )
        )
        self.assertTrue(
            torch.equal(model.layer.weight.detach(), torch.full((1, 1), 0.7))
        )

    def test_train_raises_when_forward_not_dict(self):
        """Trainer should raise TypeError if forward doesn't return a dict."""
        self.mock_trainer.model.forward = Mock(return_value=123)
        with self.assertRaisesRegex(
            TypeError, r"Forward pass must return a dictionary containing 'output' key"
        ):
            Trainer.train(self.mock_trainer)

    def test_train_raises_when_forward_dict_missing_output(self):
        """Trainer should raise TypeError if forward returns a dict without 'output' key"""
        self.mock_trainer.model.forward = Mock(return_value={"random_key": Mock()})
        with self.assertRaisesRegex(
            TypeError, r"Forward pass must return a dictionary containing 'output' key"
        ):
            Trainer.train(self.mock_trainer)

    def test_train_with_different_dataloader_value_types(self):
        """Test type errors not raised when input_dataset or ground_truth
        are either tensors or dictionaries."""
        mock_input = self.mock_input
        mock_ground_truth = self.mock_ground_truth
        self.mock_trainer.training_mode_params.number_of_epochs = 5

        test_dataloader_sample = [
            # Tensor / Tensor
            (mock_input, mock_ground_truth),
            # Dict / Tensor
            ({"input": mock_input}, mock_ground_truth),
            # Tensor / Dict
            (mock_input, {"ground_truth": mock_ground_truth}),
            # Dict / Dict
            ({"input": mock_input}, {"ground_truth": mock_ground_truth}),
            # Tuple / Tensor
            ((mock_input, mock_input), mock_ground_truth),
            # Tensor / Tuple
            (mock_input, (mock_ground_truth, mock_ground_truth)),
            # Tuple / Tuple
            ((mock_input, mock_input), (mock_ground_truth, mock_ground_truth)),
            # List / Tensor
            ([mock_input, mock_input], mock_ground_truth),
            # Tensor / List
            (mock_input, [mock_ground_truth, mock_ground_truth]),
            # List / List
            ([mock_input, mock_input], [mock_ground_truth, mock_ground_truth]),
        ]

        for dataloader_sample in test_dataloader_sample:
            with self.subTest(dataloader_output=dataloader_sample):
                self.mock_trainer.train_dataloader = [
                    dataloader_sample for _ in range(5)
                ]
                try:
                    Trainer.train(self.mock_trainer)
                except TypeError as exc:
                    self.fail(
                        "Unexpected TypeError raised for dataloader "
                        f"output {dataloader_sample}: {exc}"
                    )


class TestLossFnFactory(unittest.TestCase):
    """Tests for loss function factory method: get_loss_fn()"""

    # pylint: disable=C0116
    def test_get_loss_fn_with_valid_loss_v1(self):
        """Test get_loss_fn() returns LossV1 when requested"""
        params = create_simple_params(usecase="nss-v1")
        params.train.loss_fn = LossFn.LOSS_V1.value
        params.train.loss_args = {
            "temporal_reg_weight": 0.1,
            "alpha_reg_weight": 0.2,
            "temporal_reg_channels": 3,
            "min_weight": 0.4,
        }
        device = torch.device("cpu")

        loss_obj = get_loss_fn(params, device)

        self.assertIsInstance(loss_obj, LossV1)
        self.assertEqual(loss_obj.recurrent_samples, params.model.recurrent_samples)
        self.assertEqual(loss_obj.device, device)
        self.assertEqual(loss_obj.loss_args, params.train.loss_args)

    def test_get_loss_fn_with_loss_v1_default_loss_args(self):
        """Test get_loss_fn() preserves the NSS v1 preset loss_args defaults."""
        params = create_simple_params(usecase="nss-v1")
        params.train.loss_fn = LossFn.LOSS_V1.value
        device = torch.device("cpu")

        loss_obj = get_loss_fn(params, device)

        self.assertIsInstance(loss_obj, LossV1)
        self.assertIsNotNone(params.train.loss_args)
        self.assertEqual(loss_obj.loss_args, params.train.loss_args)

    def test_get_loss_fn_with_loss_v1_none_loss_args_defaults_to_empty_dict(self):
        """Test get_loss_fn() defaults LossV1 loss_args to an empty dict when unset."""
        params = create_simple_params(usecase="nss-v1")
        params.train.loss_fn = LossFn.LOSS_V1.value
        params.train.loss_args = None
        device = torch.device("cpu")

        loss_obj = get_loss_fn(params, device)

        self.assertIsInstance(loss_obj, LossV1)
        self.assertIsNone(params.train.loss_args)
        self.assertEqual(loss_obj.loss_args, {})

    def test_get_loss_fn_raises_exception(self):
        params = create_simple_params(usecase="nss-v1")
        params.train.loss_fn = "does_not_exist"
        with self.assertRaises(ValueError):
            _ = get_loss_fn(params, torch.device("cpu"))

    # pylint: enable=C0116


class TestOptimizerFactory(unittest.TestCase):
    """Tests for optimizer factory method: get_optimizer_type()"""

    # pylint: disable=C0116
    def test_get_optimizer_type_with_valid_lars_adam_fp32(self):
        params = create_simple_params(usecase="nss-v1")
        params.train.fp32.optimizer.optimizer_type = OptimizerType.LARS_ADAM.value

        mock_trainer = Mock(spec=Trainer)
        mock_trainer.model = TinyModel()
        mock_trainer.params = params
        mock_trainer.training_mode_params = params.train.fp32

        optimizer_obj = get_optimizer_type(
            mock_trainer.training_mode_params,
            mock_trainer.model.parameters(),
        )
        self.assertIsInstance(optimizer_obj, LARS)
        self.assertEqual(optimizer_obj.optim.__class__, optim.Adam)

    def test_get_optimizer_type_with_valid_lars_adam_qat(self):
        params = create_simple_params(usecase="nss-v1")
        params.train.qat.optimizer.optimizer_type = OptimizerType.LARS_ADAM.value

        mock_trainer = Mock(spec=Trainer)
        mock_trainer.model = TinyModel()
        mock_trainer.params = params
        mock_trainer.training_mode_params = params.train.qat

        optimizer_obj = get_optimizer_type(
            mock_trainer.training_mode_params,
            mock_trainer.model.parameters(),
        )
        self.assertIsInstance(optimizer_obj, LARS)
        self.assertEqual(optimizer_obj.optim.__class__, optim.Adam)

    def test_get_optimizer_type_with_valid_adam_w_fp32(self):
        params = create_simple_params(usecase="nss-v1")
        params.train.fp32.optimizer.optimizer_type = OptimizerType.ADAM_W.value

        mock_trainer = Mock(spec=Trainer)
        mock_trainer.model = TinyModel()
        mock_trainer.params = params
        mock_trainer.training_mode_params = params.train.fp32

        optimizer_obj = get_optimizer_type(
            mock_trainer.training_mode_params,
            mock_trainer.model.parameters(),
        )
        self.assertIsInstance(optimizer_obj, optim.AdamW)

    def test_get_optimizer_type_with_valid_adam_w_qat(self):
        params = create_simple_params(usecase="nss-v1")
        params.train.qat.optimizer.optimizer_type = OptimizerType.ADAM_W.value

        mock_trainer = Mock(spec=Trainer)
        mock_trainer.model = TinyModel()
        mock_trainer.params = params
        mock_trainer.training_mode_params = params.train.qat

        optimizer_obj = get_optimizer_type(
            mock_trainer.training_mode_params,
            mock_trainer.model.parameters(),
        )
        self.assertIsInstance(optimizer_obj, optim.AdamW)

    def test_get_optimizer_type_with_custom_eps_for_adam_w(self):
        params = create_simple_params(usecase="nss-v1")
        params.train.fp32.optimizer.optimizer_type = OptimizerType.ADAM_W.value
        params.train.fp32.optimizer.eps = 1e-5

        mock_trainer = Mock(spec=Trainer)
        mock_trainer.model = TinyModel()
        mock_trainer.params = params
        mock_trainer.training_mode_params = params.train.fp32

        optimizer_obj = get_optimizer_type(
            mock_trainer.training_mode_params,
            mock_trainer.model.parameters(),
        )
        self.assertIsInstance(optimizer_obj, optim.AdamW)
        self.assertAlmostEqual(optimizer_obj.defaults["eps"], 1e-5)

    def test_get_optimizer_type_with_valid_adam_uses_default_eps(self):
        params = create_simple_params(usecase="nss-v1")
        params.train.fp32.optimizer.optimizer_type = OptimizerType.ADAM.value

        mock_trainer = Mock(spec=Trainer)
        mock_trainer.model = TinyModel()
        mock_trainer.params = params
        mock_trainer.training_mode_params = params.train.fp32

        optimizer_obj = get_optimizer_type(
            mock_trainer.training_mode_params,
            mock_trainer.model.parameters(),
        )
        self.assertIsInstance(optimizer_obj, optim.Adam)
        self.assertAlmostEqual(optimizer_obj.defaults["eps"], 1e-7)

    def test_get_optimizer_type_with_custom_eps(self):
        params = create_simple_params(usecase="nss-v1")
        params.train.fp32.optimizer.optimizer_type = OptimizerType.ADAM.value
        params.train.fp32.optimizer.eps = 1e-5

        mock_trainer = Mock(spec=Trainer)
        mock_trainer.model = TinyModel()
        mock_trainer.params = params
        mock_trainer.training_mode_params = params.train.fp32

        optimizer_obj = get_optimizer_type(
            mock_trainer.training_mode_params,
            mock_trainer.model.parameters(),
        )
        self.assertIsInstance(optimizer_obj, optim.Adam)
        self.assertAlmostEqual(optimizer_obj.defaults["eps"], 1e-5)

    def test_get_optimizer_type_with_custom_eps_for_lars_adam(self):
        params = create_simple_params(usecase="nss-v1")
        params.train.fp32.optimizer.optimizer_type = OptimizerType.LARS_ADAM.value
        params.train.fp32.optimizer.eps = 1e-5

        mock_trainer = Mock(spec=Trainer)
        mock_trainer.model = TinyModel()
        mock_trainer.params = params
        mock_trainer.training_mode_params = params.train.fp32

        optimizer_obj = get_optimizer_type(
            mock_trainer.training_mode_params,
            mock_trainer.model.parameters(),
        )
        self.assertIsInstance(optimizer_obj, LARS)
        self.assertEqual(optimizer_obj.optim.__class__, optim.Adam)
        self.assertAlmostEqual(optimizer_obj.optim.defaults["eps"], 1e-5)
        self.assertAlmostEqual(optimizer_obj.eps, 1e-8)

    def test_get_optimizer_type_raises_exception(self):
        params = create_simple_params(usecase="nss-v1")
        params.train.fp32.optimizer.optimizer_type = "does_not_exist"

        mock_trainer = Mock(spec=Trainer)
        mock_trainer.model = TinyModel()
        mock_trainer.params = params
        mock_trainer.training_mode_params = params.train.fp32

        with self.assertRaises(ValueError):
            _ = get_optimizer_type(
                mock_trainer.training_mode_params,
                mock_trainer.model.parameters(),
            )

    # pylint: enable=C0116
