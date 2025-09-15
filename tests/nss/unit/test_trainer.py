# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import torch
from torch import nn, optim

from ng_model_gym.core.trainer.trainer import Trainer
from ng_model_gym.core.utils.types import TrainEvalMode
from tests.unit.utils.utils import create_simple_params

# pylint: disable=abstract-method, unsubscriptable-object


class TinyModel(nn.Module):
    """Simple test model"""

    def __init__(self):
        super().__init__()
        self.nss_model = None
        self.layer = nn.Linear(1, 1)
        self.layer.weight.data = torch.full((1, 1), 0.5)
        self.layer.bias.data = torch.full((1, 1), 0.4)

    def forward(self, _):
        """Forward pass returns a dict containing a mock"""
        return {"x": Mock()}


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
        self.mock_trainer.training_mode_params.checkpoints.save_frequency = 999
        self.mock_trainer.starting_epoch = 1
        self.mock_trainer.device = torch.device("cpu")
        self.mock_trainer.metrics = []

        # --- Model ---
        self.mock_trainer.model.nss_model = Mock()
        self.mock_trainer.model.nss_model.reset_history_buffers = Mock()
        self.mock_trainer.model.detach_buffers = Mock()

        # --- Dataloader ---
        mock_input = Mock()
        mock_input.to = Mock(return_value=mock_input)
        mock_ground_truth = Mock()
        mock_ground_truth.to = Mock(return_value=mock_ground_truth)
        self.mock_trainer.train_dataloader = [
            ({"x": mock_input}, mock_ground_truth) for _ in range(10)
        ]

        # --- Loss / Criterion ---
        mock_loss = Mock()
        mock_loss.backward = Mock()
        mock_loss.item = Mock(return_value=0.1)
        self.mock_trainer.criterion = Mock(return_value=(mock_loss, None))

        # --- Other ---
        self.mock_trainer.lr_schedule = None
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
        self.mock_trainer.validate_epochs = (
            self.mock_trainer.params.train.validate_frequency
        )

        # Run the real training loop, passing mock as 'self'
        Trainer.train(self.mock_trainer)

        called_epochs = [c.args[0] for c in self.mock_trainer.validate.call_args_list]
        self.assertEqual(called_epochs, [3, 6, 9])

    def test_train_calls_validate_on_specific_epochs(self):
        """Ensure Trainer.train() only triggers validate() on configured epochs"""
        self.mock_trainer.params = Mock()
        self.mock_trainer.params.train.perform_validate = True
        self.mock_trainer.params.train.validate_frequency = [1, 2, 5, 9]
        self.mock_trainer.validate_epochs = (
            self.mock_trainer.params.train.validate_frequency
        )

        # Run the real training loop, passing mock as 'self'
        Trainer.train(self.mock_trainer)

        called_epochs = [c.args[0] for c in self.mock_trainer.validate.call_args_list]
        self.assertEqual(called_epochs, [1, 2, 5, 9])

    def test_saving_checkpoints(self):
        """Test expected checkpoints are saved during mock training run"""
        trainer = self.mock_trainer

        # Test all possible combinations of starting epoch, max epoch and save_frequency
        test_range = 25
        for max_epochs in range(1, test_range):
            for starting_epoch in range(1, max_epochs + 1):
                for save_frequency in range(1, max_epochs + 1):
                    with self.subTest(
                        starting_epoch=starting_epoch,
                        max_epochs=max_epochs,
                        save_frequency=save_frequency,
                    ):
                        # Create temp dir for saving checkpoints
                        with tempfile.TemporaryDirectory() as temp_dir:
                            time_stamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")
                            save_dir = Path(temp_dir, time_stamp)
                            save_dir.mkdir(parents=True, exist_ok=True)
                            trainer.model_save_path = save_dir

                            # Mock training run
                            total_epochs = max_epochs
                            for mock_epoch in range(starting_epoch, total_epochs + 1):
                                Trainer._save_checkpoint(
                                    trainer,
                                    save_frequency=save_frequency,
                                    current_epoch=mock_epoch,
                                    total_epochs=total_epochs,
                                )

                            # Saves are done every X epochs
                            # and the last epoch in the range 1 ≤ X ≤ max_epoch.
                            # If training is resumed, the cadence of saves should be from the range
                            # 1 ≤ X ≤ max_epoch and not from the epoch resumed.

                            # E.g. If saving every 2nd epoch.
                            # Expected save epochs are 2, 4, 6, 8 etc.
                            # Training resuming from epoch 3 should not save at epoch 5 but rather 4

                            # Expected epoch numbers that should be saved
                            expected_saves_in_dir = []
                            for mock_epoch in range(1, max_epochs + 1):
                                # Skip anything before we resumed
                                if mock_epoch < starting_epoch:
                                    continue

                                # Epoch matches save frequency
                                if mock_epoch % save_frequency == 0:
                                    expected_saves_in_dir.append(mock_epoch)
                                    continue

                                # Always save last epoch
                                if mock_epoch == max_epochs:
                                    expected_saves_in_dir.append(mock_epoch)

                            epoch_files = save_dir.glob("ckpt-*.pt")
                            found_saves_in_dir = []
                            for file in epoch_files:
                                # E.g. file is called ckpt-3.pt
                                num_str = file.name.removeprefix("ckpt-").removesuffix(
                                    ".pt"
                                )
                                found_saves_in_dir.append(int(num_str))

                            # No guarantee of order files are read
                            expected_saves_in_dir.sort()
                            found_saves_in_dir.sort()

                            self.assertEqual(
                                found_saves_in_dir,
                                expected_saves_in_dir,
                                f"save_frequency={save_frequency}, "
                                f"starting_epoch={starting_epoch}, "
                                f"max_epochs={max_epochs}: "
                                f"expected {expected_saves_in_dir}, found {found_saves_in_dir}",
                            )

    def test_restoring_model_weights(self):
        """Test model weights are restored correctly"""

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
            Trainer._save_checkpoint(
                self.mock_trainer, save_frequency=1, current_epoch=10, total_epochs=15
            )

            # Fake a new training run (e.g. re-run training after interruption)
            mock_resume_trainer = Mock(spec=Trainer)
            mock_resume_trainer.model = TinyModel()
            mock_resume_trainer.optimizer = optim.Adam(
                self.mock_trainer.model.parameters(), lr=0.001
            )

            mock_resume_trainer.params = create_simple_params()
            mock_resume_trainer.params.train.resume = True
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


if __name__ == "__main__":
    unittest.main()
