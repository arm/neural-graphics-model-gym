# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ng_model_gym.dataloader import get_dataloader
from ng_model_gym.nss.dataloader.utils import DataLoaderMode
from ng_model_gym.nss.losses import LossV1
from ng_model_gym.nss.metrics import get_metrics
from ng_model_gym.nss.model.lr_scheduler import CosineAnnealingWithWarmupLR
from ng_model_gym.nss.model.model import create_model
from ng_model_gym.nss.model.model_v1 import QATNSSModel
from ng_model_gym.optimizers.lars_adam import lars_adam_torch
from ng_model_gym.utils.checkpoint_utils import latest_checkpoint_path
from ng_model_gym.utils.config_model import ConfigModel, TrainingConfig
from ng_model_gym.utils.general_utils import create_directory
from ng_model_gym.utils.types import LearningRateScheduler, TrainEvalMode

logger = logging.getLogger(__name__)

DEFAULT_EXP_DECAY_RATE = 0.977
DEFAULT_EXP_DECAY_STEP = 100


class Trainer:
    """Instantiates the model, data loader and runs training"""

    def __init__(self, params: ConfigModel):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device is {self.device.type}")

        # Training config parameters based on FP32 or QAT training
        if self.params.model_train_eval_mode == TrainEvalMode.FP32:
            self.training_mode_params = params.train.fp32
        elif self.params.model_train_eval_mode == TrainEvalMode.QAT_INT8:
            self.training_mode_params = params.train.qat
        else:
            raise ValueError("Training/Evaluation mode has not been set")

        self.train_dataloader = get_dataloader(
            self.params,
            num_workers=params.dataset.num_workers,
            prefetch_factor=params.dataset.prefetch_factor,
            loader_mode=DataLoaderMode.TRAIN,
        )
        self.criterion = LossV1(self.params.dataset.recurrent_samples, self.device)
        self.starting_epoch = 1
        self.model: nn.Module = create_model(self.params, self.device).to(self.device)

        logger.info(f"Model architecture: {self.model.nss_model}")
        total_params = sum(p.numel() for p in self.model.nss_model.parameters())
        logger.info(f"Total number of parameters: {total_params:,}")

        self.optimizer = lars_adam_torch(self.training_mode_params.learning_rate)(
            self.model.parameters()
        ).optimizer
        self.lr_schedule = get_lr_schedule(
            self.training_mode_params,
            self.optimizer,
            len(self.train_dataloader),
            params,
        )

        self._restore_model_weights()
        self._quantize_modules()
        self._set_up_tensorboard_logging()

        self.metrics = get_metrics()
        for metric in self.metrics:
            metric.to(self.device)

        if (
            self.params.dataset.path.validation
            and not self.params.train.perform_validate
        ):
            logger.warning(
                "Validation path is provided but perform_validate is set to false"
            )

    def _quantize_modules(self):
        """Quantize modules if not already quantized"""

        if (
            isinstance(self.model.nss_model, QATNSSModel)
            and not self.model.nss_model.modules_quantized
        ):
            autoencoder_input = self.model.get_model_input_for_tracing(
                next(iter(self.train_dataloader))[0]
            )
            self.model.nss_model.quantize_modules(
                autoencoder_input.shape,
                device=self.device,
            )

    def _restore_model_weights(self):
        """Load weights from a checkpoint file if specified"""

        self.model_save_path = None

        if self.params.train.resume:
            # Path to directory containing all the training runs
            user_checkpoint_save_dir = Path(self.training_mode_params.checkpoints.dir)

            # Grab the most recent ckpt-XX.pt file
            checkpoint_path = latest_checkpoint_path(user_checkpoint_save_dir)
            self.model_save_path = checkpoint_path.parent

            # If model is QAT, make sure it is in a traced state for loading in weights
            self._quantize_modules()

            # Restore model weights and optimizer state
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.starting_epoch = checkpoint["epoch"] + 1

            # Make sure starting epoch is not more than configured epochs for training
            if self.starting_epoch > self.training_mode_params.number_of_epochs:
                raise ValueError(
                    f"Restoring from checkpoint at epoch {self.starting_epoch} but "
                    f"training is configured for only {self.training_mode_params.number_of_epochs}"
                    f" epochs"
                )
            logger.info(
                f"Restoring training from checkpoint "
                f"{self.model_save_path.name} at epoch {self.starting_epoch}"
            )

        elif self.params.train.finetune:
            if self.params.train.pretrained_weights is None:
                raise ValueError(
                    "Config error: Finetuning but no pretrained weights specified"
                )

            # Read the user specified fine tune model path
            finetune_path = Path(self.params.train.pretrained_weights)

            if not finetune_path.exists() or not finetune_path.is_file():
                raise FileNotFoundError(
                    f"Couldn't find {finetune_path} for fine-tuning"
                )

            finetune_weight = torch.load(finetune_path, weights_only=True)
            self.model.load_state_dict(finetune_weight["model_state_dict"])
            logger.info(f"Fine tuning using weights {finetune_path.name}")

        # If a path has not been defined, create a new directory to store checkpoints
        if not self.model_save_path:
            time_stamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            self.model_save_path = Path(
                self.training_mode_params.checkpoints.dir, time_stamp
            )
            create_directory(self.model_save_path)
            logger.info(f"Model checkpoints save dir {self.model_save_path.absolute()}")
            logger.info(
                f"Checkpoint epoch save interval: "
                f"{self.training_mode_params.checkpoints.save_frequency}"
            )

    def _get_values_for_logging(self, values_to_log, metrics=None, name=""):
        """Gather all values and metrics to be logged into a dictionary."""

        tb_values = {}

        for k, v in values_to_log.items():
            tb_values[name + k] = v

        for metric in metrics:
            tb_values[name + str(metric)] = metric.compute()

        return tb_values

    def _set_up_tensorboard_logging(self):
        """Set up TensorBoard logging"""

        if self.params.output.tensorboard_output_dir is not None:
            tensorboard_log_path = Path(
                self.params.output.tensorboard_output_dir / self.model_save_path.name
            )

            # Create TensorBoard logs directory if it doesn't already exist
            create_directory(tensorboard_log_path)

            self.tensorboard = SummaryWriter(log_dir=str(tensorboard_log_path))

            logger.info(
                f"TensorBoard access command:"
                f" $ tensorboard --logdir {str(tensorboard_log_path.parent.absolute())} \n"
            )

        else:
            self.tensorboard = None
            logger.info(
                "TensorBoard logging is disabled (Reason: tensorboard_output_dir=None)"
            )

    def _tensorboard_update(self, values_to_log, step):
        """Update the TensorBoard writer with new values."""

        if self.tensorboard is None:
            return

        for k, v in values_to_log.items():
            self.tensorboard.add_scalar(k, v, step)

        self.tensorboard.flush()

    def train(self, profiler: Optional[torch.profiler.profile] = None):
        """Start training loop"""

        total_epochs = self.training_mode_params.number_of_epochs
        self.model.train()

        for epoch in range(self.starting_epoch, total_epochs + 1):
            self.model.nss_model.reset_history_buffers()

            running_epoch_loss = 0.0
            total_batches = len(self.train_dataloader)
            train_pbar = tqdm(
                enumerate(self.train_dataloader, 0),
                total=total_batches,
                desc=f"\nTrain: Epoch {epoch}/{total_epochs}",
                leave=True,
            )

            for iteration, (inputs_dataset, ground_truth_data) in train_pbar:
                self.optimizer.zero_grad()
                # Move tensors to device.
                inputs_dataset = {
                    key: tensor.to(self.device)
                    for key, tensor in inputs_dataset.items()
                }

                ground_truth_data = ground_truth_data.to(self.device)

                self.model.y_true = ground_truth_data

                inference_out = self.model(inputs_dataset)

                loss, _ = self.criterion(
                    ground_truth_data, inference_out | inputs_dataset
                )

                loss.backward()
                self.optimizer.step()
                self.model.detach_buffers()

                if self.lr_schedule:
                    self.lr_schedule.step()

                # Accumulate the loss
                running_epoch_loss += loss.item()

                # Calculate average loss
                avg_loss = running_epoch_loss / (iteration + 1)

                progress_bar = (
                    f"Train: Epoch {epoch}/{total_epochs}, "
                    f"Running Average Loss: {avg_loss:.4f}, "
                    f"Mini batch Loss: {loss.item():.4f}, "
                )

                for metric in self.metrics:
                    metric.update(inference_out["output"], ground_truth_data)
                    progress_bar += f"{metric}: {metric.compute():.4f}, "

                if profiler:
                    profiler.step()

                train_pbar.set_description(progress_bar)

                tb_values = self._get_values_for_logging(
                    {"Loss": avg_loss}, self.metrics, name="Train/"
                )
                self._tensorboard_update(tb_values, iteration + epoch * total_batches)

            for metric in self.metrics:
                metric.reset()

            save_frequency = int(self.training_mode_params.checkpoints.save_frequency)
            self._save_checkpoint(save_frequency, epoch, total_epochs)

            # Evaluate on the validation set.
            if self.params.train.perform_validate:
                with torch.no_grad():
                    self.validate(epoch)

    def validate(self, epoch):
        """Start validation loop."""
        total_epochs = self.training_mode_params.number_of_epochs
        self.model.nss_model.reset_history_buffers()

        val_dataloader = get_dataloader(
            self.params,
            num_workers=self.params.dataset.num_workers,
            prefetch_factor=self.params.dataset.prefetch_factor,
            loader_mode=DataLoaderMode.VAL,
        )
        val_pbar = tqdm(
            enumerate(val_dataloader, 0),
            total=len(val_dataloader),
            desc=f"Validation: Epoch {epoch}/{total_epochs}",
            leave=True,
        )

        self.model.eval()
        for iteration, (inputs_dataset, ground_truth_data) in val_pbar:
            # Move tensors to device.
            inputs_dataset = {
                key: tensor.to(self.device) for key, tensor in inputs_dataset.items()
            }

            ground_truth_data = ground_truth_data.to(self.device)

            self.model.y_true = ground_truth_data

            inference_out = self.model(inputs_dataset)

            progress_bar = f"Validation: Epoch {epoch}/{total_epochs}, "

            for metric in self.metrics:
                metric.update(inference_out["output"], ground_truth_data)
                progress_bar += f"{metric}: {metric.compute():.4f}, "

            val_pbar.set_description(progress_bar)

            #  Push validation set results to Tensorboard.
            tb_values = self._get_values_for_logging(
                {}, self.metrics, name="Validation/"
            )
            self._tensorboard_update(tb_values, iteration + epoch * len(val_dataloader))

    def _save_checkpoint(self, save_frequency, current_epoch, total_epochs):
        """Save checkpoint if end or configured save frequency epoch"""
        is_save_frequency_epoch = (current_epoch % save_frequency) == 0
        is_final_epoch = current_epoch == total_epochs

        # Ensure the directory to save checkpoints to exists, if not then create the directory
        create_directory(self.model_save_path)

        if is_save_frequency_epoch or is_final_epoch:
            save_path = Path(self.model_save_path, f"ckpt-{current_epoch}.pt")
            torch.save(
                {
                    "epoch": current_epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                save_path.absolute(),
            )

            self.latest_model_save_path = save_path

            logger.info(f"Saved to {save_path}")


def get_lr_schedule(
    training_mode: TrainingConfig,
    optimizer: torch.optim.Optimizer,
    train_data_size: int,
    params: ConfigModel,
):
    """Return the Learning Rate schedule specified in params.

    Args:
        training_mode: Set params from "train.fp32" or "train.qat" JSON config section
        optimizer: Training optimizer being used.
        train_data_size: Size of the training set.
        params: Dictionary of configuration params.

    Returns:
        Requested LR schedule or None if static.
    """

    if (
        params.optimizer.learning_rate_scheduler
        == LearningRateScheduler.COSINE_ANNEALING
    ):
        batch_size = params.train.batch_size
        dataset_length = train_data_size // batch_size
        steps_per_epoch = max(1, dataset_length)
        total_epochs = training_mode.number_of_epochs

        warmup_pct = training_mode.cosine_annealing_scheduler_config.warmup_percentage
        min_lr = training_mode.cosine_annealing_scheduler_config.min_lr

        logger.info(
            f"Using CosineAnnealing scheduler: "
            f"{warmup_pct*100:.1f}% warmup, min_lr={min_lr}, "
            f"Scheduler steps per epoch: {steps_per_epoch}"
        )

        lr_schedule = CosineAnnealingWithWarmupLR(
            optimizer=optimizer,
            steps_per_epoch=steps_per_epoch,
            total_epochs=total_epochs,
            warmup_percentage=warmup_pct,
            min_lr=min_lr,
        )

    elif params.optimizer.learning_rate_scheduler == LearningRateScheduler.EXPONENTIAL:
        decay_rate = DEFAULT_EXP_DECAY_RATE
        decay_step = DEFAULT_EXP_DECAY_STEP
        step_size = max(
            1,
            int((training_mode.number_of_epochs * train_data_size) / decay_step),
        )

        logger.info(f"Exponential optimizer learning rate step size: {step_size}")

        lr_schedule = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=step_size, gamma=decay_rate
        )
    elif params.optimizer.learning_rate_scheduler == LearningRateScheduler.STATIC:
        # Don't need a schedule for static learning rate.
        lr_schedule = None
    else:
        raise ValueError(
            f"{params.optimizer.learning_rate_scheduler} is not recognised."
        )

    return lr_schedule
