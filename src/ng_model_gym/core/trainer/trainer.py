# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from ng_model_gym.core.data.dataloader import get_dataloader
from ng_model_gym.core.data.utils import DataLoaderMode, move_to_device
from ng_model_gym.core.evaluator.metrics import get_metrics
from ng_model_gym.core.loss.losses import LossV1
from ng_model_gym.core.model.base_ng_model import BaseNGModel
from ng_model_gym.core.model.model_factory import create_model
from ng_model_gym.core.model.model_tracer import model_tracer
from ng_model_gym.core.optimizers.adam_w import adam_w_torch
from ng_model_gym.core.optimizers.lars_adam import lars_adam_torch
from ng_model_gym.core.schedulers.lr_scheduler import CosineAnnealingWithWarmupLR
from ng_model_gym.core.utils.checkpoint_utils import (
    latest_checkpoint_in_dir,
    remap_feedback_model_state_dict,
)
from ng_model_gym.core.utils.config_model import ConfigModel, TrainingConfig
from ng_model_gym.core.utils.general_utils import create_directory
from ng_model_gym.core.utils.types import (
    LearningRateScheduler,
    LossFn,
    ModelType,
    OptimizerType,
    TrainEvalMode,
)

logger = logging.getLogger(__name__)


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

        if self.params.train.perform_validate:
            self.val_dataloader = get_dataloader(
                self.params,
                num_workers=self.params.dataset.num_workers,
                prefetch_factor=self.params.dataset.prefetch_factor,
                loader_mode=DataLoaderMode.VAL,
            )

        self.criterion = get_loss_fn(self.params, self.device)
        self.starting_epoch = 1
        self.model: nn.Module = create_model(self.params, self.device).to(self.device)

        if not isinstance(self.model, BaseNGModel):
            raise ValueError("Model must be an instance of BaseNGModel")

        logger.info(f"Model architecture: {self.model}")
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Total number of parameters: {total_params:,}")

        self.optimizer = get_optimizer_type(
            self.training_mode_params, self.model.parameters()
        )

        self.lr_schedule = get_lr_schedule(
            self.training_mode_params, self.optimizer, len(self.train_dataloader)
        )

        self._restore_model_weights()
        self._quantize_modules()
        self._set_up_tensorboard_logging()

        self.metrics = get_metrics(self.params)
        for metric in self.metrics:
            metric.to(self.device)

        self.avg_val_loss = float("inf")

        if (
            self.params.dataset.path.validation
            and not self.params.train.perform_validate
        ):
            logger.warning(
                "Validation path is provided but perform_validate is set to false"
            )

    def _quantize_modules(self):
        """Quantize modules if not already quantized"""

        ng_model = self.model

        if ng_model.is_qat_model and not ng_model.is_network_quantized:
            input_data = next(iter(self.train_dataloader))[0]

            # Tracing must be the exact same model as the trainer forward pass model
            forward_input_data = model_tracer(self.model, input_data)

            ng_model.quantize_modules(forward_input_data)

    def _restore_model_weights(self):
        """Load weights from a checkpoint file if specified"""

        self.model_save_path = None

        resume_path = self.params.train.resume

        if resume_path:
            checkpoint_path = None

            if resume_path.is_dir():
                checkpoint_path = latest_checkpoint_in_dir(resume_path)

            elif resume_path.is_file() and resume_path.suffix.lower() == ModelType.PT:
                checkpoint_path = resume_path

            if checkpoint_path is None:
                raise ValueError(
                    f"Resume must be an existing directory or a .pt file, got: {resume_path}"
                )

            self.model_save_path = checkpoint_path.parent

            # If model is QAT, make sure it is in a traced state for loading in weights
            self._quantize_modules()

            # Restore model weights and optimizer state
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            model_state = remap_feedback_model_state_dict(
                checkpoint["model_state_dict"]
            )
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.lr_schedule.load_state_dict(checkpoint["lr_scheduler_state_dict"])
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
            finetune_path = Path(self.params.train.finetune)

            if not finetune_path.exists() or not finetune_path.is_file():
                raise FileNotFoundError(
                    f"Couldn't find local model at path '{finetune_path}' for fine-tuning"
                )
            if finetune_path.suffix.lower() != ModelType.PT:
                raise ValueError(
                    f"Fine-tune weights must be a .pt file, got {finetune_path.name}"
                )

            finetune_weight = torch.load(finetune_path, weights_only=True)
            model_state = remap_feedback_model_state_dict(
                finetune_weight["model_state_dict"]
            )
            self.model.load_state_dict(model_state)
            logger.info(f"Fine tuning using weights {finetune_path.name}")

        # If a path has not been defined, create a new directory to store checkpoints
        if not self.model_save_path:
            time_stamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            self.model_save_path = Path(
                self.training_mode_params.checkpoints.dir, time_stamp
            )
            create_directory(self.model_save_path)
            logger.info(f"Model checkpoints save dir {self.model_save_path.absolute()}")
            logger.info("Saving checkpoints after each epoch")

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

    def _should_validate(self, epoch):
        if self.params.train.perform_validate and (
            (
                isinstance(self.params.train.validate_frequency, int)
                and epoch % self.params.train.validate_frequency == 0
            )
            or (
                isinstance(self.params.train.validate_frequency, (list))
                and epoch in self.params.train.validate_frequency
            )
        ):
            return True
        return False

    def train(self, profiler: Optional[torch.profiler.profile] = None):
        """Start training loop"""

        total_epochs = self.training_mode_params.number_of_epochs

        for epoch in range(self.starting_epoch, total_epochs + 1):
            self.model.train()
            self.model.on_train_epoch_start()

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

                inputs_dataset = move_to_device(inputs_dataset, self.device)
                ground_truth_data = move_to_device(ground_truth_data, self.device)

                self.model.y_true = ground_truth_data

                inference_out = self.model(inputs_dataset)

                if not isinstance(inference_out, dict) or "output" not in inference_out:
                    raise TypeError(
                        "Forward pass must return a dictionary containing 'output' key"
                    )

                loss = self.criterion(ground_truth_data, inference_out)

                loss.backward()
                self.optimizer.step()

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
                self._tensorboard_update(
                    tb_values, iteration + (epoch - 1) * total_batches
                )
                self.model.on_train_batch_end()

            for metric in self.metrics:
                metric.reset()

            # Evaluate on the validation set, depending on the epoch number
            if self._should_validate(epoch):
                with torch.no_grad():
                    self.validate(epoch)
            # Save after every epoch
            self._save_checkpoint(epoch)

            self.model.on_train_epoch_end()

        self.model.on_train_end()

    def validate(self, epoch):
        """Start validation loop."""
        total_epochs = self.training_mode_params.number_of_epochs

        val_pbar = tqdm(
            enumerate(self.val_dataloader, 0),
            total=len(self.val_dataloader),
            desc=f"Validation: Epoch {epoch}/{total_epochs}",
            leave=True,
        )

        self.model.eval()
        self.model.on_validation_start()
        running_val_loss = 0.0

        for iteration, (inputs_dataset, ground_truth_data) in val_pbar:
            # Move tensors to device.
            inputs_dataset = move_to_device(inputs_dataset, self.device)
            ground_truth_data = move_to_device(ground_truth_data, self.device)

            self.model.y_true = ground_truth_data

            inference_out = self.model(inputs_dataset)

            loss = self.criterion(ground_truth_data, inference_out)

            # Accumulate the loss
            running_val_loss += loss.item()

            # Calculate average loss
            self.avg_val_loss = running_val_loss / (iteration + 1)

            progress_bar = (
                f"Validation: Epoch {epoch}/{total_epochs}, "
                f"Avg. Loss: {self.avg_val_loss:.4f}, "
            )

            for metric in self.metrics:
                metric.update(inference_out["output"], ground_truth_data)
                progress_bar += f"{metric}: {metric.compute():.4f}, "

            val_pbar.set_description(progress_bar)

        #  Push validation set results to Tensorboard.
        tb_values = self._get_values_for_logging(
            {"Loss": self.avg_val_loss}, self.metrics, name="Validation/"
        )
        self._tensorboard_update(tb_values, epoch)

        for metric in self.metrics:
            metric.reset()

        self.model.on_validation_end()

    def _save_checkpoint(self, current_epoch):
        """Save checkpoint if end or configured save frequency epoch"""
        # Ensure the directory to save checkpoints to exists, if not then create the directory
        create_directory(self.model_save_path)

        save_path = Path(self.model_save_path, f"ckpt-{current_epoch}.pt")
        torch.save(
            {
                "epoch": current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_schedule.state_dict(),
            },
            save_path.absolute(),
        )

        self.latest_model_save_path = save_path
        logger.info(f"Saved to {save_path}")

        # Track and update best checkpoint based on validation metric
        if self.params.train.perform_validate:
            best_ckpt_path = Path(self.model_save_path, "best-validated-ckpt.pt")
            best_meta_path = Path(self.model_save_path, "best-validated-ckpt.meta.json")
            current_val = self.avg_val_loss

            best_val = float("inf")
            if best_meta_path.exists():
                with open(best_meta_path, "r", encoding="utf-8") as f:
                    best_val = json.load(f).get("val_loss", best_val)

            # Use lowest validation loss to determine best ckpt
            if current_val < best_val:
                shutil.copyfile(save_path, best_ckpt_path)
                with open(best_meta_path, "w", encoding="utf-8") as f:
                    json.dump({"epoch": current_epoch, "val_loss": current_val}, f)

                self.best_model_save_path = best_ckpt_path
                logger.info(
                    f"New best checkpoint from epoch {current_epoch} copied to {best_ckpt_path}"
                )
            self.avg_val_loss = float("inf")  # Reset after saving


def get_lr_schedule(
    training_mode: TrainingConfig,
    optimizer: torch.optim.Optimizer,
    dataloader_size: int,
):
    """Return the Learning Rate schedule specified in params.

    Args:
        training_mode: Set params from "train.fp32" or "train.qat" JSON config section
        optimizer: Training optimizer being used.
        dataloader_size: Size of the dataloader.

    Returns:
        Requested LR schedule or None if static.
    """

    if training_mode.lr_scheduler.type == LearningRateScheduler.COSINE_ANNEALING:
        total_epochs = training_mode.number_of_epochs

        warmup_pct = training_mode.lr_scheduler.warmup_percentage
        min_lr = training_mode.lr_scheduler.min_lr

        logger.info(
            f"Using CosineAnnealing scheduler: "
            f"{warmup_pct*100:.1f}% warmup, min_lr={min_lr}, "
            f"Scheduler steps per epoch: {dataloader_size}"
        )

        lr_schedule = CosineAnnealingWithWarmupLR(
            optimizer=optimizer,
            steps_per_epoch=dataloader_size,
            total_epochs=total_epochs,
            warmup_percentage=warmup_pct,
            min_lr=min_lr,
        )

    elif training_mode.lr_scheduler.type == LearningRateScheduler.EXPONENTIAL_STEP:
        decay_rate = training_mode.lr_scheduler.decay_rate
        decay_factor = training_mode.lr_scheduler.decay_factor
        step_size = max(
            1,
            int((training_mode.number_of_epochs * dataloader_size) / decay_factor),
        )

        logger.info(f"Exponential Step optimizer learning rate step size: {step_size}")

        lr_schedule = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=step_size, gamma=decay_rate
        )
    elif training_mode.lr_scheduler.type == LearningRateScheduler.STATIC:
        # Don't need a schedule for static learning rate.
        lr_schedule = None
    else:
        raise ValueError(f"{training_mode.lr_scheduler.type} is not recognised.")

    return lr_schedule


def get_loss_fn(params: ConfigModel, device: torch.device):
    """
    Return loss function configured by params.train.loss_fn.
    Expected values (string) are set in the LossFn enum.
    """
    loss_name = params.train.loss_fn

    if loss_name == LossFn.LOSS_V1:
        return LossV1(params.dataset.recurrent_samples, device)

    # Defensive (all enums should be handled above)
    raise ValueError(f"No implementation for loss function {loss_name}")


def get_optimizer_type(
    training_mode_params: TrainingConfig,
    model_params: Iterator[Parameter],
):
    """
    Return optimizer type from the config.
    Expected values (string) are set in the OptimizerType enum.
    """
    optimizer_type = training_mode_params.optimizer.optimizer_type

    if optimizer_type == OptimizerType.LARS_ADAM:
        return lars_adam_torch(training_mode_params.optimizer.learning_rate)(
            model_params
        ).optimizer
    if optimizer_type == OptimizerType.ADAM_W:
        return adam_w_torch(training_mode_params.optimizer.learning_rate)(
            model_params
        ).optimizer

    # Defensive (all enums should be handled above)
    raise ValueError(f"Unsupported optimizer type {optimizer_type}")
