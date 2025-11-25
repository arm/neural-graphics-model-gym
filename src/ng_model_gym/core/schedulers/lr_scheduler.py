# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import math
from typing import List

from torch import optim
from torch.optim.lr_scheduler import _warn_get_lr_called_within_step


class CosineAnnealingWithWarmupLR(optim.lr_scheduler.LRScheduler):
    """Cosine annealing learning rate scheduler"""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        steps_per_epoch: int,
        total_epochs: int,
        warmup_percentage: float = 0.1,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        if warmup_percentage > 1.0:
            raise ValueError(
                "Cosine annealing scheduler warmup percentage "
                f"must be less than 100% of training, got: {warmup_percentage*100}%"
            )
        self.warmup_percentage = min(warmup_percentage, 1.0)
        self.total_steps = total_epochs * steps_per_epoch
        self.warmup_steps = int(self.total_steps * self.warmup_percentage)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate based on step count as a percentage of total steps."""
        _warn_get_lr_called_within_step(self)

        step = max(0, self.last_epoch)

        if step < self.warmup_steps:
            # Linear warmup phase
            return [
                self.min_lr + (base_lr - self.min_lr) * (step / self.warmup_steps)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        # Cosine annealing phase
        cosine_progress = (step - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )
        return [
            self.min_lr
            + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * cosine_progress))
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        ]
