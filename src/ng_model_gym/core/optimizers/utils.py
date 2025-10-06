# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
from functools import partial

import torch


class OptimizerWrapper:
    """Optimizer class"""

    def __init__(self, optimizer, lr_scheduler, **optimizer_args):
        if isinstance(lr_scheduler, float):
            self.lr_scheduler = None
            self.optimizer = partial(optimizer, lr=lr_scheduler, **optimizer_args)
            self.lr = lr_scheduler
        else:
            self.optimizer = partial(optimizer, lr=lr_scheduler.LR, **optimizer_args)
            self.lr_scheduler = lr_scheduler

    @torch.compiler.disable
    def step(self, *args, **kwargs):
        """Step function"""
        self.optimizer.step(*args, **kwargs)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.lr = self.lr_scheduler.get_lr()

    def __call__(self, *args, **kwargs):
        self.optimizer = self.optimizer(*args, **kwargs)
        if self.lr_scheduler is not None:
            self.lr_scheduler.initialize(self.optimizer)
            self.lr = self.lr_scheduler.get_lr()
        return self

    def zero_grad(self):
        """Zero grad"""
        self.optimizer.zero_grad()
