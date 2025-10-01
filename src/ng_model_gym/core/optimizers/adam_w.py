# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

from torch.optim import AdamW

from ng_model_gym.core.optimizers.utils import OptimizerWrapper


def adam_w_torch(learning_rate, **optimizer_args):
    """AdamW optimizer from PyTorch"""
    return OptimizerWrapper(AdamW, lr_scheduler=learning_rate, **optimizer_args)
