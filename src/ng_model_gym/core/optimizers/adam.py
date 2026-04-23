# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import logging

from torch.optim import Adam

from ng_model_gym.core.optimizers.optimizer_wrapper import OptimizerWrapper

logger = logging.getLogger(__name__)
DEFAULT_ADAM_EPS = 1e-7


def adam_torch(learning_rate, **optimizer_args):
    """Adam optimizer from PyTorch"""
    eps_was_defaulted = "eps" not in optimizer_args
    optimizer_args.setdefault("eps", DEFAULT_ADAM_EPS)
    source = "default" if eps_was_defaulted else "config"
    logger.info(f"Using Adam epsilon (eps)={optimizer_args['eps']} (source={source})")
    return OptimizerWrapper(Adam, lr_scheduler=learning_rate, **optimizer_args)
