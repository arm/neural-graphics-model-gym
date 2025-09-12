# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

"""
Code taken from:
https://github.com/kakaobrain/torchlars

Modified to implement compute_adaptive_lr in PyTorch instead of CUDA

"""
from functools import partial

import torch
from torch.optim import Adam
from torch.optim.optimizer import Optimizer


def compute_adaptive_lr(
    param_norm: torch.Tensor,
    grad_norm: torch.Tensor,
    weight_decay: torch.Tensor,
    eps: torch.Tensor,
    trust_coef: torch.Tensor,
):
    """Compute adaptative LR"""
    divisor = grad_norm + weight_decay * param_norm + eps
    return torch.where(
        torch.logical_and(param_norm > 0, grad_norm > 0),
        param_norm / (divisor * trust_coef),
        torch.ones_like(param_norm),
    )


class LARS(Optimizer):
    """Implements 'LARS (Layer-wise Adaptive Rate Scaling)'__ as Optimizer a
    :class:`~torch.optim.Optimizer` wrapper.

    __ : https://arxiv.org/abs/1708.03888

    Wraps an arbitrary optimizer like :class:`torch.optim.SGD` to use LARS. If
    you want to the same performance obtained with small-batch training when
    you use large-batch training, LARS will be helpful::

    Args:
        optimizer (Optimizer):
            optimizer to wrap
        eps (float, optional):
            epsilon to help with numerical stability while calculating the
            adaptive learning rate
        trust_coef (float, optional):
            trust coefficient for calculating the adaptive learning rate

    Example::
        base_optimizer = optim.SGD(model.parameters(), lr=0.1)
        optimizer = LARS(optimizer=base_optimizer)

        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()

        optimizer.step()

    """

    # pylint: disable-next=super-init-not-called
    def __init__(self, optimizer, eps=1e-8, trust_coef=0.001):
        if eps < 0.0:
            raise ValueError(f"invalid epsilon value: {eps:.8f}")
        if trust_coef < 0.0:
            raise ValueError(f"invalid epsilon value: {trust_coef:.3f}")

        self.optim = optimizer
        self.eps = eps
        self.trust_coef = trust_coef
        self.adaptive_lr = torch.ones([])

    def __getstate__(self):
        lars_dict = {}
        lars_dict["eps"] = self.eps
        lars_dict["trust_coef"] = self.trust_coef
        lars_dict["adaptive_lr"] = self.adaptive_lr
        return (self.optim, lars_dict)

    def __setstate__(self, state):
        self.optim, lars_dict = state

        self.eps = lars_dict["eps"]
        self.trust_coef = lars_dict["trust_coef"]
        self.adaptive_lr = lars_dict["adaptive_lr"]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.optim!r})"

    @property
    def param_groups(self):
        """params group"""
        return self.optim.param_groups

    def state_dict(self):
        """Sate dict"""
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        """Zero grad"""
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        """Add param group"""
        self.optim.add_param_group(param_group)

    def hide_weight_decays(self):
        """Returns a context manager that temporarily zeroes out weight decay"""

        class _WeightDecayHider:
            def __init__(self, optimizer):
                self.optimizer = optimizer
                self.weight_decays = []

            def __enter__(self):
                self.weight_decays = []
                for group in self.optimizer.param_groups:
                    if "weight_decay" in group:
                        self.weight_decays.append(group["weight_decay"])
                        group["weight_decay"] = 0
                    else:
                        self.weight_decays.append(None)
                return self.weight_decays

            def __exit__(self, exc_type, exc_val, exc_tb):
                for group, wd in zip(self.optimizer.param_groups, self.weight_decays):
                    if wd is not None:
                        group["weight_decay"] = wd

        return _WeightDecayHider(self.optim)

    def apply_adaptive_lrs(self, weight_decays):
        """Apply adaptative LRS"""
        with torch.no_grad():
            for group, weight_decay in zip(self.optim.param_groups, weight_decays):
                if weight_decay is None:
                    weight_decay = 0.0
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    param_norm = p.norm()
                    grad_norm = p.grad.norm()

                    # The optimizer class has no method to change `dtype` of
                    # its inner tensors (like `adaptive_lr`) and to select to
                    # use CPU or GPU with Tensor. LARS's interface follows the
                    # optimizer class's interface, so LARS cannot change
                    # `dtype` of inner tensors explicitly also. In that
                    # context, we have constructed LARS so it can modify its member
                    # variable's spec implicitly by comparing with given spec
                    # by the original optimizer's element.
                    param_norm_spec = (param_norm.is_cuda, param_norm.type())
                    adaptive_lr_spec = (
                        self.adaptive_lr.is_cuda,
                        self.adaptive_lr.type(),
                    )

                    if param_norm_spec != adaptive_lr_spec:
                        self.adaptive_lr = torch.ones_like(param_norm)

                    # calculate adaptive lr & weight decay
                    adaptive_lr = compute_adaptive_lr(
                        param_norm,
                        grad_norm,
                        weight_decay,
                        self.eps,
                        self.trust_coef,
                    )
                    p.grad.add_(p.data, alpha=weight_decay)
                    p.grad.mul_(adaptive_lr)

    @torch.compiler.disable
    def step(self, *args, **kwargs):
        """Step function"""
        with self.hide_weight_decays() as weight_decays:
            self.apply_adaptive_lrs(weight_decays)
            return self.optim.step(*args, **kwargs)


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


class LARSWrapper(OptimizerWrapper):
    """Wrapper class for LARS"""

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        self.optimizer = LARS(optimizer=self.optimizer)
        return self


def lars_adam_torch(learning_rate, **optimizer_args):
    """The LARS Adam PyTorch function"""
    return LARSWrapper(Adam, learning_rate, **optimizer_args)
