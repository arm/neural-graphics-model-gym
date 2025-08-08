# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import torch

from ng_model_gym.utils.types import HistoryBufferResetFunction


class HistoryBuffer:
    """Maintains a persistent History Buffer
    This class holds a temporal history buffer which tracks the state of current
    temporal sequence, via `seq` tensor, when the sequence changes the buffer is reset via
    `reset_func`, based on an input tensor, which is user specified.
    """

    def __init__(
        self,
        name,
        reset_key,
        reset_func=HistoryBufferResetFunction.IDENTITY,
        update_key=None,
        channel_dim=None,
        augment_func=None,
        scale=2,
    ):
        """Initialize HistoryBuffer object

        Args:
            name: Name of History Buffer
            reset_key: The tensor used to reset the buffer, or passed as reference to `reset_func`,
                       when performing a custom reset function.
            reset_func: Function to use to reset the buffer.
                        Defaults to HistoryBufferResetFunction.IDENTITY which means
                        buffer will reset to identity of `reset_key`. Can be one of:
                        IDENTITY, ZEROS, ONES, RESET_LR, RESET_HR or a callable.
            update_key: The key that the buffer uses to update state,
                        Defaults to `name` if not provided.
            channel_dim: Only used when `reset_func` is ONES, ZEROS, RESET_LR or RESET_HR.
                        Allows to reset to a zeros or ones tensor
                        with same resolution as `reset_key`, but different number of channels.
            augment_func: Optional Callable for applying augmentations to history buffer.
            scale: Option value passed when used with reset_lr or reset_hr
        """

        def _init_shape(x, channel_dim):
            sh = x.shape
            if channel_dim is not None:
                return (sh[0], channel_dim, sh[2], sh[3])
            return sh

        def _reset_lr(x, channel_dim, scale):
            sh = x.shape
            return torch.zeros(
                (sh[0], channel_dim, sh[2] // scale, sh[3] // scale), device=x.device
            )

        def _reset_hr(x, channel_dim, scale):
            sh = x.shape
            return torch.zeros(
                (sh[0], channel_dim, sh[2] * scale, sh[3] * scale), device=x.device
            )

        self.name = name
        self.reset_key = reset_key
        self.update_key = update_key if update_key is not None else name
        self.initialised = False
        self.variable_assigned = False
        self.augment_func = augment_func
        if reset_func == HistoryBufferResetFunction.IDENTITY:
            self.reset_func = lambda x: x
        elif reset_func == HistoryBufferResetFunction.ZEROS:
            self.reset_func = lambda x: torch.zeros(
                _init_shape(x, channel_dim), device=x.device
            )
        elif reset_func == HistoryBufferResetFunction.ONES:
            self.reset_func = lambda x: torch.ones(
                _init_shape(x, channel_dim), device=x.device
            )
        elif reset_func == HistoryBufferResetFunction.RESET_LR:
            self.reset_func = lambda x: _reset_lr(x, channel_dim, scale)
        elif reset_func == HistoryBufferResetFunction.RESET_HR:
            self.reset_func = lambda x: _reset_hr(x, channel_dim, scale)
        elif callable(reset_func):
            self.reset_func = reset_func
        else:
            raise TypeError(f"Unrecognised `reset_func` type: {reset_func}")

    def set(self, state_t, seq):
        """Initial set"""
        self.state_tm1 = self.reset_func(state_t)
        self.initialised = True
        self.seq_tm1 = seq
        return self.state_tm1

    def get(self, state_t, seq):
        """Get based on check of sequence, if not same updates states"""
        self.state_tm1 = torch.where(
            seq == self.seq_tm1, self.state_tm1, self.reset_func(state_t)
        )
        self.seq_tm1 = torch.where(seq == self.seq_tm1, self.seq_tm1, seq)
        return self.state_tm1

    def update(self, state_t):
        """Updates buffer state to `state_t`"""
        self.state_tm1 = state_t

    def detach(self):
        """Detaches state_tm1 and seq_tm1"""
        self.state_tm1 = torch.detach(self.state_tm1)
        self.seq_tm1 = torch.detach(self.seq_tm1)
