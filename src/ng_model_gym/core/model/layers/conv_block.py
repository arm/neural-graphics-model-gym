# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Tuple

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Convolutional block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: Tuple[int],
        bias: Optional[bool] = True,
        stride: Optional[Tuple[int]] = (1, 1),
        activation: Optional[str] = "",
        batch_norm: Optional[bool] = True,
        momentum: Optional[float] = 0.1,
    ):
        super().__init__()
        _activation_table = {
            "leakyrelu": nn.LeakyReLU,
            "relu": nn.ReLU,
            "relu6": nn.ReLU6,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "": nn.Identity,
        }

        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            stride=stride,
            padding=padding,
        )
        self.bn = (
            nn.BatchNorm2d(num_features=out_channels, momentum=momentum)
            if batch_norm
            else nn.Identity()
        )
        self.act = _activation_table[str(activation).lower()]()

    # pylint: disable=unused-argument
    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass of convolutional block"""
        x = self.conv2d(inputs)
        x = self.bn(x)
        return self.act(x)

    # pylint: enable=unused-argument
