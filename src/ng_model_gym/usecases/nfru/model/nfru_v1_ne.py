# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from ng_model_gym.core.model.layers.conv_block import ConvBlock
from ng_model_gym.usecases.nfru.model.constants import (
    _DEFAULT_SCALE_FACTOR,
    _KERNEL_SIZE_1,
    _KERNEL_SIZE_3,
    _KERNEL_SIZE_5,
    _KERNEL_SIZE_7,
    _NFRU_AUTOENCODER_BRANCH_CHANNELS,
    _NFRU_AUTOENCODER_CONCAT_CHANNELS,
    _NFRU_AUTOENCODER_HIDDEN_CHANNELS,
    _NFRU_AUTOENCODER_INPUT_CHANNELS,
    _NFRU_AUTOENCODER_OUTPUT_CHANNELS,
    _NFRU_BATCH_NORM_MOMENTUM,
    _PADDING_3X3,
    _PADDING_5X5,
    _PADDING_7X7,
    _PADDING_NONE,
    _STRIDE_1X1,
    _STRIDE_2X2,
)


class NFRUAutoEncoder(nn.Module):
    """Lightweight autoencoder used within the NFRU v1 architecture."""

    def __init__(
        self,
        in_ch=_NFRU_AUTOENCODER_INPUT_CHANNELS,
        batch_norm=True,
        activation="relu",
        momentum=_NFRU_BATCH_NORM_MOMENTUM,
    ):
        super().__init__()

        self.conv1 = ConvBlock(
            in_ch,
            _NFRU_AUTOENCODER_HIDDEN_CHANNELS,
            kernel_size=_KERNEL_SIZE_3,
            padding=_PADDING_3X3,
            batch_norm=batch_norm,
            activation=activation,
            momentum=momentum,
        )

        self.conv2 = ConvBlock(
            _NFRU_AUTOENCODER_HIDDEN_CHANNELS,
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            kernel_size=_KERNEL_SIZE_5,
            stride=_STRIDE_1X1,
            padding=_PADDING_5X5,
            batch_norm=batch_norm,
            activation=activation,
            momentum=momentum,
        )

        self.skip1_conv = ConvBlock(
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            kernel_size=_KERNEL_SIZE_3,
            batch_norm=batch_norm,
            activation=activation,
            momentum=momentum,
        )

        self.conv3 = ConvBlock(
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            kernel_size=_KERNEL_SIZE_3,
            batch_norm=batch_norm,
            activation=activation,
            momentum=momentum,
        )

        self.conv5 = ConvBlock(
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            kernel_size=_KERNEL_SIZE_5,
            stride=_STRIDE_2X2,
            padding=_PADDING_5X5,
            batch_norm=batch_norm,
            activation=activation,
            momentum=momentum,
        )

        self.conv5a = ConvBlock(
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            kernel_size=_KERNEL_SIZE_1,
            padding=_PADDING_NONE,
            batch_norm=batch_norm,
            activation=activation,
            momentum=momentum,
        )

        self.conv5b = ConvBlock(
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            kernel_size=_KERNEL_SIZE_3,
            batch_norm=batch_norm,
            activation=activation,
            momentum=momentum,
        )

        self.conv5c = ConvBlock(
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            kernel_size=_KERNEL_SIZE_7,
            padding=_PADDING_7X7,
            batch_norm=batch_norm,
            activation=activation,
            momentum=momentum,
        )
        self.conv5c_1 = ConvBlock(
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            kernel_size=_KERNEL_SIZE_7,
            padding=_PADDING_7X7,
            batch_norm=batch_norm,
            activation=activation,
            momentum=momentum,
        )

        self.conv5d = ConvBlock(
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            kernel_size=_KERNEL_SIZE_7,
            padding=_PADDING_7X7,
            batch_norm=batch_norm,
            activation=activation,
            momentum=momentum,
        )
        self.conv5d_1 = ConvBlock(
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            kernel_size=_KERNEL_SIZE_7,
            padding=_PADDING_7X7,
            batch_norm=batch_norm,
            activation=activation,
            momentum=momentum,
        )
        self.conv5d_2 = ConvBlock(
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            kernel_size=_KERNEL_SIZE_7,
            padding=_PADDING_7X7,
            batch_norm=batch_norm,
            activation=activation,
            momentum=momentum,
        )

        self.conv5e = ConvBlock(
            _NFRU_AUTOENCODER_CONCAT_CHANNELS,
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            kernel_size=_KERNEL_SIZE_1,
            padding=_PADDING_NONE,
            batch_norm=batch_norm,
            activation=activation,
            momentum=momentum,
        )

        # Note: nearest-exact (as used in custom_layers.UpSampling2D) does not map
        # cleanly to TOSA, so use the standard nearest mode here.
        self.upsample1 = nn.Upsample(mode="nearest", scale_factor=_DEFAULT_SCALE_FACTOR)
        self.conv6 = ConvBlock(
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            kernel_size=_KERNEL_SIZE_3,
            padding=_PADDING_3X3,
            batch_norm=batch_norm,
            activation=activation,
            momentum=momentum,
        )

        self.conv7 = ConvBlock(
            _NFRU_AUTOENCODER_HIDDEN_CHANNELS,
            _NFRU_AUTOENCODER_BRANCH_CHANNELS,
            kernel_size=_KERNEL_SIZE_3,
            batch_norm=batch_norm,
            activation=activation,
            momentum=momentum,
        )

        self.output_conv_mv = nn.Conv2d(
            in_channels=_NFRU_AUTOENCODER_BRANCH_CHANNELS,
            out_channels=_NFRU_AUTOENCODER_OUTPUT_CHANNELS,
            kernel_size=_KERNEL_SIZE_5,
            padding=_PADDING_5X5,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder/decoder stack to produce motion parameters."""
        # E-0 - 540
        x = self.conv1(x)
        x = self.conv2(x)

        # E-2 - 270
        x = self.conv3(x)
        skip1 = self.skip1_conv(x)
        bottneck_input = self.conv5(x)

        # B - 135
        xa = self.conv5a(bottneck_input)

        xb = self.conv5b(bottneck_input)

        xc = self.conv5c(bottneck_input)
        xc = self.conv5c_1(xc)

        xd = self.conv5d(bottneck_input)
        xd = self.conv5d_1(xd)
        xd = self.conv5d_2(xd)

        x = torch.concatenate([xa, xb, xc, xd], dim=1)
        x = self.conv5e(x)

        # D-2
        x = self.upsample1(x)
        x = self.conv6(x)
        x = torch.concatenate([x, skip1], dim=1)

        # D-1
        x = self.conv7(x)

        # D-0 / Output(s)
        learnt_params_mv = self.output_conv_mv(x)

        return learnt_params_mv
