# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Optional, Tuple, Union

import torch
from torch import nn

from ng_model_gym.core.model.layers.conv_block import ConvBlock

logger = logging.getLogger(__name__)


def split_to_texture_size(n: int) -> list[int]:
    """Takes int and divides it into sizes that are relevant to alias as textures
    e.g., 9 -> [4, 4, 1]
    """
    max_texture_sz = 4  # RGBA
    chunks = []
    full_chunks = n // max_texture_sz
    remainder = n % max_texture_sz

    chunks.extend([max_texture_sz] * full_chunks)

    if remainder == 3:  # RGB isn't well supported
        chunks.extend([2, 1])
    elif remainder > 0:
        chunks.append(remainder)

    return chunks


class AutoEncoderV1(nn.Module):
    """Neural Super Sampling v1 backbone"""

    def __init__(
        self,
        in_channels: Optional[int] = 12,
        feedback_ch: Optional[int] = 4,
        temporal_ch: Optional[int] = 2,
        batch_norm: Optional[bool] = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.feedback_ch = feedback_ch
        self.temporal_ch = temporal_ch
        self.batch_norm = batch_norm

        # Non-Trainable layers
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

        # E-0
        self.conv2d_0 = ConvBlock(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            padding=(1, 1),
            stride=(2, 2),
            batch_norm=False,
            activation="relu",
        )
        self.conv2d_1 = ConvBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=(1, 1),
            batch_norm=self.batch_norm,
            activation="relu",
        )

        # E-1
        self.conv2d_2 = ConvBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=(1, 1),
            stride=(2, 2),
            activation="relu",
            batch_norm=self.batch_norm,
        )
        self.conv2d_3 = ConvBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=(1, 1),
            batch_norm=self.batch_norm,
            activation="relu",
        )

        # E-2
        self.conv2d_4 = ConvBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=(1, 1),
            stride=(2, 2),
            batch_norm=self.batch_norm,
            activation="relu",
        )
        self.conv2d_5 = ConvBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=(1, 1),
            batch_norm=self.batch_norm,
            activation="relu",
        )

        # B
        self.conv2d_6 = ConvBlock(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            padding=(1, 1),
            batch_norm=self.batch_norm,
            activation="relu",
        )

        # D-2
        self.conv2d_7 = ConvBlock(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            padding=(1, 1),
            batch_norm=self.batch_norm,
            activation="relu",
        )
        self.conv2d_8 = ConvBlock(
            in_channels=48,
            out_channels=16,
            kernel_size=3,
            padding=(1, 1),
            batch_norm=self.batch_norm,
            activation="relu",
        )

        # D-1
        self.conv2d_9 = ConvBlock(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            padding=(1, 1),
            batch_norm=self.batch_norm,
            activation="relu",
        )
        self.conv2d_10 = ConvBlock(
            in_channels=48,
            out_channels=16,
            kernel_size=3,
            padding=(1, 1),
            batch_norm=self.batch_norm,
            activation="relu",
        )

        # D-0
        self.conv2d_11 = ConvBlock(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            padding=(1, 1),
            batch_norm=self.batch_norm,
            activation="relu",
        )

        # Output(s)
        # `sigmoid`` activation is performed in-shader due to jitter-aware-conv
        self.kpn_0_3_out_conv = ConvBlock(
            in_channels=16,
            out_channels=4,
            kernel_size=3,
            padding=(1, 1),
            batch_norm=False,
            activation="sigmoid",
        )
        self.kpn_4_7_out_conv = ConvBlock(
            in_channels=16,
            out_channels=4,
            kernel_size=3,
            padding=(1, 1),
            batch_norm=False,
            activation="sigmoid",
        )
        self.kpn_8_11_out_conv = ConvBlock(
            in_channels=16,
            out_channels=4,
            kernel_size=3,
            padding=(1, 1),
            batch_norm=False,
            activation="sigmoid",
        )
        self.kpn_12_15_out_conv = ConvBlock(
            in_channels=16,
            out_channels=4,
            kernel_size=3,
            padding=(1, 1),
            batch_norm=False,
            activation="sigmoid",
        )

        self.temporal_params_out_conv = ConvBlock(
            in_channels=16,
            out_channels=4,
            kernel_size=3,
            padding=(1, 1),
            batch_norm=False,
            activation="sigmoid",
        )
        self.feedback_out_conv = ConvBlock(
            in_channels=16,
            out_channels=feedback_ch,
            kernel_size=3,
            padding=(1, 1),
            batch_norm=False,
            activation="sigmoid",
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[Union[Tuple[torch.Tensor], torch.Tensor]]:
        """
        Input(s)
        --------
        x : torch.Tensor
            Combined input tensor of shape **(1, 540, 960, 12)**, consisting of:

            • **Warped previous prediction** — RGB, tonemapped with the Karis operator.
              Shape: **(1, 540, 960, 3)**. Already aligned to frame *t* via rendered motion
              vectors and linearly down-sampled. Zero-initialised on scene change or the
              first inference.

            • **Jittered current frame *t*** — RGB, Karis-tonemapped.
              Shape: **(1, 540, 960, 3)**.

            • **Disocclusion mask** — computed as in ASR.
              Shape: **(1, 540, 960, 1)**.

            • **Warped recurrent feedback features** — previous network output warped to *t*.
              Shape: **(1, 540, 960, 4)**.

            • **Luma derivative** — time-integrated variance of jittered-colour luminance;
              helps detect thin flickering features.
              Shape: **(1, 540, 960, 1)**.

        Output(s)
        ---------
        4 x 4 filter window with tap indices (column-major order):

                 0    4    8   12    ← 0: NW,  4: WNW,  8: WSW, 12: SW
                 1    5    9   13    ← 1: NNW, 5: N,    9: S,   13: SSW
                 2    6   10   14    ← 2: NNE, 6: E,   10: ESE, 14: SSE
                 3    7   11   15    ← 3: NE,  7: ENE, 11: E,   15: SE

        k0 : torch.Tensor of shape **(1, H, W, 4)**
            • Filter coefficients for taps 0, 4, 8, 12 (1st column: NW → SW).

        k1 : torch.Tensor of shape **(1, H, W, 4)**
            • Filter coefficients for taps 1, 5, 9, 13 (2nd column: NNW → SSW).

        k2 : torch.Tensor of shape **(1, H, W, 4)**
            • Filter coefficients for taps 2, 6, 10, 14 (3rd column: NNE → SSE).

        k3 : torch.Tensor of shape **(1, H, W, 4)**
            • Filter coefficients for taps 3, 7, 11, 15 (4th column: NE → SE).

        temporal_params : torch.Tensor of shape **(1, H, W, 4)**
            • Temporal parameters per pixel:
                • **θ (theta)** — history-rectification weight
                • **a (alpha)** — sample-accumulation weight
                • Final two channels are currently unused or reserved.

        feedback : torch.Tensor of shape **(1, H, W, 4)**
            • Four-channel temporal feedback passed into the next iteration.
        """

        # E-0
        x = self.conv2d_0(x)
        x = skip1 = self.conv2d_1(x)

        # E-1
        x = self.conv2d_2(x)
        x = skip2 = self.conv2d_3(x)

        # E-2
        x = self.conv2d_4(x)
        x = self.conv2d_5(x)

        # B
        x = self.conv2d_6(x)

        # D-2
        x = self.conv2d_7(self.upsample(x))
        x = torch.concatenate([x, skip2], dim=1)
        x = self.conv2d_8(x)

        # D-1
        x = self.conv2d_9(self.upsample(x))
        x = torch.concatenate([x, skip1], dim=1)
        x = self.conv2d_10(x)
        x = self.conv2d_11(x)

        # D-0 / Output(s)
        k0 = self.kpn_0_3_out_conv(self.upsample(x))
        k1 = self.kpn_4_7_out_conv(self.upsample(x))
        k2 = self.kpn_8_11_out_conv(self.upsample(x))
        k3 = self.kpn_12_15_out_conv(self.upsample(x))
        temporal_params = self.temporal_params_out_conv(self.upsample(x))
        feedback = self.feedback_out_conv(self.upsample(x))

        return (k0, k1, k2, k3), temporal_params, feedback
