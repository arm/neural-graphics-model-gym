# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code
"""NSS v1 model blocks."""

import math
from typing import Optional, Tuple

import torch
from torch import nn

from ng_model_gym.core.model.layers.conv_block import ConvBlock


class AutoEncoderV1(nn.Module):
    """NSS v1 backbone."""

    def __init__(
        self,
        in_channels: Optional[int] = 12,
        temporal_ch: Optional[int] = 4,
        kpn_size: Optional[Tuple[int, int]] = (6, 6),
        batch_norm: Optional[bool] = False,
    ):
        super().__init__()

        self.batch_norm = batch_norm

        self.in_channels = in_channels
        self.temporal_ch = temporal_ch
        self.kpn_ch = math.prod(kpn_size)

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
            out_channels=32,
            kernel_size=3,
            padding=(1, 1),
            batch_norm=self.batch_norm,
            activation="relu",
        )

        # D-1
        self.conv2d_9 = ConvBlock(
            in_channels=32,
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

        # Outputs
        self.kpn_params = ConvBlock(
            in_channels=32,
            out_channels=self.kpn_ch,
            kernel_size=3,
            padding=(1, 1),
            batch_norm=False,
            activation="sigmoid",
        )
        self.temporal_params_out_conv = ConvBlock(
            in_channels=16,
            out_channels=self.temporal_ch,
            kernel_size=3,
            padding=(1, 1),
            batch_norm=False,
            activation="sigmoid",
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the AutoEncoderV1 forward pass."""

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

        # Output(s) @ 16th-res
        kpn_params = self.kpn_params(x)

        # D-1
        x = self.conv2d_9(self.upsample(x))
        x = torch.concatenate([x, skip1], dim=1)
        x = self.conv2d_10(x)
        x = self.conv2d_11(x)

        # Output(s) @ input-res
        temporal_params = self.temporal_params_out_conv(self.upsample(x))

        return kpn_params, temporal_params


def get_kpn_prune_indices(
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
    mode: str = "centered",
) -> Tuple[int, ...]:
    """Return channel indices for pruning a larger KPN grid down to a smaller one.

    Channels are flattened in column-major order:
        tap_index(row, col) = row + col * height
    """
    source_h, source_w = (int(v) for v in source_size)
    target_h, target_w = (int(v) for v in target_size)

    if source_h < target_h or source_w < target_w:
        raise ValueError(
            f"Cannot prune KPN from smaller source grid {source_size} "
            f"to larger target {target_size}"
        )

    if mode != "centered":
        raise ValueError(f"Unsupported KPN prune mode: {mode}")

    row_delta = source_h - target_h
    col_delta = source_w - target_w
    if (row_delta % 2) != 0 or (col_delta % 2) != 0:
        raise ValueError(
            "Centered KPN prune requires even deltas, "
            f"got source={source_size}, target={target_size}"
        )

    row_offset = row_delta // 2
    col_offset = col_delta // 2

    indices = []
    for col in range(col_offset, col_offset + target_w):
        for row in range(row_offset, row_offset + target_h):
            indices.append(row + col * source_h)

    return tuple(indices)
