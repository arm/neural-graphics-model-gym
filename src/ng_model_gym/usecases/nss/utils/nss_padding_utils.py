# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
from typing import NamedTuple, Tuple


class Resolution(NamedTuple):
    """Named tuple for resolution"""

    height: int
    width: int


class NSSPaddingPolicy:
    """Calculate and apply padding/unpadding for NSS models"""

    def __init__(self, lr: Resolution, hr: Resolution, multiple: int, scale: int):
        """Precalculate padding for the lr and hr tensors"""

        if scale != 2:
            raise ValueError("Supported scale for padding is 2")

        self.lr = lr
        self.hr = hr

        self.lr_padding = self._compute_padding_next_multiple(self.lr, multiple)
        lr_pad_h, lr_pad_w = self.lr_padding

        self.hr_padding = (lr_pad_h * scale, lr_pad_w * scale)
        hr_pad_h, hr_pad_w = self.hr_padding

        self.padded_lr = (self.lr.height + lr_pad_h, self.lr.width + lr_pad_w)
        self.padded_hr = (self.hr.height + hr_pad_h, self.hr.width + hr_pad_w)

    def _compute_padding_next_multiple(
        self, res: Resolution, multiple
    ) -> tuple[int, int]:
        """Calculate how much to pad a resolution to a given multiple"""
        pad_h = (-res.height) % multiple
        pad_w = (-res.width) % multiple
        return pad_h, pad_w

    def calculate_padding(
        self, height: int, width: int, is_unpad: bool
    ) -> Tuple[int, int]:
        """Given height, width and is_unpad, calculate how much padding is applied"""

        match (height, width), is_unpad:
            # Scalar values
            case (1, 1), _:
                return 0, 0

            # Pad cases
            case self.lr, False:
                return self.lr_padding
            case self.hr, False:
                return self.hr_padding

            # Unpad cases
            case self.padded_lr, True:
                return self.lr_padding
            case self.padded_hr, True:
                return self.hr_padding

            case _:
                raise ValueError(
                    f"Unexpected height/width resolution: "
                    f"({height=}, {width=}), {is_unpad=}"
                )
