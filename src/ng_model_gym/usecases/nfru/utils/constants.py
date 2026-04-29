# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for NFRU Python utilities."""

# Motion vector quantisation params
_MAX_VAL = 1023.0
_BITS_EXP = 4
_BITS_X = 7
_BITS_Y = 7

# Colour processing defaults
_HALF_FLOAT_MAX = 65504.0
_TEMPERATURE_TINT_SCALE = 0.3
_REC709_LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)

# Randomized colour augmentation ranges
_RANDOM_CONTRAST_STRENGTH_RANGE = (0.85, 1.4)
_RANDOM_SATURATION_STRENGTH_RANGE = (0.75, 1.5)
_RANDOM_TEMPERATURE_RANGE = (-0.4, 0.4)
_RANDOM_TINT_RANGE = (-0.2, 0.2)
