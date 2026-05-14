# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
"""Quality mode guardrails for NSS v1."""

from enum import Enum
from typing import Optional


class NSSV1Quality(str, Enum):
    """NSS v1 quality modes."""

    HIGH = "high"
    MID = "mid"
    LOW = "low"


def resolve_nss_v1_quality(value: Optional[str]) -> NSSV1Quality:
    """Resolve and validate the requested NSS v1 quality mode.

    Args:
        value: Quality mode string from configuration, or None for the default.

    Returns:
        The supported high-quality NSS v1 mode.

    Raises:
        ValueError: If the quality mode is unknown.
        NotImplementedError: If the quality mode is known but not implemented.
    """

    if value is None:
        return NSSV1Quality.HIGH

    normalized_value = value.strip().lower()
    try:
        quality = NSSV1Quality(normalized_value)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported NSS-v1 quality '{value}'. "
            "Supported quality values are: high, mid, low."
        ) from exc

    if quality in (NSSV1Quality.MID, NSSV1Quality.LOW):
        raise NotImplementedError(
            f"NSS-v1 quality '{quality.value}' is planned follow-up work. "
            "The initial NSS-v1 implementation supports high-quality "
            "training/evaluation only."
        )

    return quality
