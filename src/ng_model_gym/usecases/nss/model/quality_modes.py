# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
"""Quality mode guardrails for NSS v1."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


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
        NSS v1 quality mode equivalent to the passed string.

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

    return quality


@dataclass(frozen=True)
class NSSV1QualitySettings:
    """
    The different NSSV1Quality levels are implemented by turning parts of the
    algorithm on and off. This class defines which parts of the algorithm will
    be enabled for each given quality level.
    """

    quality: NSSV1Quality
    preprocess_half_res_input: bool
    depth_scatter_quarter_res_input: bool
    use_sparse_filter_2x2: bool
    use_history_catmull: bool
    packed_nearest_offset_quad: bool
    low_mid_luma_derivative: bool
    kpn_size: Tuple[int, int]
    filter_kernel_size: int

    @classmethod
    def preset(cls, quality: NSSV1Quality) -> "NSSV1QualitySettings":
        """
        Returns the parts of the NSS algorithm corresponding to the given
        quality level.
        """

        if quality == NSSV1Quality.LOW:
            return cls(
                quality=quality,
                preprocess_half_res_input=True,
                depth_scatter_quarter_res_input=True,
                use_sparse_filter_2x2=True,
                use_history_catmull=False,
                packed_nearest_offset_quad=True,
                low_mid_luma_derivative=True,
                kpn_size=(4, 4),
                filter_kernel_size=2,
            )

        if quality == NSSV1Quality.MID:
            return cls(
                quality=quality,
                preprocess_half_res_input=True,
                depth_scatter_quarter_res_input=True,
                use_sparse_filter_2x2=True,
                use_history_catmull=True,
                packed_nearest_offset_quad=True,
                low_mid_luma_derivative=True,
                kpn_size=(4, 4),
                filter_kernel_size=2,
            )

        assert quality == NSSV1Quality.HIGH  # nosec B101

        return cls(
            quality=quality,
            preprocess_half_res_input=False,
            depth_scatter_quarter_res_input=False,
            use_sparse_filter_2x2=False,
            use_history_catmull=True,
            packed_nearest_offset_quad=False,
            low_mid_luma_derivative=False,
            kpn_size=(6, 6),
            filter_kernel_size=3,
        )
