# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

from ng_model_gym.usecases.nss.model.quality_modes import (
    NSSV1Quality,
    resolve_nss_v1_quality,
)


class TestNSSV1QualityModes(unittest.TestCase):
    """Tests for NSS v1 quality mode guardrails."""

    def test_none_defaults_to_high_quality(self):
        """Test missing NSS v1 quality resolves to high quality."""

        self.assertEqual(resolve_nss_v1_quality(None), NSSV1Quality.HIGH)

    def test_high_quality_is_supported(self):
        """Test explicit high quality resolves to high quality."""

        self.assertEqual(resolve_nss_v1_quality("high"), NSSV1Quality.HIGH)

    def test_mid_quality_is_planned_follow_up(self):
        """Test mid quality is represented but unsupported initially."""

        with self.assertRaisesRegex(
            NotImplementedError,
            "NSS-v1 quality 'mid'",
        ):
            resolve_nss_v1_quality("mid")

    def test_low_quality_is_planned_follow_up(self):
        """Test low quality is represented but unsupported initially."""

        with self.assertRaisesRegex(
            NotImplementedError,
            "NSS-v1 quality 'low'",
        ):
            resolve_nss_v1_quality("low")

    def test_unknown_quality_raises_value_error(self):
        """Test unknown NSS v1 quality values fail clearly."""

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported NSS-v1 quality",
        ):
            resolve_nss_v1_quality("fast")
