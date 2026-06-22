# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

from ng_model_gym.usecases.nss.model.quality_modes import (
    NSSV1Quality,
    NSSV1QualitySettings,
    resolve_nss_v1_quality,
)


class TestNSSV1QualityModes(unittest.TestCase):
    """Tests for NSS v1 quality mode guardrails."""

    def test_none_defaults_to_high_quality(self):
        """Test missing NSS v1 quality resolves to high quality."""

        self.assertEqual(resolve_nss_v1_quality(None), NSSV1Quality.HIGH)

    def test_high_quality_is_supported(self):
        """Test explicit high quality resolves and enables its expected preset."""

        quality = resolve_nss_v1_quality("high")
        self.assertEqual(quality, NSSV1Quality.HIGH)

        settings = NSSV1QualitySettings.preset(quality)
        self.assertFalse(settings.preprocess_half_res_input)
        self.assertFalse(settings.depth_scatter_quarter_res_input)
        self.assertFalse(settings.use_sparse_filter_2x2)
        self.assertTrue(settings.use_history_catmull)
        self.assertFalse(settings.packed_nearest_offset_quad)
        self.assertEqual(settings.kpn_size, (6, 6))

    def test_mid_quality_is_supported(self):
        """Test mid quality resolves and enables its expected preset."""

        quality = resolve_nss_v1_quality("mid")
        self.assertEqual(quality, NSSV1Quality.MID)

        settings = NSSV1QualitySettings.preset(quality)
        self.assertTrue(settings.preprocess_half_res_input)
        self.assertTrue(settings.depth_scatter_quarter_res_input)
        self.assertTrue(settings.use_sparse_filter_2x2)
        self.assertTrue(settings.use_history_catmull)
        self.assertTrue(settings.packed_nearest_offset_quad)
        self.assertEqual(settings.kpn_size, (4, 4))

    def test_low_quality_is_supported(self):
        """Test low quality resolves and enables its expected preset."""

        quality = resolve_nss_v1_quality("low")
        self.assertEqual(quality, NSSV1Quality.LOW)

        settings = NSSV1QualitySettings.preset(quality)
        self.assertTrue(settings.preprocess_half_res_input)
        self.assertTrue(settings.depth_scatter_quarter_res_input)
        self.assertTrue(settings.use_sparse_filter_2x2)
        self.assertFalse(settings.use_history_catmull)
        self.assertTrue(settings.packed_nearest_offset_quad)
        self.assertEqual(settings.kpn_size, (4, 4))

    def test_unknown_quality_raises_value_error(self):
        """Test unknown NSS v1 quality values fail clearly."""

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported NSS-v1 quality",
        ):
            resolve_nss_v1_quality("fast")
