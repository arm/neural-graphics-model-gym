# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

from ng_model_gym.core.model.shaders.slang_utils import load_slang_module
from tests.base_gpu_test import BaseGPUMemoryTest


class TestNSSV1ShaderLoading(BaseGPUMemoryTest):
    """Tests for NSS v1 shader module loading."""

    _REPO_ROOT = Path(__file__).resolve().parents[4]
    _NSS_V1_SHADER_SOURCES = (
        _REPO_ROOT / "src/ng_model_gym/usecases/nss/model/shaders/nss_v1.slang",
        _REPO_ROOT
        / "src/ng_model_gym/usecases/nss/model/shaders/nss_v1/pre_process.slang",
        _REPO_ROOT
        / "src/ng_model_gym/usecases/nss/model/shaders/nss_v1/post_process.slang",
    )

    def assert_nss_v1_exports(self, module):
        """Check that the NSS v1 shader module exposes required entry points."""

        self.assertTrue(hasattr(module, "depth_scatter"))
        self.assertTrue(hasattr(module, "pre_process"))
        self.assertTrue(hasattr(module, "post_process"))

    def test_nss_v1_shader_loads(self):
        """The v1 Slang wrapper can be loaded from package resources."""

        module = load_slang_module(
            "ng_model_gym.usecases.nss.model.shaders",
            "nss_v1.slang",
        )

        self.assert_nss_v1_exports(module)

    def test_nss_v1_shader_loads_with_v1_defines(self):
        """The v1 Slang wrapper accepts NSS v1 macro names."""

        module = load_slang_module(
            "ng_model_gym.usecases.nss.model.shaders",
            "nss_v1.slang",
            defines={
                "NSS_V1_LUMA_DERIVATIVE": 1,
                "NSS_V1_SHARP_THETA": 1,
                "SHADER_ACCURATE": True,
            },
        )

        self.assert_nss_v1_exports(module)

    def test_nss_v1_shader_sources_use_v1_macro_names_only(self):
        """The v1 shader port uses NSS v1 macro names only."""

        expected_versioned_macros = {
            "NSS_V1_LUMA_DERIVATIVE",
            "NSS_V1_SHARP_THETA",
        }
        found_versioned_macros = set()
        for shader_source in self._NSS_V1_SHADER_SOURCES:
            with self.subTest(shader_source=shader_source):
                macro_names = set(
                    re.findall(r"\bNSS_[A-Z0-9_]+\b", shader_source.read_text())
                )
                versioned_macros = {
                    macro_name
                    for macro_name in macro_names
                    if macro_name.startswith("NSS_V")
                }
                self.assertLessEqual(versioned_macros, expected_versioned_macros)
                found_versioned_macros.update(versioned_macros)

        self.assertEqual(found_versioned_macros, expected_versioned_macros)

    def test_nss_v1_shader_loads_with_v1_defines_disabled(self):
        """The NSS v1 macro names compile both enabled and disabled."""

        define_cases = (
            {
                "NSS_V1_LUMA_DERIVATIVE": 0,
                "NSS_V1_SHARP_THETA": 1,
                "SHADER_ACCURATE": True,
            },
            {
                "NSS_V1_LUMA_DERIVATIVE": 1,
                "NSS_V1_SHARP_THETA": 0,
                "SHADER_ACCURATE": True,
            },
            {
                "NSS_V1_LUMA_DERIVATIVE": 0,
                "NSS_V1_SHARP_THETA": 0,
                "SHADER_ACCURATE": True,
            },
        )

        for defines in define_cases:
            with self.subTest(defines=defines):
                module = load_slang_module(
                    "ng_model_gym.usecases.nss.model.shaders",
                    "nss_v1.slang",
                    defines=defines,
                )

                self.assert_nss_v1_exports(module)

    def test_nss_v1_shader_loads_with_defines_at_each_quality_level(self):
        """The v1 Slang wrapper loads with defines at each quality level."""

        define_cases = {
            "low": {
                "NSS_QUALITY": 0,
                "NSS_QUALITY_LOW": 0,
                "NSS_QUALITY_MEDIUM": 1,
                "NSS_QUALITY_HIGH": 2,
                "NSS_PREPROCESS_HALF_RES_INPUT": 1,
                "NSS_DEPTH_SCATTER_QUARTER_RES_INPUT": 1,
                "NSS_USE_SPARSE_2X2_FILTER": 1,
                "NSS_USE_HISTORY_CATMULL": 0,
                "NSS_PACKED_NEAREST_OFFSET_QUAD": 1,
                "FILTER_COLOUR_KERNEL_SZ": 4,
                "NSS_V1_LUMA_DERIVATIVE": 1,
                "NSS_V1_SHARP_THETA": 1,
                "SHADER_ACCURATE": True,
            },
            "mid": {
                "NSS_QUALITY": 1,
                "NSS_QUALITY_LOW": 0,
                "NSS_QUALITY_MEDIUM": 1,
                "NSS_QUALITY_HIGH": 2,
                "NSS_PREPROCESS_HALF_RES_INPUT": 1,
                "NSS_DEPTH_SCATTER_QUARTER_RES_INPUT": 1,
                "NSS_USE_SPARSE_2X2_FILTER": 1,
                "NSS_USE_HISTORY_CATMULL": 1,
                "NSS_PACKED_NEAREST_OFFSET_QUAD": 1,
                "FILTER_COLOUR_KERNEL_SZ": 4,
                "NSS_V1_LUMA_DERIVATIVE": 1,
                "NSS_V1_SHARP_THETA": 1,
                "SHADER_ACCURATE": True,
            },
            "high": {
                "NSS_QUALITY": 2,
                "NSS_QUALITY_LOW": 0,
                "NSS_QUALITY_MEDIUM": 1,
                "NSS_QUALITY_HIGH": 2,
                "NSS_PREPROCESS_HALF_RES_INPUT": 0,
                "NSS_DEPTH_SCATTER_QUARTER_RES_INPUT": 0,
                "NSS_USE_SPARSE_2X2_FILTER": 0,
                "NSS_USE_HISTORY_CATMULL": 1,
                "NSS_PACKED_NEAREST_OFFSET_QUAD": 0,
                "FILTER_COLOUR_KERNEL_SZ": 9,
                "NSS_V1_LUMA_DERIVATIVE": 1,
                "NSS_V1_SHARP_THETA": 1,
                "SHADER_ACCURATE": True,
            },
        }

        for quality, defines in define_cases.items():
            with self.subTest(quality=quality):
                module = load_slang_module(
                    "ng_model_gym.usecases.nss.model.shaders",
                    "nss_v1.slang",
                    defines=defines,
                )

                self.assert_nss_v1_exports(module)
