# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import unittest

from ng_model_gym.core.model.shaders.slang_utils import load_slang_module


class TestNSSV1ShaderLoading(unittest.TestCase):
    """Tests for NSS v1 shader module loading."""

    def test_nss_v1_shader_loads(self):
        """The v1 Slang wrapper can be loaded from package resources."""

        module = load_slang_module(
            "ng_model_gym.usecases.nss.model.shaders",
            "nss_v1.slang",
        )

        self.assertTrue(hasattr(module, "depth_scatter"))
        self.assertTrue(hasattr(module, "pre_process"))
        self.assertTrue(hasattr(module, "post_process"))
