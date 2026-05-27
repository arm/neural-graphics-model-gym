# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import unittest
from importlib.metadata import version

import ng_model_gym
from ng_model_gym._version import get_version


class TestVersion(unittest.TestCase):
    """Unit test for package version."""

    def test_version_attribute_matches_metadata(self):
        """`ng_model_gym.__version__` should match the installed package metadata version."""
        version_attr = ng_model_gym.__version__
        metadata_version = version("ng-model-gym")

        self.assertEqual(version_attr, metadata_version)

    def test_get_version_matches_metadata(self):
        """`get_version()` should return the installed package metadata version."""
        helper_version = get_version()
        metadata_version = version("ng-model-gym")

        self.assertEqual(helper_version, metadata_version)
