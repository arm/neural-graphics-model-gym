# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

from ng_model_gym.usecases.nss.model.model_v0_1 import (
    NSSModel,  # TODO: Remove this import when NSS v0.1 is removed
)
from tests.testing_utils import create_simple_params


class TestNSSV0_1Deprecation(unittest.TestCase):
    """Tests for legacy NSS v0.1 deprecation behavior."""

    def test_nss_v0_1_model_warns_on_instantiation(self):
        """Creating the legacy NSS v0.1 model emits a deprecation warning."""
        params = create_simple_params(usecase="nss")

        with self.assertWarnsRegex(
            DeprecationWarning,
            "NSS v0.1 is deprecated.*NSS v1",
        ):
            NSSModel(params)
