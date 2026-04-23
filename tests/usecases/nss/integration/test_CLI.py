# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import subprocess
import tempfile
from pathlib import Path

from tests.usecases.nss.integration.base_integration import NSSBaseIntegrationTest

# pylint: disable=duplicate-code


class CLIIntegrationTest(NSSBaseIntegrationTest):
    """Tests for NSS training pipeline CLI options."""

    def test_listing_models(self):
        """Test listing HF models lists NSS"""
        sub_process = subprocess.run(
            ["ng-model-gym", "list-models"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(sub_process.returncode, 0, sub_process.stderr)

        self.assertIn("HuggingFace", sub_process.stdout)
        self.assertIn("neural-super-sampling @ ", sub_process.stdout)
        self.assertIn(
            "https://huggingface.co/Arm/neural-super-sampling", sub_process.stdout
        )
        self.assertIn("* nss_v0.1.0_fp32.pt", sub_process.stdout)

    def test_downloading_models(self):
        """Test downloading NSS model"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_process = subprocess.run(
                [
                    "ng-model-gym",
                    "download",
                    "neural-super-sampling/nss_v0.1.0_fp32.pt",
                    tmpdir,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(sub_process.returncode, 0, sub_process.stderr)

            self.assertIn(
                "Downloaded neural-super-sampling/nss_v0.1.0_fp32.pt to",
                sub_process.stdout,
            )

            self.assertTrue((tmpdir / Path("nss_v0.1.0_fp32.pt")).exists())
