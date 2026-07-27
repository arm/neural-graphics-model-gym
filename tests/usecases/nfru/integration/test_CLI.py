# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import subprocess
import tempfile
from pathlib import Path

from tests.usecases.nfru.integration.base_integration import NFRUBaseIntegrationTest

# pylint: disable=duplicate-code


class CLIIntegrationTest(NFRUBaseIntegrationTest):
    """Tests for NFRU training pipeline CLI options."""

    def test_listing_models(self):
        """Test listing HF models lists NFRU"""
        sub_process = subprocess.run(
            ["ng-model-gym", "list-models"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(sub_process.returncode, 0, sub_process.stderr)

        self.assertIn("HuggingFace", sub_process.stdout)
        self.assertIn("neural-frame-rate-upscaling @ ", sub_process.stdout)
        self.assertIn(
            "https://huggingface.co/Arm/neural-frame-rate-upscaling", sub_process.stdout
        )
        self.assertIn("* nfru_v1_fp32.pt", sub_process.stdout)

    def test_downloading_models(self):
        """Test downloading NFRU model"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_process = subprocess.run(
                [
                    "ng-model-gym",
                    "download",
                    "neural-frame-rate-upscaling/nfru_v1_fp32.pt",
                    tmpdir,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(sub_process.returncode, 0, sub_process.stderr)

            self.assertIn(
                "Downloaded neural-frame-rate-upscaling/nfru_v1_fp32.pt to",
                sub_process.stdout,
            )

            self.assertTrue((tmpdir / Path("nfru_v1_fp32.pt")).exists())
