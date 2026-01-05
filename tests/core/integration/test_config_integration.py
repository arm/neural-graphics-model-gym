# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import json
import shutil
import subprocess
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path


class TestConfigSchemaIntegration(unittest.TestCase):
    """Integration tests for config schema validation"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        default_config_path = (
            files("ng_model_gym.usecases.nss.configs") / "default.json"
        )
        with default_config_path.open(encoding="utf-8") as f:
            self.default_config = json.load(f)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _run_train(self, config_dict):
        config_path = Path(self.test_dir, "config.json")
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config_dict, f)

        return subprocess.run(
            [
                "ng-model-gym",
                f"--config-path={config_path}",
                "train",
                "--no-evaluate",
            ],
            capture_output=True,
            text=True,
        )

    def test_cli_rejects_missing_schema_version(self):
        """Test configs without config_schema_version are rejected"""
        config = json.loads(json.dumps(self.default_config))
        config.pop("config_schema_version", None)

        result = self._run_train(config)
        output = result.stdout + result.stderr

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Configuration file version mismatch", output)
        self.assertIn("missing", output)

    def test_cli_rejects_outdated_schema_version(self):
        """Test configs with outdated config_schema_version are rejected"""
        config = json.loads(json.dumps(self.default_config))
        config["config_schema_version"] = "0"

        result = self._run_train(config)
        output = result.stdout + result.stderr

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Configuration file version mismatch", output)
        self.assertIn("Expected", output)
        self.assertIn("Provided", output)
        self.assertIn("0", output)
