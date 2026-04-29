# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path


class TestCLIDatasetValidationIntegration(unittest.TestCase):
    """Shared CLI integration tests for dataset-path validation."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        default_config_path = (
            files("ng_model_gym.usecases.nss.configs") / "nss_template.json"
        )
        with default_config_path.open(encoding="utf-8") as f:
            self.default_config = json.load(f)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @staticmethod
    def _cli_base():
        return [sys.executable, "-m", "ng_model_gym.cli"]

    @staticmethod
    def _test_env():
        return {"PYTHONPATH": "src", **os.environ}

    def _run_cli(self, config_dict, *args):
        config_path = Path(self.test_dir, "config.json")
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config_dict, f)

        return subprocess.run(
            [*self._cli_base(), f"--config-path={config_path}", *args],
            capture_output=True,
            text=True,
            env=self._test_env(),
        )

    def _valid_config(self):
        """Return a template config with concrete dataset paths."""
        config = json.loads(json.dumps(self.default_config))
        config["dataset"]["path"]["train"] = "tests/usecases/nss/datasets/train"
        config["dataset"]["path"]["validation"] = "tests/usecases/nss/datasets/val"
        config["dataset"]["path"]["test"] = "tests/usecases/nss/datasets/test"
        return config

    def test_train_with_evaluate_rejects_missing_test_dataset_path(self):
        """CLI train --evaluate should fail when test dataset path is missing."""
        config = self._valid_config()
        config["dataset"]["path"]["test"] = None

        result = self._run_cli(config, "train", "--evaluate")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "config error: evaluation is specified but no test dataset path is provided",
            (result.stdout + result.stderr).lower(),
        )

    def test_train_rejects_missing_train_dataset_path(self):
        """CLI train should fail when train dataset path is missing."""
        config = self._valid_config()
        config["dataset"]["path"]["train"] = None

        result = self._run_cli(config, "train", "--no-evaluate")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "config error: no path specified for the train dataset path",
            (result.stdout + result.stderr).lower(),
        )

    def test_qat_with_evaluate_rejects_missing_test_dataset_path(self):
        """CLI qat --evaluate should fail when test dataset path is missing."""
        config = self._valid_config()
        config["dataset"]["path"]["test"] = None

        result = self._run_cli(config, "qat", "--evaluate")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "config error: evaluation is specified but no test dataset path is provided",
            (result.stdout + result.stderr).lower(),
        )

    def test_evaluate_rejects_missing_test_dataset_path(self):
        """CLI evaluate should fail when test dataset path is missing."""
        config = self._valid_config()
        config["dataset"]["path"]["test"] = None

        result = self._run_cli(
            config,
            "evaluate",
            "--model-path=tests/usecases/nss/weights/nss_v0.1.0_fp32.pt",
            "--model-type=fp32",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "config error: no test dataset path provided for evaluation",
            (result.stdout + result.stderr).lower(),
        )

    def test_train_with_validation_rejects_missing_validation_dataset_path(self):
        """CLI train should fail when validation is enabled without validation path."""
        config = self._valid_config()
        config["train"]["perform_validate"] = True
        config["dataset"]["path"]["validation"] = None

        result = self._run_cli(config, "train", "--evaluate")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            (
                "config error: perform validate is enabled and no path specified "
                "for validation dataset"
            ),
            (result.stdout + result.stderr).lower(),
        )
