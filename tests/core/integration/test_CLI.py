# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from ng_model_gym import load_config_file
from tests.base_gpu_test import BaseGPUMemoryTest
from tests.testing_utils import create_simple_params


class CLIIntegrationTest(BaseGPUMemoryTest):
    """Tests for non usecase specific training pipeline CLI options."""

    def setUp(self):
        """Load a fresh config before each test."""
        super().setUp()

        self.test_dir = Path(tempfile.mkdtemp())

        # Create a valid config to use
        self.config = create_simple_params(
            usecase="nss",
            output_dir=self.test_dir / "output",
            dataset_path="tests/usecases/nss/mini_datasets/train",
            checkpoints=self.test_dir / "checkpoints",
        )

        self.config.model.recurrent_samples = 4
        self.config.train.batch_size = 4

        self.config.dataset.health_check = False

        self.test_cfg_path = Path(self.test_dir, "test_config.json")

        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(self.config.model_dump(mode="json"), f)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        super().tearDown()

    def run_training_logging_test(self, log_level):
        """Testing each of the log level options."""

        sub_proc = subprocess.run(
            [
                "ng-model-gym",
                f"--config-path={self.test_cfg_path}",
                f"--log-level={log_level}",
                "train",
            ]
        )
        self.assertEqual(sub_proc.returncode, 0)

    def test_logging(self):
        """Run entire training pipeline with each log level."""
        # pylint: disable-next=import-outside-toplevel
        from ng_model_gym.cli import AppLogLevel

        log_level_options = [log_level.value for log_level in AppLogLevel]

        for log_level in log_level_options:
            self.run_training_logging_test(log_level)

    def test_cli_invocation_time(self):
        """CLI help interactions launch should be less than <1 second"""
        cli_commands = [
            (["ng-model-gym", "--help"], 1.0),
            (["ng-model-gym", "train", "--help"], 1.0),
            (["ng-model-gym", "--version"], 1.0),
            (["ng-model-gym", "qat", "--help"], 1.0),
            (["ng-model-gym", "list-models", "--help"], 1.0),
            (["ng-model-gym", "download", "--help"], 1.0),
        ]

        warmup_proc = subprocess.run(
            ["ng-model-gym", "--help"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        self.assertEqual(
            warmup_proc.returncode,
            0,
        )

        # Check all subsequent help commands take less than the max time
        for cmd, max_time in cli_commands:
            with self.subTest(cmd=" ".join(cmd), max_time=max_time):
                start = time.perf_counter()
                sub_proc = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                elapsed = time.perf_counter() - start

                self.assertEqual(
                    sub_proc.returncode,
                    0,
                )

                self.assertLess(
                    elapsed,
                    max_time,
                    f"{cmd!r} took {elapsed:.2f}s (must be < {max_time:.1f}s)",
                )

    def test_listing_config(self):
        """Test listing config CLI command"""

        sub_process = subprocess.run(
            [
                "ng-model-gym",
                "config-options",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(sub_process.returncode, 0, sub_process.stderr)

        self.assertIn(
            "config_schema_version",
            sub_process.stdout,
        )

    def test_init_config_file(self):
        """Testing creating config files from CLI"""
        cases = [
            ("custom", "custom_config"),
            ("nss", "nss_config"),
            ("nfru", "nfru_config"),
        ]

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        for template, config_path in cases:
            with self.subTest(template=template):
                with tempfile.TemporaryDirectory() as tmpdir:
                    out_dir = Path(tmpdir)

                    # First invocation: should create specific template json and schema_config.json
                    sub_process1 = subprocess.run(
                        ["ng-model-gym", "init", template, tmpdir],
                        capture_output=True,
                        text=True,
                        env=env,
                    )

                    self.assertEqual(sub_process1.returncode, 0, sub_process1.stderr)

                    # Expected files
                    config1 = out_dir / f"{config_path}.json"
                    schema = out_dir / "schema_config.json"
                    self.assertTrue(config1.exists())
                    self.assertTrue(schema.exists())

                    # Check CLI output mentions the right filenames
                    self.assertIn("Config file written", sub_process1.stdout)
                    self.assertIn("Schema file copied to", sub_process1.stdout)

                    # Call second time, the file name should increment
                    sub_process_2 = subprocess.run(
                        ["ng-model-gym", "init", template, tmpdir],
                        capture_output=True,
                        text=True,
                        env=env,
                    )
                    self.assertEqual(sub_process_2.returncode, 0, sub_process_2.stderr)

                    config2 = out_dir / f"{config_path}_1.json"
                    self.assertTrue(config2.exists())
                    self.assertTrue(schema.exists())

                    self.assertIn(
                        "Config file written",
                        sub_process_2.stdout,
                    )

                    # Test loading config with placeholder fails
                    # Only need to check for one template file
                    if template == "nss":
                        with open(config1, "r", encoding="utf-8") as f:
                            config_data = json.load(f)

                        self.assertEqual(
                            config_data["dataset"]["path"]["train"],
                            "<PATH/TO/TRAIN_DATA_DIR>",
                        )

                        output_buffer = io.StringIO()
                        with self.assertRaises(SystemExit):
                            with contextlib.redirect_stdout(output_buffer):
                                load_config_file(config1)

                        self.assertIn("Placeholder", output_buffer.getvalue())
