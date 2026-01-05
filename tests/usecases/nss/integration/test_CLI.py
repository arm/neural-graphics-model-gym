# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from ng_model_gym import load_config_file
from tests.usecases.nss.integration.base_integration import BaseIntegrationTest


class CLIIntegrationTest(BaseIntegrationTest):
    """Tests for NSS training pipeline CLI options."""

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
        """CLI help interactions launch should be  less than <1 second"""
        cli_commands = [
            ["ng-model-gym", "--help"],
            ["ng-model-gym", "train", "--help"],
            ["ng-model-gym", "--version"],
            ["ng-model-gym", "qat", "--help"],
        ]

        for cmd in cli_commands:
            with self.subTest(cmd=" ".join(cmd)):
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
                    elapsed, 1.0, f"{cmd!r} took {elapsed:.2f}s (must be < 1s)"
                )

    def test_init_config_file(self):
        """Testing creating config file from CLI"""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"

            # First invocation: should create config.json and schema_config.json
            sub_process1 = subprocess.run(
                ["ng-model-gym", "init", "--out-dir", tmpdir],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(sub_process1.returncode, 0, sub_process1.stderr)

            # Expected files
            config1 = out_dir / "config.json"
            schema = out_dir / "schema_config.json"
            self.assertTrue(config1.exists())
            self.assertTrue(schema.exists())

            # Check CLI output mentions the right filenames
            self.assertIn("Config file written", sub_process1.stdout)
            self.assertIn("Schema file copied to", sub_process1.stdout)

            # Call second time, the file name should increment
            sub_process_2 = subprocess.run(
                ["ng-model-gym", "init", "--out-dir", tmpdir],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(sub_process_2.returncode, 0, sub_process_2.stderr)

            config2 = out_dir / "config_1.json"
            self.assertTrue(config2.exists())
            self.assertTrue(schema.exists())

            self.assertIn(
                "Config file written",
                sub_process_2.stdout,
            )

            with open(config1, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            self.assertEqual(
                config_data["dataset"]["path"]["train"],
                "<PATH/TO/TRAIN_DATA_DIR>",
            )
            self.assertEqual(
                config_data["train"]["fp32"]["checkpoints"]["dir"],
                "<OUTPUT/PATH/FOR/CHECKPOINTS_DIR>",
            )

            # Test loading config with placeholder fails
            output_buffer = io.StringIO()
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stdout(output_buffer):
                    load_config_file(config1)

            self.assertIn("Placeholder", output_buffer.getvalue())
