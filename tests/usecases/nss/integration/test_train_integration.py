# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import unittest
from pathlib import Path

from tests.usecases.nss.integration.base_integration import BaseIntegrationTest


# pylint: disable=duplicate-code
class TrainingIntegrationTest(BaseIntegrationTest):
    """Tests for NSS training pipeline."""

    def _assert_peak_vram_usage(
        self, stdout: str, expected_vram_usage: int, tolerance: float
    ) -> None:
        """Assert vram usage does exceed expectation"""

        # Extract memory used from stdout
        stdout_lines = stdout.lower().splitlines()
        peak_vram_usage = None
        for line in reversed(stdout_lines):
            if "memory used:" in line:
                peak_vram_usage = float(line.split()[-2])
                break

        if peak_vram_usage is None:
            self.fail("Could not find GPU memory usage in stdout")

        print(f"Observed peak GPU memory usage: {peak_vram_usage:.2f} MiB")

        max_vram_allowed = expected_vram_usage * (1 + tolerance)
        self.assertLessEqual(
            peak_vram_usage,
            max_vram_allowed,
            (
                "Peak VRAM usage exceeded tolerance- "
                f"Peak usage: {peak_vram_usage:.2f} MiB, "
                f"Max allowance: {max_vram_allowed:.2f} MiB based on {expected_vram_usage:.2f}"
                f" with a tolerance of {1 + tolerance}%"
            ),
        )

    def run_finetune_training_test(self):
        """E2E test of the model to finetune training."""

        sub_proc = subprocess.run(
            [
                "ng-model-gym",
                f"--config-path={self.test_cfg_path}",
                "train",
                "--no-evaluate",
                "--finetune",
                self.finetune_weights,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(sub_proc.returncode, 0)
        self.check_log(["Fine tuning using weights"])
        return sub_proc

    def test_training_raises_error_missing_dataset(self):
        """Test train raises error if missing dataset path"""

        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        # Override dataset paths
        cfg_json["dataset"]["path"]["train"] = None
        cfg_json["dataset"]["path"]["validation"] = None
        cfg_json["dataset"]["path"]["test"] = None

        self.test_cfg_path = Path(self.test_dir, "test_empty_datasets_path.json")
        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        with self.assertRaises(subprocess.CalledProcessError) as sub_proc_out:
            subprocess.run(
                [
                    "ng-model-gym",
                    f"--config-path={self.test_cfg_path}",
                    "train",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        exc = sub_proc_out.exception

        self.assertNotEqual(
            exc.returncode,
            0,
            f"Expected non zero exit code for missing dataset, got {exc.returncode}",
        )

        self.assertIn("config error", exc.stderr.lower())

    def test_model_train(self):
        """Run entire training pipeline."""
        subprocess_out = self.run_training_test()
        self._assert_peak_vram_usage(subprocess_out.stdout, 14500, 0.005)

    def test_model_train_finetune(self):
        """Run entire training pipeline with finetuning."""
        subprocess_out = self.run_finetune_training_test()
        self._assert_peak_vram_usage(subprocess_out.stdout, 14500, 0.005)

    def test_model_train_resume(self):
        """Run entire training pipeline with resuming."""
        self.run_training_test()
        self.run_resume_training_test(mode="train", num_epochs=2)

    def test_train_flags_mutually_exclusive(self):
        """CLI should reject resume and finetune together."""
        with self.assertRaises(subprocess.CalledProcessError) as ctx:
            subprocess.run(
                [
                    "ng-model-gym",
                    f"--config-path={self.test_cfg_path}",
                    "train",
                    "--no-evaluate",
                    "--resume",
                    str(self.finetune_weights),
                    "--finetune",
                    str(self.finetune_weights),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(ctx.exception.returncode, 0)
        stderr_lower = ctx.exception.stderr.lower()
        for keyword in ("cannot specify both", "resume", "finetune"):
            self.assertIn(keyword, stderr_lower)

    def test_trace_profiler(self):
        """Test JSON trace is generated with profiler=trace flag"""
        self.run_model_profiler("train")

    def test_cuda_profiler(self):
        """Test trace is generated with profiler=gpu_memory flag"""
        self.run_cuda_profiler_test("train")

    @unittest.skip("AMP Currently disabled for training")
    def test_amp_training(self):
        """Test training with Automatic Mixed Precision policy"""

        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        cfg_json["train"]["amp"] = True

        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        sub_proc = subprocess.run(
            [
                "ng-model-gym",
                f"--config-path={self.test_cfg_path}",
                "train",
                "--no-evaluate",
            ]
        )
        self.assertEqual(sub_proc.returncode, 0)
        self.check_log("AMP is available: true")

    def test_validation(self):
        """Test train with validation"""
        # Update config to enable validation
        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        # Override validation dataset path and perform_validate
        cfg_json["dataset"]["path"]["validation"] = "tests/usecases/nss/datasets/val"
        cfg_json["train"]["perform_validate"] = True

        self.test_cfg_path = Path(self.test_dir, "test_validate.json")

        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        sub_proc = subprocess.run(
            [
                "ng-model-gym",
                f"--config-path={self.test_cfg_path}",
                "train",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(sub_proc.returncode, 0)
        self.assertIn("Validation:", sub_proc.stdout + sub_proc.stderr)

    def test_train_with_validation_missing_dataset(self):
        """Test train with validation raises error if missing dataset path"""

        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        # Override validation dataset path and perform_validate
        cfg_json["dataset"]["path"]["validation"] = None
        cfg_json["train"]["perform_validate"] = True

        self.test_cfg_path = Path(self.test_dir, "test_validate_error.json")

        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        with self.assertRaises(subprocess.CalledProcessError) as sub_proc_out:
            subprocess.run(
                ["ng-model-gym", f"--config-path={self.test_cfg_path}", "train"],
                check=True,
                capture_output=True,
                text=True,
            )

        exc = sub_proc_out.exception

        self.assertNotEqual(
            exc.returncode,
            0,
            f"Expected non zero exit code for missing dataset, got {exc.returncode}",
        )

        self.assertIn("config error", exc.stderr.lower())

    def test_train_no_eval_no_test_dataset(self):
        """Test train succeeds with no evaluate and no test dataset path"""

        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        # Disable test dataset and evaluation
        cfg_json["dataset"]["path"]["test"] = None
        cfg_json["train"]["perform_validate"] = False

        self.test_cfg_path = Path(self.test_dir, "test_no_eval_no_test_dataset.json")

        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        sub_proc = subprocess.run(
            [
                "ng-model-gym",
                f"--config-path={self.test_cfg_path}",
                "train",
                "--no-evaluate",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(sub_proc.returncode, 0)
        self.assertNotIn("config error", (sub_proc.stdout + sub_proc.stderr).lower())

    def test_train_evaluate_but_no_train_dataset(self):
        """Test train with evaluation but no test dataset raises error"""

        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        # Override validation dataset path and perform_validate
        cfg_json["dataset"]["path"]["test"] = None
        cfg_json["train"]["perform_validate"] = False

        self.test_cfg_path = Path(self.test_dir, "test_validate_error.json")

        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        with self.assertRaises(subprocess.CalledProcessError) as sub_proc_out:
            subprocess.run(
                ["ng-model-gym", f"--config-path={self.test_cfg_path}", "train"],
                check=True,
                capture_output=True,
                text=True,
            )

        exc = sub_proc_out.exception

        self.assertNotEqual(
            exc.returncode,
            0,
            f"Expected non zero exit code for missing dataset, got {exc.returncode}",
        )

        self.assertIn("config error: evaluation", exc.stderr.lower())

    def test_warning_train_with_validation_false_with_dataset(self):
        """Test train with validation set to false raises warning if dataset path provided"""

        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        # Override validation dataset path and perform_validate
        cfg_json["dataset"]["path"]["validation"] = "tests/usecases/nss/datasets/val"
        cfg_json["train"]["perform_validate"] = False

        self.test_cfg_path = Path(self.test_dir, "test_validate_warning.json")

        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        sub_proc = subprocess.run(
            [
                "ng-model-gym",
                f"--config-path={self.test_cfg_path}",
                "train",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn(
            "[WARNING] Validation path is provided but perform_validate is set to false",
            sub_proc.stdout,
        )

    def test_validation_false_no_dataset_provided(self):
        """Test train without validation and no dataset provided"""
        # Update config to enable validation
        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        # Override validation dataset path and perform_validate
        cfg_json["dataset"]["path"]["validation"] = None
        cfg_json["train"]["perform_validate"] = False

        self.test_cfg_path = Path(self.test_dir, "test_no_validate.json")

        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        sub_proc = subprocess.run(
            [
                "ng-model-gym",
                f"--config-path={self.test_cfg_path}",
                "train",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(sub_proc.returncode, 0)
        self.assertNotIn("Validation:", sub_proc.stdout + sub_proc.stderr)


# pylint: enable=duplicate-code
