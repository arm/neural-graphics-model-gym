# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import unittest
from pathlib import Path

from tests.nss.integration.base_integration import BaseIntegrationTest


# pylint: disable=duplicate-code
class TrainingIntegrationTest(BaseIntegrationTest):
    """Tests for NSS training pipeline."""

    def run_finetune_training_test(self):
        """E2E test of the model to finetune training."""

        sub_proc = subprocess.run(
            [
                "ng-model-gym",
                f"--config-path={self.test_cfg_path}",
                "train",
                "--no-evaluate",
                "--finetune",
            ]
        )
        self.assertEqual(sub_proc.returncode, 0)
        self.check_log(["Fine tuning using weights"])

    def test_training_raises_error_missing_dataset(self):
        """Test train raises error if missing dataset path"""

        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        # Override number_of_epochs to test resuming
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
        self.run_training_test()

    def test_model_train_finetune(self):
        """Run entire training pipeline with finetuning."""
        self.run_finetune_training_test()

    def test_model_train_resume(self):
        """Run entire training pipeline with resuming."""
        self.run_training_test()
        self.run_resume_training_test(mode="train", num_epochs=2)

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


# pylint: enable=duplicate-code


if __name__ == "__main__":
    unittest.main()
