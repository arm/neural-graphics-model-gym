# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import unittest
from pathlib import Path

from tests.nss.integration.base_integration import BaseIntegrationTest

# pylint: disable=duplicate-code


class QATIntegrationTest(BaseIntegrationTest):
    """Tests for NSS training pipeline."""

    # pylint: disable=duplicate-code
    def run_finetune_training_test(self):
        """E2E test of the model to finetune training."""

        sub_proc = subprocess.run(
            [
                "ng-model-gym",
                f"--config-path={self.test_cfg_path}",
                "qat",
                "--no-evaluate",
                "--finetune",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(sub_proc.returncode, 0)

        out = sub_proc.stdout.decode("utf-8", errors="ignore")
        err = sub_proc.stderr.decode("utf-8", errors="ignore")

        self.assertEqual(
            sub_proc.returncode,
            0,
            (
                f"QAT finetune failed exit code: {sub_proc.returncode})\n"
                f"STDOUT:\n{out}\n"
                f"STDERR: \n{err}"
            ),
        )
        # pylint: enable=duplicate-code

    def test_qat_raises_error_missing_dataset(self):
        """Test qat raises error if missing dataset path"""

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
                    "qat",
                    "--no-evaluate",
                    "--finetune",
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
        self.run_training_test_qat()

    def test_model_train_finetune(self):
        """Run entire training pipeline with finetuning."""
        self.run_finetune_training_test()

    def test_model_train_resume(self):
        """Run entire training pipeline with resuming."""
        self.test_model_train()
        self.run_resume_training_test(mode="qat", num_epochs=3)

    def test_trace_profiler(self):
        """Test JSON trace is generated with profiler=trace flag"""
        self.run_model_profiler("qat")

    def test_cuda_profiler(self):
        """Test trace is generated with profiler=gpu_memory flag"""
        self.run_cuda_profiler_test("qat")


if __name__ == "__main__":
    unittest.main()
