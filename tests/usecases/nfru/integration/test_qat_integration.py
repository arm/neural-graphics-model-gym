# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import unittest
from pathlib import Path

from tests.usecases.nfru.integration.base_integration import NFRUBaseIntegrationTest

# pylint: disable=duplicate-code


@unittest.skip("NFRU CI/assets disabled for now")
class QATIntegrationTest(NFRUBaseIntegrationTest):
    """Tests for NFRU QAT pipeline."""

    def run_finetune_training_test(self):
        """E2E test of QAT with finetuning from local model weights."""
        sub_proc = subprocess.run(
            [
                *self._cli_base(),
                f"--config-path={self.test_cfg_path}",
                "qat",
                "--no-evaluate",
                "--finetune",
                self.eval_weights,
            ],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src", **os.environ},
        )
        self.assertEqual(sub_proc.returncode, 0)
        self.check_log(["Fine tuning using weights nfru_v1_fp32.pt"])

    def run_finetune_training_test_hf_model(self):
        """E2E test of QAT with finetuning from HF model identifier."""
        sub_proc = subprocess.run(
            [
                *self._cli_base(),
                f"--config-path={self.test_cfg_path}",
                "qat",
                "--no-evaluate",
                "--finetune",
                "@neural-framerate-upscaling/nfru_v1_fp32.pt",
            ],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src", **os.environ},
        )
        self.assertEqual(sub_proc.returncode, 0)
        self.check_log(["Fine tuning using weights nfru_v1_fp32.pt"])

    def test_qat_raises_error_missing_dataset(self):
        """Test qat raises error when dataset paths are missing."""
        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        cfg_json["dataset"]["path"]["train"] = None
        cfg_json["dataset"]["path"]["validation"] = None
        cfg_json["dataset"]["path"]["test"] = None

        self.test_cfg_path = Path(self.test_dir, "test_empty_datasets_path.json")
        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        with self.assertRaises(subprocess.CalledProcessError) as sub_proc_out:
            subprocess.run(
                [
                    *self._cli_base(),
                    f"--config-path={self.test_cfg_path}",
                    "qat",
                    "--no-evaluate",
                    "--finetune",
                    self.eval_weights,
                ],
                check=True,
                capture_output=True,
                text=True,
                env={"PYTHONPATH": "src", **os.environ},
            )

        exc = sub_proc_out.exception
        self.assertNotEqual(exc.returncode, 0)
        self.assertIn("config error", exc.stderr.lower())

    def test_model_train(self):
        """Run entire QAT pipeline."""
        self.run_training_test_qat()

    def test_model_train_finetune(self):
        """Run entire QAT pipeline with finetuning."""
        self.run_finetune_training_test()

    @unittest.skip("TODO: enable once HF assets are added for NFRU QAT.")
    def test_model_train_finetune_model_from_hf(self):
        """Run finetuning pipeline using model from HF using unique identifier."""
        self.run_finetune_training_test_hf_model()

    def test_model_train_resume(self):
        """Run entire QAT pipeline with resume."""
        self.test_model_train()
        self.run_resume_training_test(mode="qat", num_epochs=3)

    def test_qat_flags_mutually_exclusive(self):
        """CLI should reject resume and finetune together."""
        with self.assertRaises(subprocess.CalledProcessError) as ctx:
            subprocess.run(
                [
                    *self._cli_base(),
                    f"--config-path={self.test_cfg_path}",
                    "qat",
                    "--no-evaluate",
                    "--resume",
                    self.qat_checkpoint_dir,
                    "--finetune",
                    self.eval_weights,
                ],
                check=True,
                capture_output=True,
                text=True,
                env={"PYTHONPATH": "src", **os.environ},
            )

        self.assertNotEqual(ctx.exception.returncode, 0)
        stderr_lower = ctx.exception.stderr.lower()
        for keyword in ("cannot specify both", "resume", "finetune"):
            self.assertIn(keyword, stderr_lower)

    @unittest.skip(
        "Issue: trace profiler export for NFRU eval fails on short runs "
        "with AssertionError 'Profiler must be initialized before exporting chrome trace'."
        "NFRU dataset has fewer samples than NSS, so these will be skipped due to the existing "
        "profiler schedule."
    )
    def test_trace_profiler(self):
        """Test trace file is generated with profiler=trace flag."""
        self.run_model_profiler("qat")

    def test_cuda_profiler(self):
        """Test profile snapshot is generated with profiler=gpu_memory flag."""
        self.run_cuda_profiler("qat")
