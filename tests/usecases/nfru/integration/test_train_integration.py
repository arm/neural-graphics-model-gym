# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import unittest
from pathlib import Path

from tests.usecases.nfru.integration.base_integration import NFRUBaseIntegrationTest

# pylint: disable=duplicate-code


@unittest.skip("NFRU CI/assets disabled for now")
class TrainingIntegrationTest(NFRUBaseIntegrationTest):
    """Tests for NFRU training pipeline."""

    def run_finetune_training_test(self):
        """E2E test of model fine-tuning on local checkpoint."""
        sub_proc = subprocess.run(
            [
                *self._cli_base(),
                f"--config-path={self.test_cfg_path}",
                "train",
                "--no-evaluate",
                "--finetune",
                self.eval_weights,
            ],
            capture_output=True,
            text=True,
            env=self._test_env(),
        )
        self.assertEqual(sub_proc.returncode, 0)
        self.check_log(["Fine tuning using weights nfru_v1_fp32.pt"])
        return sub_proc

    @unittest.skip(
        "TODO: add stable remote NFRU checkpoint identifier and CI coverage for "
        "train-finetune-from-identifier flow."
    )
    def run_finetune_training_test_model_hf(self):
        """E2E test of model fine-tuning from a remote model identifier."""
        sub_proc = subprocess.run(
            [
                *self._cli_base(),
                f"--config-path={self.test_cfg_path}",
                "train",
                "--no-evaluate",
                "--finetune",
                "@neural-framerate-upscaling/nfru_v1_fp32.pt",
            ],
            capture_output=True,
            text=True,
            env=self._test_env(),
        )
        self.assertEqual(sub_proc.returncode, 0)
        self.check_log(["Fine tuning using weights nfru_v1_fp32.pt"])
        return sub_proc

    def test_model_train(self):
        """Run train pipeline."""
        subprocess_out = self.run_training_test()
        self._assert_peak_vram_usage(subprocess_out.stdout, 14800, 0.005)

    def test_model_train_finetune(self):
        """Run train pipeline with fine-tuning."""
        subprocess_out = self.run_finetune_training_test()
        self._assert_peak_vram_usage(subprocess_out.stdout, 14800, 0.005)

    @unittest.skip(
        "TODO: add stable remote NFRU checkpoint identifier and CI coverage for "
        "train-finetune-from-identifier flow."
    )
    def test_model_train_finetune_model_from_hf(self):
        """Run train pipeline with fine-tuning from remote identifier."""
        self.run_finetune_training_test_model_hf()

    def test_model_train_resume(self):
        """Run train pipeline with resume."""
        self.run_training_test()
        self.run_resume_training_test(mode="train", num_epochs=2)

    def test_train_flags_mutually_exclusive(self):
        """CLI should reject resume and finetune together."""
        with self.assertRaises(subprocess.CalledProcessError) as ctx:
            subprocess.run(
                [
                    *self._cli_base(),
                    f"--config-path={self.test_cfg_path}",
                    "train",
                    "--no-evaluate",
                    "--resume",
                    str(self.eval_weights),
                    "--finetune",
                    str(self.eval_weights),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=self._test_env(),
            )

        self.assertNotEqual(ctx.exception.returncode, 0)
        stderr_lower = ctx.exception.stderr.lower()
        for keyword in ("cannot specify both", "resume", "finetune"):
            self.assertIn(keyword, stderr_lower)

    @unittest.skip(
        "Issue: trace profiler export for NFRU train fails on short runs "
        "with AssertionError 'Profiler must be initialized before exporting chrome trace'. "
        "Pending upstream/profiler schedule stabilization."
    )
    def test_trace_profiler(self):
        """Test JSON trace is generated with profiler=trace."""
        self.run_model_profiler("train")

    def test_cuda_profiler(self):
        """Test trace is generated with profiler=gpu_memory."""
        self.run_cuda_profiler("train")

    def test_validation(self):
        """Test train with validation."""
        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        cfg_json["dataset"]["path"]["validation"] = self.sample_data_dir
        cfg_json["train"]["perform_validate"] = True
        self.test_cfg_path = Path(self.test_dir, "test_validate.json")

        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        sub_proc = subprocess.run(
            [
                *self._cli_base(),
                f"--config-path={self.test_cfg_path}",
                "train",
                "--no-evaluate",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=self._test_env(),
        )

        self.assertEqual(sub_proc.returncode, 0)
        self.assertIn("Validation:", sub_proc.stdout + sub_proc.stderr)

    def test_train_no_eval_no_test_dataset(self):
        """Test train succeeds with --no-evaluate and no test dataset path."""
        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        cfg_json["dataset"]["path"]["test"] = None
        cfg_json["train"]["perform_validate"] = False
        self.test_cfg_path = Path(self.test_dir, "test_no_eval_no_test_dataset.json")

        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        sub_proc = subprocess.run(
            [
                *self._cli_base(),
                f"--config-path={self.test_cfg_path}",
                "train",
                "--no-evaluate",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=self._test_env(),
        )

        self.assertEqual(sub_proc.returncode, 0)
        self.assertNotIn("config error", (sub_proc.stdout + sub_proc.stderr).lower())

    def test_train_evaluate_but_no_test_dataset(self):
        """Test train with evaluate enabled raises error when test path is missing."""
        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        cfg_json["dataset"]["path"]["test"] = None
        cfg_json["train"]["perform_validate"] = False
        self.test_cfg_path = Path(self.test_dir, "test_train_eval_error.json")

        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        with self.assertRaises(subprocess.CalledProcessError) as sub_proc_out:
            subprocess.run(
                [
                    *self._cli_base(),
                    f"--config-path={self.test_cfg_path}",
                    "train",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=self._test_env(),
            )

        exc = sub_proc_out.exception
        self.assertNotEqual(exc.returncode, 0)
        self.assertIn("config error: evaluation", exc.stderr.lower())

    def test_warning_train_with_validation_false_with_dataset(self):
        """Test warning is emitted when validation path is set but validation is disabled."""
        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        cfg_json["dataset"]["path"]["validation"] = self.sample_data_dir
        cfg_json["train"]["perform_validate"] = False
        self.test_cfg_path = Path(self.test_dir, "test_validate_warning.json")

        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        sub_proc = subprocess.run(
            [
                *self._cli_base(),
                f"--config-path={self.test_cfg_path}",
                "train",
                "--no-evaluate",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=self._test_env(),
        )

        self.assertIn(
            "[WARNING] Validation path is provided but perform_validate is set to false",
            sub_proc.stdout + sub_proc.stderr,
        )

    def test_validation_false_no_dataset_provided(self):
        """Test train succeeds without validation when no validation dataset is provided."""
        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        cfg_json["dataset"]["path"]["validation"] = None
        cfg_json["train"]["perform_validate"] = False
        self.test_cfg_path = Path(self.test_dir, "test_no_validate.json")

        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        sub_proc = subprocess.run(
            [
                *self._cli_base(),
                f"--config-path={self.test_cfg_path}",
                "train",
                "--no-evaluate",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=self._test_env(),
        )

        self.assertEqual(sub_proc.returncode, 0)
        self.assertNotIn("Validation:", sub_proc.stdout + sub_proc.stderr)
