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
class EvaluationIntegrationTest(NFRUBaseIntegrationTest):
    """Tests for NFRU evaluation pipeline."""

    def test_train_eval_pipeline(self):
        """E2E test of training a model and evaluating it."""
        sub_proc = subprocess.run(
            [
                *self._cli_base(),
                f"--config-path={self.test_cfg_path}",
                "train",
                "--evaluate",
            ],
            env={"PYTHONPATH": "src", **os.environ},
        )

        self.assertEqual(sub_proc.returncode, 0)
        self.check_log(
            [
                "Evaluating the trained model...",
                "-------------- Evaluation Complete --------------",
            ]
        )

    def test_qat_eval_pipeline(self):
        """E2E test of QAT and evaluation."""
        sub_proc = subprocess.run(
            [
                *self._cli_base(),
                f"--config-path={self.test_cfg_path}",
                "qat",
                "--evaluate",
            ],
            env={"PYTHONPATH": "src", **os.environ},
        )

        self.assertEqual(sub_proc.returncode, 0)
        self.check_log(
            [
                "Evaluating the trained model...",
                "-------------- Evaluation Complete --------------",
            ]
        )

    def test_evaluate_from_checkpoints_qat(self):
        """Evaluate using local QAT checkpoint."""
        self._evaluate_from_checkpoints(self.qat_eval_weights, model_type="qat_int8")

    def test_evaluate_from_checkpoint_local(self):
        """Evaluate using local FP32 model."""
        self._evaluate_from_checkpoints(self.eval_weights, model_type="fp32")

    @unittest.skip(
        "TODO: add stable remote NFRU checkpoint identifier and CI coverage for "
        "evaluate-from-identifier flow."
    )
    def test_evaluate_from_identifier(self):
        """Evaluate using remote model identifier."""
        self._evaluate_from_checkpoints(
            "@neural-framerate-upscaling/nfru_v1_fp32.pt",
            model_type="fp32",
        )

    def test_validation(self):
        """Evaluate on validation set after each training epoch."""
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
                "--evaluate",
            ],
            env={"PYTHONPATH": "src", **os.environ},
        )

        self.assertEqual(sub_proc.returncode, 0)

    def test_validation_validate_frequency_array(self):
        """Validate only on explicitly configured epochs."""
        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        cfg_json["dataset"]["path"]["validation"] = self.sample_data_dir
        cfg_json["train"]["perform_validate"] = True
        cfg_json["train"]["fp32"]["number_of_epochs"] = 4
        cfg_json["train"]["validate_frequency"] = [1, 3, 4]
        self.test_cfg_path = Path(self.test_dir, "test_validate.json")

        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        sub_proc = subprocess.run(
            [
                *self._cli_base(),
                f"--config-path={self.test_cfg_path}",
                "train",
                "--evaluate",
            ],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src", **os.environ},
        )

        self.assertEqual(sub_proc.returncode, 0)
        self.assertIn("Validation: Epoch 1/4", sub_proc.stderr)
        self.assertNotIn("Validation: Epoch 2/4", sub_proc.stderr)
        self.assertIn("Validation: Epoch 3/4", sub_proc.stderr)
        self.assertIn("Validation: Epoch 4/4", sub_proc.stderr)

    @unittest.skip(
        "Issue: trace profiler export for NFRU eval fails on short runs "
        "with AssertionError 'Profiler must be initialized before exporting chrome trace'."
        "NFRU dataset has fewer samples than NSS, so these will be skipped due to the existing "
        "profiler schedule."
    )
    def test_trace_profiler(self):
        """Test JSON trace is generated with profiler=trace."""
        self.run_model_profiler("eval")

    def test_cuda_profiler(self):
        """Test CUDA memory profile snapshot is generated in eval mode."""
        self.run_cuda_profiler("eval")
