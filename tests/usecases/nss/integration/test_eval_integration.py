# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import json
import re
import subprocess
import unittest
from pathlib import Path

from tests.usecases.nss.integration.base_integration import BaseIntegrationTest

# pylint: disable=duplicate-code


class EvaluationIntegrationTest(BaseIntegrationTest):
    """Tests for NSS Evaluation pipeline."""

    def _extract_metric_value(self, metric, log_line):
        """Helper function to extract metric values from log line."""
        match = re.search(f"{metric}" + r": (\d+\.\d+),", log_line)
        if match:
            ans = float(match.group(1))
            return ans
        return None

    def _read_metric_value(self, metric, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                metric_val = self._extract_metric_value(metric, line)
                if metric_val is not None:
                    return metric_val
        return None

    # pylint: disable=duplicate-code
    def test_train_eval_pipeline(self):
        """E2E test of training a model and evaluating it"""
        sub_proc = subprocess.run(
            [
                "ng-model-gym",
                f"--config-path={self.test_cfg_path}",
                "train",
                "--evaluate",
            ]
        )

        self.assertEqual(sub_proc.returncode, 0)
        self.check_log(
            [
                "Evaluating the trained model...",
                "-------------- Evaluation Complete --------------",
            ]
        )

    def test_train_eval_raises_error_missing_dataset(self):
        """Test train --eval raises error if missing dataset path"""

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
                    "--evaluate",
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

    def test_qat_eval_pipeline(self):
        """E2E test of QAT model and evaluating it"""
        sub_proc = subprocess.run(
            [
                "ng-model-gym",
                f"--config-path={self.test_cfg_path}",
                "qat",
                "--evaluate",
            ]
        )

        self.assertEqual(sub_proc.returncode, 0)
        self.check_log(
            [
                "Evaluating the trained model...",
                "-------------- Evaluation Complete --------------",
            ]
        )

    def test_qat_eval_raises_error_missing_dataset(self):
        """Test qat --eval raises error if missing dataset path"""

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
                    "qat",
                    "--evaluate",
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

    # pylint: enable=duplicate-code

    def test_evaluate_from_checkpoints(self):
        """E2E test of evaluating a previously trained model from checkpoints"""
        model_path = "tests/usecases/nss/weights/nss_v0.1.0_fp32.pt"

        # Update config to enable frame export
        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        # Override to ensure we test frames export
        cfg_json["output"]["export_frame_png"] = True
        cfg_json["dataset"]["path"]["train"] = "tests/usecases/nss/datasets/train"
        cfg_json["dataset"]["path"]["test"] = "tests/usecases/nss/datasets/test"
        self.test_cfg_path = Path(self.test_dir, "test_eval_export_frame.json")

        # pylint: disable=duplicate-code
        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        sub_proc = subprocess.run(
            [
                "ng-model-gym",
                f"--config-path={self.test_cfg_path}",
                "evaluate",
                f"--model-path={model_path}",
                "--model-type=fp32",
            ]
        )
        # pylint: enable=duplicate-code

        self.assertEqual(sub_proc.returncode, 0)
        self.check_log(
            [
                f"Loading model from checkpoint: {model_path}",
                "Evaluating the trained model...",
                "-------------- Evaluation Complete --------------",
            ]
        )

        # Check that results are logged to the correct file.
        expected_results_path = Path(self.model_out_dir, "results.log")
        self.assertTrue(Path(expected_results_path).exists())

        # Check that the logged metric values are reasonably high.
        expected_psnr = 26.4
        expected_tpsnr = 24.4
        expected_recpsnr = 26.3
        expected_ssim = 0.89
        ssim_max = 1.0

        # Test PSNR value.
        psnr = self._read_metric_value("PSNR", expected_results_path)
        self.assertIsNotNone(psnr)
        self.assertGreater(
            psnr, expected_psnr, f"PSNR should be greater than {expected_psnr}"
        )

        # Test tPSNR value.
        tpsnr = self._read_metric_value("tPSNRStreaming", expected_results_path)
        self.assertIsNotNone(tpsnr)
        self.assertGreater(
            tpsnr,
            expected_tpsnr,
            f"tPSNRStreaming should be greater than {expected_tpsnr}",
        )

        # Test recPSNR value.
        recpsnr = self._read_metric_value("recPSNRStreaming", expected_results_path)
        self.assertIsNotNone(recpsnr)
        self.assertGreater(
            recpsnr,
            expected_recpsnr,
            f"recPSNRStreaming should be greater than {expected_recpsnr}",
        )
        # Ensure png directory is created for exporting
        exported_png = Path(self.model_out_dir, "png", "frame_0000_pred.png")
        self.assertTrue(exported_png.exists())

        # Test SSIM value.
        ssim = self._read_metric_value("SSIM", expected_results_path)
        self.assertIsNotNone(ssim)
        self.assertGreater(
            ssim, expected_ssim, f"SSIM should be greater than {expected_ssim}"
        )
        self.assertLessEqual(
            ssim, ssim_max, f"SSIM should be less than or equal to {ssim_max}"
        )

    def test_evaluate_from_checkpoints_qat(self):
        """E2E test of evaluating a previously trained model from checkpoints QAT"""
        model_path = "./tests/usecases/nss/weights/nss_v0.1.1_int8.pt"

        # Update config to use full datasets
        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        cfg_json["dataset"]["path"]["train"] = "tests/usecases/nss/datasets/train"
        cfg_json["dataset"]["path"]["test"] = "tests/usecases/nss/datasets/test"

        # pylint: disable=duplicate-code
        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        sub_proc = subprocess.run(
            [
                "ng-model-gym",
                f"--config-path={self.test_cfg_path}",
                "evaluate",
                f"--model-path={model_path}",
                "--model-type=qat_int8",
            ]
        )

        self.assertEqual(sub_proc.returncode, 0)
        self.check_log(
            [
                "Evaluating the trained model...",
                "-------------- Evaluation Complete --------------",
            ]
        )

        # Check that results are logged to the correct file.
        expected_results_path = Path(self.model_out_dir, "results.log")
        self.assertTrue(Path(expected_results_path).exists())

        # Check that the logged metric values are reasonably high.
        expected_psnr = 26.6
        expected_tpsnr = 24.6
        expected_recpsnr = 26.6
        expected_ssim = 0.89
        ssim_max = 1.0

        # Test PSNR value.
        psnr = self._read_metric_value("PSNR", expected_results_path)
        self.assertIsNotNone(psnr)
        self.assertGreater(
            psnr, expected_psnr, f"PSNR should be greater than {expected_psnr}"
        )

        # Test tPSNR value.
        tpsnr = self._read_metric_value("tPSNRStreaming", expected_results_path)
        self.assertIsNotNone(tpsnr)
        self.assertGreater(
            tpsnr,
            expected_tpsnr,
            f"tPSNRStreaming should be greater than {expected_tpsnr}",
        )

        # Test recPSNR value.
        recpsnr = self._read_metric_value("recPSNRStreaming", expected_results_path)
        self.assertIsNotNone(recpsnr)
        self.assertGreater(
            recpsnr,
            expected_recpsnr,
            f"recPSNRStreaming should be greater than {expected_recpsnr}",
        )

        # Test SSIM value.
        ssim = self._read_metric_value("SSIM", expected_results_path)
        self.assertIsNotNone(ssim)
        self.assertGreater(
            ssim, expected_ssim, f"SSIM should be greater than {expected_ssim}"
        )
        self.assertLessEqual(
            ssim, ssim_max, f"SSIM should be less than or equal to {ssim_max}"
        )

    def test_eval_command_raises_error_missing_dataset(self):
        """Test eval cli raises error if missing dataset path"""

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
                    "evaluate",
                    "--model-path=tests/usecases/nss/weights/nss_v0.1.0_fp32.pt",
                    "--model-type=fp32",
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

    # pylint: disable=duplicate-code
    def test_validation(self):
        """E2E test of evaluating on a validation set after each training epoch"""
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
                "--evaluate",
            ]
        )

        self.assertEqual(sub_proc.returncode, 0)

    def test_validation_validate_frequency_array(self):
        """E2E test of evaluating on a validation set after specific training epochs"""
        # Update config to enable validation
        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        # Override number_of_epochs to test resuming
        cfg_json["dataset"]["path"]["validation"] = "tests/usecases/nss/datasets/val"
        cfg_json["train"]["perform_validate"] = True

        # Set specific epochs to perform validation after
        cfg_json["train"]["fp32"]["number_of_epochs"] = 4
        cfg_json["train"]["validate_frequency"] = [1, 3, 4]
        self.test_cfg_path = Path(self.test_dir, "test_validate.json")

        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        sub_proc = subprocess.run(
            [
                "ng-model-gym",
                f"--config-path={self.test_cfg_path}",
                "train",
                "--evaluate",
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(sub_proc.returncode, 0)

        # Ensure Validation only runs for specific epochs
        self.assertIn("Validation: Epoch 1/4", sub_proc.stderr)
        self.assertNotIn("Validation: Epoch 2/4", sub_proc.stderr)
        self.assertIn("Validation: Epoch 3/4", sub_proc.stderr)
        self.assertIn("Validation: Epoch 4/4", sub_proc.stderr)

    def test_train_command_with_validation_raises_error_missing_dataset(self):
        """Test train with validation raises error if missing dataset path"""

        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        # Override validation dataset path and perform_validate
        cfg_json["dataset"]["path"]["validation"] = None
        cfg_json["train"]["perform_validate"] = True

        self.test_cfg_path = Path(self.test_dir, "test_empty_datasets_path.json")
        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        with self.assertRaises(subprocess.CalledProcessError) as sub_proc_out:
            subprocess.run(
                [
                    "ng-model-gym",
                    f"--config-path={self.test_cfg_path}",
                    "train",
                    "--evaluate",
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

    # pylint: enable=duplicate-code

    def test_trace_profiler(self):
        """Test JSON trace is generated with profiler=trace flag"""
        self.run_model_profiler("eval")

    def test_cuda_profiler(self):
        """Test trace is generated with profiler=gpu_memory flag"""
        self.run_cuda_profiler_test("eval")


if __name__ == "__main__":
    unittest.main()
