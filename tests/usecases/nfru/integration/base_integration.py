# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import List

import torch

from ng_model_gym.core.model.checkpoint_loader import latest_checkpoint_in_dir
from ng_model_gym.core.utils.io.file_utils import create_directory
from tests.base_gpu_test import BaseGPUMemoryTest
from tests.testing_utils import clear_loggers

# pylint: disable=duplicate-code


class NFRUBaseIntegrationTest(BaseGPUMemoryTest):
    """Base class for NFRU integration tests."""

    SSIM_MAX = 1.0

    def setUp(self) -> None:
        """Create a temporary config file with NFRU test-specific overrides."""
        super().setUp()
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.test_dir, "test_checkpoints")
        create_directory(self.checkpoint_dir)
        self.qat_checkpoint_dir = self.checkpoint_dir / "qat_checkpoints"
        create_directory(self.qat_checkpoint_dir)
        self.model_out_dir = Path(self.test_dir, "model_output")
        create_directory(self.model_out_dir)
        self.tensorboard_dir = Path(self.test_dir, "tensorboard-logs")
        create_directory(self.tensorboard_dir)

        self.sample_data_dir = "tests/usecases/nfru/data/nfru_sample"
        # TODO Add mini dataset
        self.mini_dataset_dir = self.sample_data_dir

        if os.getenv("FAST_TEST") == "1":
            self.train_data_dir = self.mini_dataset_dir
            self.test_data_dir = self.mini_dataset_dir
        else:
            self.train_data_dir = self.sample_data_dir
            self.test_data_dir = self.sample_data_dir

        self.eval_weights = "tests/usecases/nfru/weights/nfru_v1_fp32.pt"
        self.qat_eval_weights = "tests/usecases/nfru/weights/nfru_v1_int8.pt"

        config_path = files("ng_model_gym.usecases.nfru.configs") / "nfru_template.json"
        with open(config_path, encoding="utf-8") as f:
            self.cfg_json = json.load(f)

        num_workers = 0 if platform.system() == "Windows" else 1
        self.cfg_json["dataset"]["num_workers"] = num_workers
        self.cfg_json["dataset"]["prefetch_factor"] = 1
        self.cfg_json["dataset"]["gt_augmentation"] = False

        self.cfg_json["train"]["batch_size"] = 2
        self.cfg_json["train"]["fp32"]["number_of_epochs"] = 1
        self.cfg_json["train"]["qat"]["number_of_epochs"] = 1
        self.cfg_json["train"]["perform_validate"] = False
        self.cfg_json["train"]["fp32"]["checkpoints"]["dir"] = str(self.checkpoint_dir)
        self.cfg_json["train"]["qat"]["checkpoints"]["dir"] = str(
            self.qat_checkpoint_dir
        )

        self.cfg_json["output"]["dir"] = str(self.model_out_dir)
        self.cfg_json["output"]["export_frame_png"] = False
        self.cfg_json["output"]["tensorboard_output_dir"] = str(self.tensorboard_dir)

        self.cfg_json["dataset"]["path"]["train"] = self.train_data_dir
        self.cfg_json["dataset"]["path"]["test"] = self.test_data_dir
        self.cfg_json["dataset"]["path"]["validation"] = ""

        self.cfg_json["output"][
            "export_frame_png"
        ] = False  # Speed up integration tests

        self.test_cfg_path = Path(self.test_dir, "test_nfru_template.json")

        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(self.cfg_json, f)

    def tearDown(self):
        """Clean up temporary directories and loggers."""
        super().tearDown()
        clear_loggers()
        shutil.rmtree(self.test_dir)

    @staticmethod
    def _cli_base():
        """Use module invocation for portability across local and CI setups."""
        return [sys.executable, "-m", "ng_model_gym.cli"]

    @staticmethod
    def _test_env():
        """Environment for subprocess calls to ensure src/ imports resolve consistently."""
        return {"PYTHONPATH": "src", **os.environ}

    def check_log(self, msgs: List[str]):
        """Verify that expected messages are present in output.log."""
        log_file_path = Path(self.model_out_dir, "output.log")
        text = log_file_path.read_text(encoding="utf-8").replace("\\", "/")

        for msg in msgs:
            normalized_msg = msg.replace("\\", "/")
            if normalized_msg not in text and Path(normalized_msg).name not in text:
                self.fail(f"Message not found in log: {msg}")

    @staticmethod
    def _extract_metric_value(metric: str, log_line: str):
        """Extract a metric value from a single metric line."""
        match = re.search(f"{metric}" + r": (\d+\.\d+),", log_line)
        if match:
            return float(match.group(1))
        return None

    def _read_metric_value(self, metric: str, file_path: Path):
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                metric_val = self._extract_metric_value(metric, line)
                if metric_val is not None:
                    return metric_val
        return None

    def _evaluate_from_checkpoints(
        self,
        model_path: str,
        model_type: str = "fp32",
        expected_psnr: float = 29.0,
        expected_ssim: float = 0.93,
        expected_stlpips_max: float = 0.13,
    ):
        """Evaluate a local checkpoint and assert metrics and png export."""
        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        cfg_json["output"]["export_frame_png"] = True
        cfg_json["dataset"]["path"]["train"] = self.train_data_dir
        cfg_json["dataset"]["path"]["test"] = self.test_data_dir
        self.test_cfg_path = Path(self.test_dir, "test_eval_export_frame.json")
        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        sub_proc = subprocess.run(
            [
                *self._cli_base(),
                f"--config-path={self.test_cfg_path}",
                "evaluate",
                f"--model-path={model_path}",
                f"--model-type={model_type}",
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

        expected_results_path = Path(self.model_out_dir, "results.log")
        self.assertTrue(expected_results_path.exists())

        psnr = self._read_metric_value("PSNR", expected_results_path)
        self.assertIsNotNone(psnr)
        self.assertGreater(
            psnr, expected_psnr, f"PSNR should be greater than {expected_psnr}"
        )

        ssim = self._read_metric_value("SSIM", expected_results_path)
        self.assertIsNotNone(ssim)
        self.assertGreater(
            ssim, expected_ssim, f"SSIM should be greater than {expected_ssim}"
        )
        self.assertLessEqual(
            ssim,
            self.SSIM_MAX,
            f"SSIM should be less than or equal to {self.SSIM_MAX}",
        )

        stlpips = self._read_metric_value("STLPIPS", expected_results_path)
        self.assertIsNotNone(stlpips)
        self.assertLess(
            stlpips,
            expected_stlpips_max,
            f"STLPIPS should be less than {expected_stlpips_max}",
        )

        exported_png = Path(
            self.model_out_dir, "png", "predicted", "frame_0000_pred.png"
        )
        self.assertTrue(exported_png.exists())
        exported_png = Path(
            self.model_out_dir, "png", "ground_truth", "frame_0000_gt.png"
        )
        self.assertTrue(exported_png.exists())

    def run_model_profiler(self, mode: str):
        """Test that profiler=trace emits a trace file."""
        self.assertIn(mode, ("train", "qat", "eval"))

        if mode == "eval":
            sub_proc = subprocess.run(
                [
                    *self._cli_base(),
                    f"--config-path={self.test_cfg_path}",
                    "--profiler=trace",
                    "evaluate",
                    f"--model-path={self.eval_weights}",
                    "--model-type=fp32",
                ],
                env={"PYTHONPATH": "src", **os.environ},
            )
            self.assertEqual(sub_proc.returncode, 0)
            trace_save_dir = self.model_out_dir
        else:
            sub_proc = subprocess.run(
                [
                    *self._cli_base(),
                    f"--config-path={self.test_cfg_path}",
                    "--profiler=trace",
                    mode,
                    "--no-evaluate",
                ],
                env={"PYTHONPATH": "src", **os.environ},
            )
            self.assertEqual(sub_proc.returncode, 0)
            dir_to_search = (
                self.checkpoint_dir if mode == "train" else self.qat_checkpoint_dir
            )
            trace_save_dir = latest_checkpoint_in_dir(Path(dir_to_search)).parent

        trace_file_exists = False
        for item in trace_save_dir.iterdir():
            if (
                item.suffix == ".json"
                and item.stat().st_size > 0
                and "trace" in item.name
            ):
                trace_file_exists = True
                break
        self.assertTrue(trace_file_exists)

    def run_cuda_profiler(self, mode: str):
        """Test that profiler=gpu_memory emits a pickle snapshot."""
        self.assertIn(mode, ("train", "qat", "eval"))
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable for gpu_memory profiler test")

        if mode == "eval":
            sub_proc = subprocess.run(
                [
                    *self._cli_base(),
                    f"--config-path={self.test_cfg_path}",
                    "--profiler=gpu_memory",
                    "evaluate",
                    f"--model-path={self.eval_weights}",
                    "--model-type=fp32",
                ],
                env={"PYTHONPATH": "src", **os.environ},
            )
            self.assertEqual(sub_proc.returncode, 0)
            trace_save_dir = self.model_out_dir
        else:
            sub_proc = subprocess.run(
                [
                    *self._cli_base(),
                    f"--config-path={self.test_cfg_path}",
                    "--profiler=gpu_memory",
                    mode,
                    "--no-evaluate",
                ],
                env={"PYTHONPATH": "src", **os.environ},
            )
            self.assertEqual(sub_proc.returncode, 0)
            dir_to_search = (
                self.checkpoint_dir if mode == "train" else self.qat_checkpoint_dir
            )
            trace_save_dir = latest_checkpoint_in_dir(Path(dir_to_search)).parent

        self.check_log(["CUDA memory profiler is enabled"])

        profile_file_exists = False
        for item in trace_save_dir.iterdir():
            if (
                "cuda_profiler" in item.name
                and item.suffix == ".pickle"
                and item.stat().st_size > 0
            ):
                profile_file_exists = True
                break

        self.assertTrue(profile_file_exists)

    def run_training_test(self):
        """E2E test of the model training components."""
        sub_proc = subprocess.run(
            [
                *self._cli_base(),
                f"--config-path={self.test_cfg_path}",
                "train",
                "--no-evaluate",
            ],
            capture_output=True,
            text=True,
            env=self._test_env(),
        )
        self.assertEqual(sub_proc.returncode, 0)

        with self.subTest("TensorBoard logs"):
            files_exist = any(self.tensorboard_dir.iterdir())
            self.assertTrue(files_exist, "TensorBoard logs not found.")

        return sub_proc

    def run_training_test_qat(self):
        """E2E test of the model training components for QAT."""
        sub_proc = subprocess.run(
            [
                *self._cli_base(),
                f"--config-path={self.test_cfg_path}",
                "qat",
                "--no-evaluate",
            ],
            env=self._test_env(),
        )
        self.assertEqual(sub_proc.returncode, 0)

        with self.subTest("TensorBoard logs"):
            files_exist = any(self.tensorboard_dir.iterdir())
            self.assertTrue(files_exist, "TensorBoard logs not found.")

    def run_resume_training_test(self, mode="train", num_epochs=2):
        """E2E test of the model resume training."""
        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        training_mode = "fp32" if mode == "train" else "qat"

        # Override number_of_epochs to test resuming
        cfg_json["train"][training_mode]["number_of_epochs"] = num_epochs
        self.test_cfg_path = Path(self.test_dir, "test_resume.json")

        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        sub_proc = subprocess.run(
            [
                *self._cli_base(),
                f"--config-path={self.test_cfg_path}",
                mode,
                "--no-evaluate",
                "--resume",
                str(
                    self.checkpoint_dir if mode == "train" else self.qat_checkpoint_dir
                ),
            ],
            capture_output=True,
            text=True,
            env=self._test_env(),
        )
        self.assertEqual(sub_proc.returncode, 0)
        self.check_log(["Restoring training from checkpoint"])
        return sub_proc
