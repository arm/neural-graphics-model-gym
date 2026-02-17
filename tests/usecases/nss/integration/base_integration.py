# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import json
import os
import platform
import shutil
import subprocess
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import List

from ng_model_gym.core.utils.checkpoint_utils import latest_checkpoint_in_dir
from ng_model_gym.core.utils.general_utils import create_directory
from tests.base_gpu_test import BaseGPUMemoryTest
from tests.testing_utils import clear_loggers


class BaseIntegrationTest(BaseGPUMemoryTest):
    """Base class for Integration Tests for NSS training pipeline."""

    def setUp(self) -> None:
        """Create tmp test_default.json with overridden params."""
        super().setUp()
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.test_dir, "test_checkpoints")
        create_directory(self.checkpoint_dir)
        self.qat_checkpoint_dir = self.checkpoint_dir / "qat_checkpoints"
        create_directory(self.qat_checkpoint_dir)
        self.model_out_dir = Path(self.test_dir, "model_output")
        create_directory(self.model_out_dir)

        if os.getenv("FAST_TEST") == "1":
            self.train_data_dir = "tests/usecases/nss/mini_datasets/train"
            self.test_data_dir = "tests/usecases/nss/mini_datasets/test"
        else:
            self.train_data_dir = "tests/usecases/nss/datasets/train"
            self.test_data_dir = "tests/usecases/nss/datasets/test"

        self.finetune_weights = "tests/usecases/nss/weights/nss_v0.1.0_fp32.pt"
        self.tensorboard_dir = Path(self.test_dir, "tensorboard-logs")
        create_directory(self.tensorboard_dir)
        config_path = files("ng_model_gym.usecases.nss.configs") / "default.json"
        with open(config_path, encoding="utf-8") as f:
            self.cfg_json = json.load(f)

        # Override default params for the test
        num_workers = 0 if platform.system() == "Windows" else 1
        self.cfg_json["dataset"]["num_workers"] = num_workers
        self.cfg_json["dataset"]["prefetch_factor"] = 1
        self.cfg_json["train"]["batch_size"] = 4
        self.cfg_json["train"]["fp32"]["number_of_epochs"] = 1
        self.cfg_json["train"]["qat"]["number_of_epochs"] = 1
        self.cfg_json["train"]["fp32"]["checkpoints"]["dir"] = str(self.checkpoint_dir)
        self.cfg_json["train"]["qat"]["checkpoints"]["dir"] = str(
            self.qat_checkpoint_dir
        )
        self.cfg_json["train"]["perform_validate"] = False
        self.cfg_json["output"]["dir"] = str(self.model_out_dir)
        # integration-test dataset
        self.cfg_json["dataset"]["path"]["train"] = self.train_data_dir
        self.cfg_json["dataset"]["path"]["test"] = self.test_data_dir
        self.cfg_json["output"][
            "export_frame_png"
        ] = False  # Speed up integration tests

        self.cfg_json["output"]["tensorboard_output_dir"] = str(self.tensorboard_dir)
        self.test_cfg_path = Path(self.test_dir, "test_default.json")

        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(self.cfg_json, f)

    def tearDown(self):
        """Clean up the temporary directory."""
        super().tearDown()

        clear_loggers()
        shutil.rmtree(self.test_dir)

    def check_log(self, msgs: List):
        """Verify that the log file contents matches the log."""
        log_file_path = Path(self.model_out_dir, "output.log")

        text = log_file_path.read_text(encoding="utf-8")
        # Replace new line and path characters for Windows
        text = text.replace("\\", "/")

        for msg in msgs:
            # Replace new line and path characters for Windows
            m = msg.replace("\\", "/")
            if m not in text and Path(m).name not in text:
                self.fail(f"Message not found in log: {msg}")

    def run_training_test(self):
        """E2E test of the model training components."""
        sub_proc = subprocess.run(
            [
                "ng-model-gym",
                f"--config-path={self.test_cfg_path}",
                "train",
                "--no-evaluate",
            ],
            capture_output=True,
            text=True,
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
                "ng-model-gym",
                f"--config-path={self.test_cfg_path}",
                "qat",
                "--no-evaluate",
            ]
        )
        self.assertEqual(sub_proc.returncode, 0)

        with self.subTest("TensorBoard logs"):
            files_exist = any(self.tensorboard_dir.iterdir())
            self.assertTrue(files_exist, "TensorBoard logs not found.")

    def run_model_profiler(self, mode):
        """Test JSON trace is generated with profiler=trace flag"""
        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        # Override default params for the test
        cfg_json["train"]["batch_size"] = 4
        cfg_json["train"]["fp32"]["number_of_epochs"] = 1
        cfg_json["dataset"]["recurrent_samples"] = 2
        cfg_json["dataset"]["path"]["train"] = "tests/usecases/nss/datasets/train"
        cfg_json["dataset"]["path"]["test"] = "tests/usecases/nss/datasets/test"

        # pylint: disable=duplicate-code
        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        if mode == "eval":
            sub_proc = subprocess.run(
                [
                    "ng-model-gym",
                    f"--config-path={self.test_cfg_path}",
                    "--profiler=trace",
                    "evaluate",
                    "--model-path=tests/usecases/nss/weights/nss_v0.1.0_fp32.pt",
                    "--model-type=fp32",
                ]
            )
            self.assertEqual(sub_proc.returncode, 0)

            # For eval, trace is saved in the output dir set in the config
            trace_save_dir = self.model_out_dir
        else:
            sub_proc = subprocess.run(
                [
                    "ng-model-gym",
                    f"--config-path={self.test_cfg_path}",
                    "--profiler=trace",
                    mode,
                    "--no-evaluate",
                ]
            )
            self.assertEqual(sub_proc.returncode, 0)

            # For train/qat, trace is saved in its checkpoint dir
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

        self.assertTrue(trace_file_exists)

    def run_cuda_profiler_test(self, mode):
        """Test JSON trace is generated with profiler=gpu_memory flag"""
        self.assertIn(mode, ("train", "qat", "eval"))

        with open(self.test_cfg_path, encoding="utf-8") as f:
            cfg_json = json.load(f)

        # Override default params for the test
        cfg_json["train"]["batch_size"] = 4
        cfg_json["train"]["qat"]["number_of_epochs"] = 1
        cfg_json["dataset"]["recurrent_samples"] = 4

        with open(self.test_cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        if mode == "eval":
            sub_proc = subprocess.run(
                [
                    "ng-model-gym",
                    f"--config-path={self.test_cfg_path}",
                    "--profiler=gpu_memory",
                    "evaluate",
                    "--model-path=tests/usecases/nss/weights/nss_v0.1.0_fp32.pt",
                    "--model-type=fp32",
                ]
            )
            self.assertEqual(sub_proc.returncode, 0)

            # For eval, trace is saved in the output dir set in the config
            trace_save_dir = self.model_out_dir
        else:
            sub_proc = subprocess.run(
                [
                    "ng-model-gym",
                    f"--config-path={self.test_cfg_path}",
                    "--profiler=gpu_memory",
                    mode,
                    "--no-evaluate",
                ]
            )
            self.assertEqual(sub_proc.returncode, 0)

            # For train/qat, trace is saved in its checkpoint dir
            dir_to_search = (
                self.checkpoint_dir if mode == "train" else self.qat_checkpoint_dir
            )
            trace_save_dir = latest_checkpoint_in_dir(Path(dir_to_search)).parent

        self.check_log("CUDA memory profiler is enabled")

        profile_file_exists = False
        for item in trace_save_dir.iterdir():
            if (
                "cuda_profiler" in item.name
                and item.suffix == ".pickle"
                and item.stat().st_size > 0
            ):
                profile_file_exists = True

        self.assertTrue(profile_file_exists)

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
                "ng-model-gym",
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
        )
        self.assertEqual(sub_proc.returncode, 0)
        self.check_log(["Restoring training from checkpoint"])
        return sub_proc
