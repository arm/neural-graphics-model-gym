# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import shutil
import tempfile
from pathlib import Path

from ng_model_gym.api import do_evaluate, do_training
from ng_model_gym.core.utils.enum_definitions import ProfilerType, TrainEvalMode
from ng_model_gym.core.utils.logging_utils import logging_config
from tests.base_gpu_test import BaseGPUMemoryTest
from tests.testing_utils import create_simple_params


class ApiCoreIntegrationTest(BaseGPUMemoryTest):
    """Integration tests for API functions in ng_model_gym, non usecase specific."""

    def setUp(self):
        """Load a fresh config before each test."""
        super().setUp()
        self.tmp_dir = Path(tempfile.mkdtemp())

        # Create a valid config to use
        self.config = create_simple_params(
            usecase="nss",
            output_dir=self.tmp_dir / "output",
            dataset_path="",
            checkpoints=self.tmp_dir / "checkpoints",
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_logging_config_no_mutation(self):
        """logging_config should not modify the config."""
        before = self.config.model_dump()
        logging_config(self.config, "ng_model_gym", log_level=10)
        after = self.config.model_dump()

        self.assertEqual(before, after, "logging_config mutated the config!")

    def test_resume_finetune_flags_mutually_exclusive(self):
        """Test error raised if both resume and finetune args specified"""
        with self.assertRaises(ValueError):
            do_training(
                self.config,
                TrainEvalMode.FP32,
                ProfilerType.DISABLED,
                finetune_model_path="random_weight.pt",
                resume_model_path="ckpt10.pt",
            )

    def test_training_raises_error_missing_train_dataset_path(self):
        """Training should fail fast when train dataset path is missing."""
        self.config.dataset.path.train = None

        with self.assertRaisesRegex(
            ValueError, "Config error: No path specified for the train dataset path"
        ):
            do_training(
                self.config,
                TrainEvalMode.FP32,
                ProfilerType.DISABLED,
            )

    def test_training_raises_error_missing_validation_dataset_path(self):
        """Training should fail when validation is enabled without validation path."""
        self.config.dataset.path.train = "tests/usecases/nss/datasets/train"
        self.config.train.perform_validate = True
        self.config.dataset.path.validation = None

        with self.assertRaisesRegex(
            ValueError,
            (
                "Config error: Perform validate is enabled and no path specified "
                "for validation dataset"
            ),
        ):
            do_training(
                self.config,
                TrainEvalMode.FP32,
                ProfilerType.DISABLED,
            )

    def test_evaluate_raises_error_missing_test_dataset_path(self):
        """Evaluate should fail fast when test dataset path is missing."""
        self.config.dataset.path.test = None

        with self.assertRaisesRegex(
            ValueError, "Config error: No test dataset path provided for evaluation"
        ):
            do_evaluate(
                self.config,
                model_path="unused.pt",
                model_type=TrainEvalMode.FP32,
                profile_setting=ProfilerType.DISABLED,
            )
