# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path

from ng_model_gym.api import do_evaluate, do_export, do_training
from ng_model_gym.utils.config_utils import load_config_file
from ng_model_gym.utils.logging import logging_config
from ng_model_gym.utils.types import ExportType, ProfilerType, TrainEvalMode
from tests.nss.integration.base_integration import BaseIntegrationTest


class ApiIntegrationTest(BaseIntegrationTest):
    """Integration tests for API functions in ng_model_gym."""

    def setUp(self):
        """Load a fresh config before each test."""
        super().setUp()
        # load config from the BaseIntegrationTest-provided path
        self.config = load_config_file(Path(self.test_cfg_path))

    def test_logging_config_no_mutation(self):
        """logging_config should not modify the config."""
        before = (
            self.config.dict()
            if hasattr(self.config, "dict")
            else self.config.model_dump()
        )
        logging_config(self.config, "ng_model_gym", log_level=10)
        after = (
            self.config.dict()
            if hasattr(self.config, "dict")
            else self.config.model_dump()
        )
        self.assertEqual(before, after, "logging_config mutated the config!")

    def test_do_training_no_mutation(self):
        """do_training should not modify the config and return a model."""
        self.config.model_train_eval_mode = TrainEvalMode.FP32
        before = (
            self.config.dict()
            if hasattr(self.config, "dict")
            else self.config.model_dump()
        )
        model = do_training(self.config, TrainEvalMode.FP32, ProfilerType.DISABLED)
        after = (
            self.config.dict()
            if hasattr(self.config, "dict")
            else self.config.model_dump()
        )
        self.assertEqual(before, after, "do_training mutated the config!")
        self.assertIsNotNone(model, "do_training did not return a model")

    def test_do_evaluate_no_mutation(self):
        """do_evaluate should not modify the config."""
        # Load model from .pt file first
        self.config.model_train_eval_mode = TrainEvalMode.FP32
        model_file = Path("tests/nss/weights/nss_v0.1.0_fp32.pt")
        before = (
            self.config.dict()
            if hasattr(self.config, "dict")
            else self.config.model_dump()
        )
        do_evaluate(self.config, model_file, TrainEvalMode.FP32, ProfilerType.DISABLED)
        after = (
            self.config.dict()
            if hasattr(self.config, "dict")
            else self.config.model_dump()
        )
        self.assertEqual(before, after, "do_evaluate mutated the config!")

    def test_do_export_no_mutation_and_outputs(self):
        """do_export should not modify the config and produce output files."""
        # Train a model to generate a checkpoint to read from in export
        do_training(self.config, TrainEvalMode.FP32, ProfilerType.DISABLED)
        # Set an export directory under BaseIntegrationTest's test_dir
        export_dir = Path(self.test_dir) / "export" / "vgf"
        model_file = Path("tests/nss/weights/nss_v0.1.0_fp32.pt")
        self.config.output.export.vgf_output_dir = export_dir
        before = (
            self.config.dict()
            if hasattr(self.config, "dict")
            else self.config.model_dump()
        )
        do_export(self.config, model_file, export_type=ExportType.FP32)
        after = (
            self.config.dict()
            if hasattr(self.config, "dict")
            else self.config.model_dump()
        )
        self.assertEqual(before, after, "do_export mutated the config!")
        # verify at least one file was created in export_dir
        files = list(export_dir.rglob("*"))
        self.assertTrue(
            any(f.is_file() for f in files), "do_export did not produce any files"
        )


if __name__ == "__main__":
    unittest.main()
