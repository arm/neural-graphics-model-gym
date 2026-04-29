# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import tempfile
import unittest
from pathlib import Path

from ng_model_gym.api import do_evaluate, do_export, do_training
from ng_model_gym.core.config.config_utils import load_config_file
from ng_model_gym.core.utils.enum_definitions import (
    ExportType,
    ProfilerType,
    TrainEvalMode,
)
from tests.usecases.nfru.integration.base_integration import NFRUBaseIntegrationTest

# pylint: disable=duplicate-code


def _assert_one_or_more_files_at_path(path_name, message):
    """
    Asserts that at least one "regular" file is present in "path_name".
    If not, an assertion failure occurs using "message".
    """
    files = list(path_name.rglob("*"))

    assert any(f.is_file() for f in files), message


@unittest.skip("NFRU CI/assets disabled for now")
class ApiIntegrationTest(NFRUBaseIntegrationTest):
    """NFRU specific integration tests for API functions in ng_model_gym."""

    def setUp(self):
        """Load a fresh config before each test."""
        super().setUp()
        # load config from the NFRUBaseIntegrationTest-provided path
        self.config = load_config_file(Path(self.test_cfg_path))

        self.MODEL_FILE = Path("tests/usecases/nfru/weights/nfru_v1_fp32.pt")

    def _get_output_root(self):
        """Gets a directory for exports, without checking that it exists"""
        repo_root = Path(__file__).resolve().parents[4]
        return repo_root / "output"

    def test_do_training_no_mutation(self):
        """do_training should not modify the config and return a model."""
        self.config.model_train_eval_mode = TrainEvalMode.FP32

        before = self.config.model_dump()
        model = do_training(self.config, TrainEvalMode.FP32, ProfilerType.DISABLED)
        after = self.config.model_dump()

        self.assertEqual(before, after, "do_training mutated the config")
        self.assertIsNotNone(model, "do_training did not return a model")

    def test_do_evaluate_no_mutation(self):
        """do_evaluate should not modify the config."""
        # Load model from .pt file first
        self.config.model_train_eval_mode = TrainEvalMode.FP32

        before = self.config.model_dump()
        do_evaluate(
            self.config, self.MODEL_FILE, TrainEvalMode.FP32, ProfilerType.DISABLED
        )
        after = self.config.model_dump()

        self.assertEqual(before, after, "do_evaluate mutated the config")

    def _assert_do_export_no_mutation_and_outputs(self):
        """
        Check that do_export() doesn't change its inputs and does create an
        output file. Configure self.config as necessary before calling.
        """
        with tempfile.TemporaryDirectory(dir=self._get_output_root()) as tmp_dir:
            export_dir = Path(tmp_dir) / "export" / "vgf"
            self.config.output.export.vgf_output_dir = export_dir

            before = self.config.model_dump()
            do_export(self.config, self.MODEL_FILE, export_type=ExportType.FP32)
            after = self.config.model_dump()

            self.assertEqual(before, after, "do_export mutated the config")

            _assert_one_or_more_files_at_path(
                export_dir, "do_export did not produce any files"
            )

    def test_static_do_export_no_mutation_and_outputs(self):
        """
        Test static shape export. do_export() should not modify the config and
        should produce output files.
        """
        self.config.output.export.dynamic_shape = False
        self.config.output.export.vgf_static_input_shape = [[1, 16, 272, 480]]

        self._assert_do_export_no_mutation_and_outputs()

    def test_dynamic_do_export_no_mutation_and_outputs(self):
        """
        Test dynamic shape export, similarly to the above "static" test.
        Note that the dataset inherently has height 270 and this should be
        determined automatically. (For dynamic export specifically, the input
        height must be a multiple of 2, which is the case here.)
        """
        self.config.output.export.dynamic_shape = True

        self._assert_do_export_no_mutation_and_outputs()
