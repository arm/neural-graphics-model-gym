# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

from ng_model_gym.core.model.recurrent_model import FeedbackModel
from ng_model_gym.usecases.nss.nss_evaluator import ModelEvaluator
from tests.testing_utils import create_simple_params


class NSSModelEvaluatorTest(unittest.TestCase):
    """Tests for ModelEvaluator."""

    def setUp(self):
        """Setup common test data and state"""
        self.test_dir = tempfile.mkdtemp()
        data_dir = Path("./tests/usecases/nss/datasets/test")
        # Create a temporary output directory
        output_dir = Path(self.test_dir, "output")
        output_dir.mkdir()

        # Create config model using our own data
        self.params = create_simple_params()
        self.params.dataset.path.train = data_dir
        self.params.dataset.path.test = data_dir

        self.model = Mock(spec=FeedbackModel)
        self.model.device = torch.device("cpu")

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_run_model(self):
        """Check that _run_model() calls forward pass from recurrent_model once"""
        # Mock the dataloader to yield a single example batch
        sample_input = torch.rand((1, 1, 3, 256, 256))
        sample_target = torch.rand((1, 1, 3, 256, 256))
        import ng_model_gym.core.evaluator.evaluator as evaluator_module  # pylint: disable=import-outside-toplevel

        evaluator_module.get_dataloader = lambda *args, **kwargs: [
            (sample_input, sample_target)
        ]
        # Stub the model to return a dict so it's subscriptable
        self.model.return_value = {"output": sample_input}
        model_evaluator = ModelEvaluator(self.model, self.params)

        model_evaluator._run_model()
        # Ensure the model was called exactly once
        self.assertEqual(self.model.call_count, 1)

    def test_evaluate(self):
        """Test that evaluate() calls the necessary functions and outputs a JSON file"""
        import ng_model_gym.core.evaluator.evaluator as evaluator_module  # pylint: disable=import-outside-toplevel

        evaluator_module.get_dataloader = lambda *args, **kwargs: []
        # Stub the model to return a dict with the expected output key
        predicted = torch.rand((1, 32, 3, 256, 256))
        self.model.return_value = {"output": predicted}

        model_evaluator = ModelEvaluator(self.model, self.params)

        existing_results = set(Path(self.params.output.dir).glob("eval_metrics_*.json"))

        model_evaluator.evaluate()

        # Check that results.log and metrics json exist
        expected_results_logfile = Path(self.params.output.dir, "results.log")
        self.assertTrue(expected_results_logfile.exists())

        all_results_json = set(Path(self.params.output.dir).glob("eval_metrics_*.json"))
        expected_results_json = all_results_json - existing_results
        self.assertTrue(expected_results_json)


if __name__ == "__main__":
    unittest.main()
