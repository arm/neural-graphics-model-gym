# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from ng_model_gym.core.data.utils import DataLoaderMode
from ng_model_gym.core.evaluator.evaluator import NGModelEvaluator
from ng_model_gym.core.model.recurrent_model import FeedbackModel
from tests.testing_utils import create_simple_params


class TestNGModelEvaluator(unittest.TestCase):
    """Tests for NGModelEvaluator."""

    def setUp(self):
        """Setup common test data and state"""
        self.test_dir = tempfile.mkdtemp()

        train_data_dir = Path("./tests/usecases/nss/datasets/train")
        val_data_dir = Path("./tests/usecases/nss/datasets/val")
        test_data_dir = Path("./tests/usecases/nss/datasets/test")

        # Create a temporary output directory
        output_dir = Path(self.test_dir, "output")
        output_dir.mkdir()

        # Create config model using our own data
        self.params = create_simple_params()
        self.params.dataset.path.train = train_data_dir
        self.params.dataset.path.validation = val_data_dir
        self.params.dataset.path.test = test_data_dir

        self.model = Mock(spec=FeedbackModel)
        self.model.device = torch.device("cpu")

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.test_dir)

    @patch("ng_model_gym.core.evaluator.evaluator.get_dataloader")
    def test_prepare_datasets_evaluation(self, get_dataloader):
        """
        Test that prepare_datasets() calls get_dataloader() with the correct loader mode
        for evaluation
        """
        model_evaluator = NGModelEvaluator(self.model, self.params)
        model_evaluator.prepare_datasets()

        # Assert that get_dataloader() has been called once in evaluation mode
        get_dataloader.assert_called_once()
        get_dataloader.assert_called_with(
            self.params,
            num_workers=self.params.dataset.num_workers,
            prefetch_factor=self.params.dataset.prefetch_factor,
            loader_mode=DataLoaderMode.TEST,
        )

        # Assert that the returned dataloader object exists
        self.assertIsNotNone(model_evaluator.dataloader)

    @patch("ng_model_gym.core.evaluator.evaluator.get_dataloader")
    def test_prepare_datasets_validation(self, get_dataloader):
        """
        Test that prepare_datasets() calls get_dataloader() with the correct loader mode
        """
        # Change params to validation settings
        self.params.dataset.path.validation = self.params.dataset.path.train
        self.params.train.perform_validate = True

        model_evaluator = NGModelEvaluator(self.model, self.params)
        model_evaluator.prepare_datasets()

        # Assert that get_dataloader() has been called once in validation mode
        get_dataloader.assert_called_once()
        get_dataloader.assert_called_with(
            self.params,
            num_workers=self.params.dataset.num_workers,
            prefetch_factor=self.params.dataset.prefetch_factor,
            loader_mode=DataLoaderMode.TEST,
        )

        # Assert that the dataloader object exists
        self.assertIsNotNone(model_evaluator.dataloader)

    def test_get_results(self):
        """Test that get_results() returns results for PSNR, tPSNR, recPSNR and SSIM"""
        model_evaluator = NGModelEvaluator(self.model, self.params)

        # Pre-load metrics with random values
        for metric in model_evaluator.metrics:
            pred = torch.rand((1, 1, 3, 256, 256))
            target = torch.rand((1, 1, 3, 256, 256))
            if metric.is_streaming:
                metric.update(pred, target, seq_id=1)
            else:
                metric.update(pred, target)

        # Call get_results()
        results = model_evaluator.get_results()

        self.assertEqual(len(results), 4)
        self.assertIn("PSNR", results.keys())
        self.assertIn("tPSNR", results.keys())
        self.assertIn("recPSNR", results.keys())
        self.assertIn("SSIM", results.keys())

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
        model_evaluator = NGModelEvaluator(self.model, self.params)

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

        model_evaluator = NGModelEvaluator(self.model, self.params)

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
