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
from ng_model_gym.core.evaluator.evaluator import BaseModelEvaluator
from ng_model_gym.usecases.nss.model.recurrent_model import FeedbackModel
from tests.testing_utils import create_simple_params


class BaseModelEvaluatorTest(unittest.TestCase):
    """Tests for BaseModelEvaluator."""

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
        model_evaluator = BaseModelEvaluator(self.model, self.params)
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

        model_evaluator = BaseModelEvaluator(self.model, self.params)
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

    def test_run_model(self):
        """
        Test that _run_model is unimplemented for BaseModelEvaluator.
        It should be implemented by sub-classes.
        """
        model_evaluator = BaseModelEvaluator(self.model, self.params)

        with self.assertRaises(NotImplementedError):
            model_evaluator._run_model()

    def test_get_results(self):
        """Test that get_results() returns results for PSNR, tPSNR, recPSNR and SSIM"""
        model_evaluator = BaseModelEvaluator(self.model, self.params)

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


if __name__ == "__main__":
    unittest.main()
