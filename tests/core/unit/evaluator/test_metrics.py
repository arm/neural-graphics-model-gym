# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import pyiqa
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

from ng_model_gym.core.evaluator import (
    get_metrics,
    Psnr,
    RecPsnr,
    RecPsnrStreaming,
    Ssim,
    Stlpips,
    TPsnr,
    TPsnrStreaming,
)
from tests.testing_utils import create_simple_params


def generate_mock_data(shape):
    """Mock up some data for tests."""
    return torch.rand(shape), torch.rand(shape)


class TestMetrics(unittest.TestCase):
    """Unit tests for metrics in the file metrics.py."""

    def test_psnr(self):
        """Testing using PeakSignalNoiseRatio from torchmetrics.image"""
        # batch_size, seq_length, number_of_channels, h, w
        preds, target = generate_mock_data((10, 5, 3, 256, 256))

        psnr = Psnr()
        psnr.update(preds, target)
        result = psnr.compute()

        self.assertIsInstance(result, torch.Tensor)
        # Assert that the returned Tensor is a scalar
        self.assertEqual(result.shape, ())

        psnr = peak_signal_noise_ratio(
            preds,
            target,
            data_range=1.0,
            reduction="elementwise_mean",
            dim=(2, 3, 4),
        )

        self.assertAlmostEqual(psnr.numpy(), result.numpy())

    def test_tpsnr(self):
        """Test Temporal PSNR."""
        # batch_size, seq_length, number_of_channels, h, w
        preds, target = generate_mock_data((10, 5, 3, 256, 256))

        tpsnr = TPsnr()
        tpsnr.update(preds, target)
        result = tpsnr.compute()
        tpsnr.reset()

        self.assertIsInstance(result, torch.Tensor)
        # Assert that the returned Tensor is a scalar
        self.assertEqual(result.shape, ())

        tpsnr.update(preds, target)

    def test_rec_psnr(self):
        """Test Recurrent PSNR from torchmetrics.image"""
        # batch_size, seq_length, number_of_channels, h, w
        preds, target = generate_mock_data((10, 5, 3, 256, 256))

        recpsnr = RecPsnr()
        recpsnr.update(preds, target)
        result = recpsnr.compute()
        recpsnr.reset()

        self.assertIsInstance(result, torch.Tensor)
        # Assert that the returned Tensor is a scalar
        self.assertEqual(result.shape, ())

        recpsnr.update(preds, target)

    def test_ssim(self):
        """Test SSIM from torchmetrics.image"""
        # batch_size, seq_length, number_of_channels, h, w
        preds, target = generate_mock_data((10, 5, 3, 256, 256))

        ssim = Ssim()
        ssim.update(preds, target)
        result = ssim.compute()

        self.assertIsInstance(result, torch.Tensor)
        # Assert that the returned Tensor is a scalar
        self.assertEqual(result.shape, ())

        # Reshape from 5 channels to 4
        N, T, C, H, W = preds.shape
        preds_4d = preds.view(N * T, C, H, W)
        target_4d = target.view(N * T, C, H, W)

        ssim = StructuralSimilarityIndexMeasure(
            data_range=1.0, kernel_size=11, sigma=1.5, gaussian_kernel=True
        ).to(preds_4d.device)

        ssim = ssim(preds_4d, target_4d)

        self.assertAlmostEqual(ssim.numpy(), result.numpy())

        # Assert ssim >= 0.0 and <= 1.0
        self.assertGreaterEqual(result.item(), 0.0)
        self.assertLessEqual(result.item(), 1.0)

    def test_ssim_4d(self):
        """Test our SSIM update matches torchmetrics.image SSIM for 4D input."""
        # batch_size, number_of_channels, h, w
        preds, target = generate_mock_data((10, 3, 512, 512))

        ssim = Ssim()
        ssim.update(preds, target)
        result = ssim.compute()

        self.assertIsInstance(result, torch.Tensor)
        # Assert that the returned Tensor is a scalar
        self.assertEqual(result.shape, ())

        ssim = StructuralSimilarityIndexMeasure(
            data_range=1.0, kernel_size=11, sigma=1.5, gaussian_kernel=True
        ).to(preds.device)
        ssim = ssim(preds, target)

        self.assertAlmostEqual(ssim.numpy(), result.numpy())

        # Assert ssim >= 0.0 and <= 1.0
        self.assertGreaterEqual(result.item(), 0.0)
        self.assertLessEqual(result.item(), 1.0)

    def test_stlpips_5d(self):
        """Test STLPIPS from pyiqa for 5D input"""
        # batch_size, seq_length, number_of_channels, h, w
        preds, target = generate_mock_data((10, 5, 3, 256, 256))

        stlpips = Stlpips()
        stlpips.update(preds, target)
        result = stlpips.compute()

        self.assertIsInstance(result, torch.Tensor)
        # Assert that the returned Tensor is a scalar
        self.assertEqual(result.shape, ())

        # Reshape from 5 channels to 4
        N, T, C, H, W = preds.shape
        preds_4d = preds.view(N * T, C, H, W)
        target_4d = target.view(N * T, C, H, W)

        pyiqa_metric = pyiqa.create_metric("stlpips-vgg", device=preds_4d.device)
        pyiqa_result = pyiqa_metric(preds_4d, target_4d).mean()

        self.assertAlmostEqual(pyiqa_result.numpy(), result.numpy())

    def test_stlpips_4d(self):
        """Test STLPIPS from pyiqa for 4D input"""
        # batch_size, number_of_channels, h, w
        preds, target = generate_mock_data((10, 3, 256, 256))

        stlpips = Stlpips()
        stlpips.update(preds, target)
        result = stlpips.compute()

        self.assertIsInstance(result, torch.Tensor)
        # Assert that the returned Tensor is a scalar
        self.assertEqual(result.shape, ())

        pyiqa_metric = pyiqa.create_metric("stlpips-vgg", device=preds.device)
        pyiqa_result = pyiqa_metric(preds, target).mean()

        self.assertAlmostEqual(pyiqa_result.numpy(), result.numpy())

    def test_get_metrics(self):
        """Test that get_metrics returns all our expected metrics."""
        params = create_simple_params()

        metrics = get_metrics(params, is_test=False)
        self.assertEqual(len(metrics), 4)
        for metric in metrics:
            self.assertTrue(callable(metric))
        metric_types = {type(metric) for metric in metrics}
        self.assertIn(TPsnr, metric_types)
        self.assertIn(RecPsnr, metric_types)
        self.assertNotIn(TPsnrStreaming, metric_types)
        self.assertNotIn(RecPsnrStreaming, metric_types)

        metrics = get_metrics(params, is_test=True)
        self.assertEqual(len(metrics), 4)
        for metric in metrics:
            self.assertTrue(callable(metric))
        metric_types = {type(metric) for metric in metrics}
        self.assertIn(TPsnrStreaming, metric_types)
        self.assertIn(RecPsnrStreaming, metric_types)
        self.assertNotIn(TPsnr, metric_types)
        self.assertNotIn(RecPsnr, metric_types)

    def test_recpsnr_streaming_different_seq_id_types(self):
        """Recurrent PSNR streamingwith and different seq_id types."""
        # batch_size, seq_length, number_of_channels, h, w
        preds, target = generate_mock_data((1, 1, 3, 256, 256))

        recpsnr = RecPsnrStreaming()
        recpsnr.update(preds * 2, target, seq_id=torch.tensor([[[[[1]]]]]))
        recpsnr.update(preds * 4, target, seq_id=1)
        recpsnr.update(preds, target, seq_id=torch.tensor(1))
        recpsnr_result_1 = recpsnr.compute()
        recpsnr.reset()

        # Create a PSNR instance to compare, only last frame in sequence should be used.
        psnr = Psnr()
        psnr.update(preds, target)
        psnr_result_1 = psnr.compute()

        self.assertAlmostEqual(psnr_result_1.numpy(), recpsnr_result_1.numpy())

    def test_recpsnr_streaming_single_timesteps(self):
        """Recurrent PSNR with single time steps - same and different sequences."""
        # batch_size, seq_length, number_of_channels, h, w
        preds, target = generate_mock_data((1, 1, 3, 256, 256))

        recpsnr = RecPsnrStreaming()
        recpsnr.update(preds, target, seq_id=torch.tensor([1]))
        recpsnr.update(preds, target, seq_id=torch.tensor([1]))
        recpsnr.update(preds, target, seq_id=torch.tensor([1]))
        recpsnr_result_1 = recpsnr.compute()

        # Create a PSNR instance to compare
        psnr = Psnr()
        psnr.update(preds, target)
        psnr_result_1 = psnr.compute()

        self.assertAlmostEqual(psnr_result_1.numpy(), recpsnr_result_1.numpy())

        # Second sequence
        preds_2, target_2 = generate_mock_data((1, 1, 3, 256, 256))
        psnr.reset()
        psnr.update(preds_2, target_2)
        psnr_result_2 = psnr.compute()

        recpsnr.update(preds_2, target_2, seq_id=torch.tensor([2]))
        result_2 = recpsnr.compute()
        # Should not be the same as we have 2 sequences passed in now.
        self.assertNotAlmostEqual(result_2.numpy(), psnr_result_2.numpy())

        # As we have 2 sequences the result should be the average.
        self.assertAlmostEqual(
            result_2.numpy(), (psnr_result_2.numpy() + psnr_result_1.numpy()) / 2
        )

        recpsnr.reset()
        recpsnr.update(preds_2, target_2, seq_id=torch.tensor(2))
        recpsnr_result_2 = recpsnr.compute()

        self.assertAlmostEqual(psnr_result_2.numpy(), recpsnr_result_2.numpy())

    def test_tpsnr_streaming_different_seq_id_types(self):
        """tPSNR streaming with single time steps and different seq_id types."""
        # batch_size, seq_length, number_of_channels, h, w
        preds_1, target_1 = generate_mock_data((1, 1, 3, 256, 256))
        preds_2, target_2 = generate_mock_data((1, 1, 3, 256, 256))
        preds_3, target_3 = generate_mock_data((1, 1, 3, 256, 256))

        tpsnr = TPsnrStreaming()
        tpsnr.update(preds_1, target_1, seq_id=torch.tensor([[[[[1]]]]]))
        tpsnr.update(preds_2, target_2, seq_id=1)
        tpsnr.update(preds_3, target_3, seq_id=torch.tensor(1))
        tpsnr_result_1 = tpsnr.compute()
        tpsnr.reset()

        # Create a PSNR instance to compare.
        psnr = Psnr()
        psnr.update(preds_2 - preds_1, target_2 - target_1)
        psnr.update(preds_3 - preds_2, target_3 - target_2)
        psnr_result_1 = psnr.compute()

        self.assertAlmostEqual(psnr_result_1.numpy(), tpsnr_result_1.numpy())

    def test_tpsnr_streaming_single_timestep(self):
        """tPSNR with single time steps - same and different sequences."""
        # batch_size, seq_length, number_of_channels, h, w
        preds_1, target_1 = generate_mock_data((1, 1, 3, 256, 256))
        preds_2, target_2 = generate_mock_data((1, 1, 3, 256, 256))
        preds_3, target_3 = generate_mock_data((1, 1, 3, 256, 256))

        tpsnr = TPsnrStreaming()
        tpsnr.update(preds_1, target_1, seq_id=torch.tensor([1]))
        tpsnr.update(preds_2, target_2, seq_id=torch.tensor([1]))
        tpsnr.update(preds_3, target_3, seq_id=torch.tensor([1]))
        tpsnr_result_1 = tpsnr.compute()

        # Create a PSNR instance to compare.
        psnr = Psnr()
        psnr.update(preds_2 - preds_1, target_2 - target_1)
        psnr.update(preds_3 - preds_2, target_3 - target_2)
        psnr_result_1 = psnr.compute()

        self.assertAlmostEqual(psnr_result_1.numpy(), tpsnr_result_1.numpy())

        # Second sequence
        tpsnr.update(preds_1, target_1, seq_id=torch.tensor([2]))
        tpsnr.update(preds_2, target_2, seq_id=torch.tensor([2]))
        tpsnr_result_2 = tpsnr.compute()

        psnr.reset()
        psnr.update(preds_2 - preds_1, target_2 - target_1)
        psnr_result_2 = psnr.compute()

        # As we have 2 sequences the result should be the average.
        self.assertAlmostEqual(
            tpsnr_result_2.numpy(), (psnr_result_2.numpy() + psnr_result_1.numpy()) / 2
        )


if __name__ == "__main__":
    unittest.main()
