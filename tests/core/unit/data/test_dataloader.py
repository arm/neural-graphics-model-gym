# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ng_model_gym.core.data import DataLoaderMode
from ng_model_gym.core.data.dataloader import _get_dataloader_batch_size, get_dataloader


class TestDataloaderBatchSize(unittest.TestCase):
    """Tests determination of dataloader batch size."""

    def test_trace_mode_train_val_uses_configured_batch_size(self):
        """
        TRAIN and VAL dataloaders should use the configured batch size when in
        trace mode. (TEST dataloaders use a batch size of 1: see
        test_test_uses_batch_size_1().)
        """
        for loader_mode in [
            DataLoaderMode.TRAIN,
            DataLoaderMode.VAL,
        ]:
            batch_size = _get_dataloader_batch_size(
                configured_batch_size=8,
                dataset_size=16,
                loader_mode=loader_mode,
                trace_mode=True,
            )

            self.assertEqual(batch_size, 8)

    def test_train_val_uses_configured_batch_size(self):
        """
        TRAIN and VAL dataloaders should use the explicitly configured batch
        size when *not* in trace mode. Also confirm that single element batches
        (edge case) do not produce errors. Compare with
        test_train_val_trace_mode_cannot_use_batch_size_of_1().
        """
        for loader_mode in [
            DataLoaderMode.TRAIN,
            DataLoaderMode.VAL,
        ]:
            batch_size = _get_dataloader_batch_size(
                configured_batch_size=1,
                dataset_size=16,
                loader_mode=loader_mode,
                trace_mode=False,
            )

            self.assertEqual(batch_size, 1)

    def test_train_val_throws_when_batch_size_larger_than_dataset(self):
        """
        TRAIN and VAL dataloaders should throw when the batch size is larger
        than the dataset size.
        """
        for loader_mode in [
            DataLoaderMode.TRAIN,
            DataLoaderMode.VAL,
        ]:
            with pytest.raises(ValueError) as exception_info:
                _ = _get_dataloader_batch_size(
                    configured_batch_size=2,
                    dataset_size=1,
                    loader_mode=loader_mode,
                    trace_mode=False,
                )

            self.assertIn("larger than the dataset", str(exception_info.value))

    def test_test_uses_batch_size_1(self):
        """
        TEST dataloaders have a batch size of 1 regardless of configured
        batch size.
        """

        batch_size = _get_dataloader_batch_size(
            configured_batch_size=8,
            dataset_size=16,
            loader_mode=DataLoaderMode.TEST,
            trace_mode=False,
        )

        self.assertEqual(batch_size, 1)

    def test_test_trace_mode_throws(self):
        """
        TEST dataloaders cannot be used in trace mode and should throw an
        exception.
        """
        with pytest.raises(ValueError) as exception_info:
            _ = _get_dataloader_batch_size(
                configured_batch_size=8,
                dataset_size=16,
                loader_mode=DataLoaderMode.TEST,
                trace_mode=True,
            )

        self.assertIn("cannot be used together", str(exception_info.value))

    def test_train_val_trace_mode_cannot_use_batch_size_of_1(self):
        """
        TRAIN and VAL dataloaders cannot use a batch size of 1 if in trace
        mode.
        """
        for loader_mode in [
            DataLoaderMode.TRAIN,
            DataLoaderMode.VAL,
        ]:
            with pytest.raises(ValueError) as exception_info:
                _ = _get_dataloader_batch_size(
                    configured_batch_size=1,
                    dataset_size=16,
                    loader_mode=loader_mode,
                    trace_mode=True,
                )

            self.assertIn("batch size must be 2 or more", str(exception_info.value))


class TestGetDataloader(unittest.TestCase):
    """Tests dataloader construction."""

    @patch("ng_model_gym.core.data.dataloader.get_dataset", return_value=[])
    def test_raises_on_empty_dataset(self, _mock_get_dataset):
        """get_dataloader() should reject empty datasets."""
        config_params = SimpleNamespace(
            train=SimpleNamespace(batch_size=1, seed=123),
            dataset=SimpleNamespace(health_check=False),
        )

        with pytest.raises(ValueError) as exception_info:
            _ = get_dataloader(config_params)

        self.assertEqual(str(exception_info.value), "Cannot process empty dataset.")
