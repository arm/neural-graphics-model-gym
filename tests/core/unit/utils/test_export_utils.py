# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import DEFAULT, patch

import torch
from torch import nn

from ng_model_gym.core.model.base_ng_model import BaseNGModel
from ng_model_gym.core.model.base_ng_model_wrapper import BaseNGModelWrapper
from ng_model_gym.core.model.model import get_model_key
from ng_model_gym.core.utils.export_utils import (
    DataLoaderMode,
    executorch_vgf_export,
    ExportType,
    TrainEvalMode,
)


class MockNSS(BaseNGModel):
    """Mock class for NSS model."""

    def __init__(self):
        super().__init__()
        self.autoencoder = nn.Identity()

    def get_neural_network(self):
        """Mock get_neural_network"""
        return self.autoencoder

    def set_neural_network(self, neural_network):
        """Mock set_neural_network"""
        self.autoencoder = neural_network

    def get_additional_constants(self):
        """Mock method to return additional constants."""
        return {"foo": "bar"}


class MockFeedbackModel(BaseNGModelWrapper):
    """Mock class for Feedback model."""

    def __init__(
        self,
    ):
        super().__init__()
        self.ng_model = MockNSS()

    def get_ng_model(self) -> BaseNGModel:
        return self.ng_model

    def set_ng_model(self, ng_model: BaseNGModel) -> None:
        self.ng_model = ng_model

    def get_model_input_for_tracing(self, x):
        """Mock method to return model input for tracing."""
        return x

    def get_neural_network(self):
        """Mock get_neural_network"""
        return self.ng_model.get_neural_network()

    def set_neural_network(self, neural_network):
        """Mock set_neural_network"""
        self.ng_model = neural_network


# pylint: disable-next=unused-argument
def fake_dl(params, num_workers, prefetch_factor, loader_mode, trace_mode):
    """A mock one‐batch dataloader factory"""
    # We could record loader_mode/trace_mode here if needed
    yield (torch.zeros(1, 1),)


def make_params(tmp_path):
    """Make mock parameters for testing."""
    p = SimpleNamespace()

    p.dataset = SimpleNamespace(
        num_workers=2,
        prefetch_factor=4,
        path=SimpleNamespace(train="train_data", test="test_data"),
    )

    p.model = SimpleNamespace(name="NSS", version="42")

    p.output = SimpleNamespace(
        export=SimpleNamespace(vgf_output_dir=str(tmp_path / "vgf_output"))
    )
    return p


class TestExportUtils(unittest.TestCase):
    """Tests for export_utils module."""

    def setUp(self):
        """Setup common test data and state"""
        self.tmp_path = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.tmp_path)

    # Patch the heavy or external dependencies on every test:
    @patch("ng_model_gym.core.utils.export_utils._update_metadata_file", new=DEFAULT)
    @patch("ng_model_gym.core.utils.export_utils.get_dataloader", side_effect=fake_dl)
    @patch(
        "ng_model_gym.core.utils.export_utils.load_checkpoint",
        new=lambda *a, **k: MockFeedbackModel(),
    )
    @patch("ng_model_gym.core.utils.export_utils._export_module_to_vgf", new=DEFAULT)
    @patch("ng_model_gym.core.utils.export_utils._check_cuda", new=lambda: None)
    @patch(
        "ng_model_gym.core.utils.export_utils.model_tracer",
        new=lambda model, preprocess: torch.zeros(1, 1),
    )
    def test_qat_int8_path(
        self, mock_export_module_to_vgf, mock_get_dataloader, mock_update_metadata_file
    ):
        """Test the QAT INT8 export path."""
        params = make_params(self.tmp_path)
        executorch_vgf_export(
            params=params,
            export_type=ExportType.QAT_INT8,
            model_path=Path("checkpoint.pth"),
        )

        # Train/eval mode set correctly.
        self.assertEqual(params.model_train_eval_mode, TrainEvalMode.QAT_INT8)

        model_key = get_model_key(params.model.name, params.model.version)

        # Metadata file should be updated with constants.
        expected_meta = Path(params.output.export.vgf_output_dir) / (
            f"{model_key}-{ExportType.QAT_INT8.name}-metadata.json"
        )

        mock_update_metadata_file.assert_called_once_with(expected_meta, {"foo": "bar"})

        # Dataloader correct arguments passed.
        _, kwargs = mock_get_dataloader.call_args
        self.assertEqual(kwargs["loader_mode"], DataLoaderMode.TRAIN)
        self.assertFalse(kwargs["trace_mode"])

        # Export module to a VGF file is called with correct parameters.
        mock_export_module_to_vgf.assert_called_once()
        export_args, _ = mock_export_module_to_vgf.call_args
        _, module, trace_input, etype, meta_path = export_args
        self.assertIsInstance(module, nn.Identity)
        self.assertIsInstance(trace_input, torch.Tensor)
        self.assertEqual(etype, ExportType.QAT_INT8)

        model_key = get_model_key(params.model.name, params.model.version)

        # Metadata path should match the expected path.
        expected_meta = Path(params.output.export.vgf_output_dir) / (
            f"{model_key}-{ExportType.QAT_INT8.name}-metadata.json"
        )

        self.assertEqual(meta_path, expected_meta)

    @patch("ng_model_gym.core.utils.export_utils._update_metadata_file", new=DEFAULT)
    @patch("ng_model_gym.core.utils.export_utils.get_dataloader", side_effect=fake_dl)
    @patch(
        "ng_model_gym.core.utils.export_utils.load_checkpoint",
        new=lambda *a, **k: MockFeedbackModel(),
    )
    @patch("ng_model_gym.core.utils.export_utils._export_module_to_vgf", new=DEFAULT)
    @patch("ng_model_gym.core.utils.export_utils._check_cuda", new=lambda: None)
    @patch(
        "ng_model_gym.core.utils.export_utils.model_tracer",
        new=lambda model, preprocess: torch.zeros(1, 1),
    )
    def test_fp32_path(
        self, mock_export_module_to_vgf, mock_get_dataloader, mock_update_metadata_file
    ):
        """Test the FP32 export path."""
        params = make_params(self.tmp_path)
        executorch_vgf_export(
            params=params,
            export_type=ExportType.FP32,
            model_path=Path("checkpoint.pth"),
        )

        # Train/eval mode set correctly.
        self.assertEqual(params.model_train_eval_mode, TrainEvalMode.FP32)

        model_key = get_model_key(params.model.name, params.model.version)

        # Metadata file should be updated with constants.
        expected_meta = Path(params.output.export.vgf_output_dir) / (
            f"{model_key}-{ExportType.FP32.name}-metadata.json"
        )
        mock_update_metadata_file.assert_called_once_with(expected_meta, {"foo": "bar"})

        # Dataloader correct arguments passed.
        _, kwargs = mock_get_dataloader.call_args
        self.assertEqual(kwargs["loader_mode"], DataLoaderMode.TEST)
        self.assertTrue(kwargs["trace_mode"])

        # Export module to a VGF file called with correct parameters.
        mock_export_module_to_vgf.assert_called_once()
        export_args, _ = mock_export_module_to_vgf.call_args
        _, module, trace_input, etype, meta_path = export_args
        self.assertIsInstance(module, nn.Identity)
        self.assertIsInstance(trace_input, torch.Tensor)
        self.assertEqual(etype, ExportType.FP32)

        model_key = get_model_key(params.model.name, params.model.version)

        # Metadata path should match the expected path.
        expected_meta = Path(params.output.export.vgf_output_dir) / (
            f"{model_key}-{ExportType.FP32.name}-metadata.json"
        )
        self.assertEqual(meta_path, expected_meta)

    @patch("ng_model_gym.core.utils.export_utils._check_cuda", new=lambda: None)
    @patch(
        "ng_model_gym.core.utils.export_utils.load_checkpoint",
        new=lambda *a, **k: MockFeedbackModel(),
    )
    @patch("ng_model_gym.core.utils.export_utils.get_dataloader", new=fake_dl)
    @patch(
        "ng_model_gym.core.utils.export_utils._export_module_to_vgf",
        new=lambda *a, **k: None,
    )
    @patch(
        "ng_model_gym.core.utils.export_utils.model_tracer",
        new=lambda model, preprocess: torch.zeros(1, 1),
    )
    def test_metadata_file_is_created(self):
        """Test that metadata file is created with constants when exporting."""
        params = make_params(self.tmp_path)

        # Run with FP32 branch.
        executorch_vgf_export(params, ExportType.FP32, Path("doesnt_matter.pth"))

        model_key = get_model_key(params.model.name, params.model.version)

        # Build expected path.
        meta_path = (
            Path(params.output.export.vgf_output_dir)
            / f"{model_key}-{ExportType.FP32.name}-metadata.json"
        )

        # File must now exist.
        self.assertTrue(
            meta_path.exists(), f"Expected metadata file at {meta_path} to exist"
        )

        # And contain exactly the constants dict.
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(data, {"foo": "bar"})


if __name__ == "__main__":
    unittest.main()
