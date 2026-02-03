# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import json
import platform
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import DEFAULT, patch

import torch
from torch import nn

from ng_model_gym.core.model import BaseNGModel, BaseNGModelWrapper, get_model_key
from ng_model_gym.core.utils.export_utils import (
    DataLoaderMode,
    executorch_vgf_export,
    ExportType,
    TrainEvalMode,
)


def _flatten(container):
    """Flatten nested containers into a list of tensors."""
    if isinstance(container, torch.Tensor):
        return [container]
    if isinstance(container, (list, tuple)):
        out = []
        for it in container:
            out.extend(_flatten(it))
        return out
    if isinstance(container, dict):
        out = []
        for v in container.values():
            out.extend(_flatten(v))
        return out
    if hasattr(container, "_asdict"):  # namedtuple
        return _flatten(container._asdict())
    return []


class MockNSS(BaseNGModel):
    """Mock class for NSS model."""

    def __init__(self, params):
        super().__init__(params)
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

    def __init__(self, params):
        super().__init__()
        self.ng_model = MockNSS(params)

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

    num_workers = 0 if platform.system() == "Windows" else 2

    p.dataset = SimpleNamespace(
        num_workers=num_workers,
        prefetch_factor=4,
        path=SimpleNamespace(train="train_data", test="test_data"),
    )

    p.model = SimpleNamespace(name="NSS", version="42")

    p.output = SimpleNamespace(
        export=SimpleNamespace(
            vgf_output_dir=str(tmp_path / "vgf_output"),
            dynamic_shape=True,
            vgf_static_input_shape=None,
        )
    )
    return p


class TestExportUtils(unittest.TestCase):
    """Tests for export_utils module."""

    def setUp(self):
        """Setup common test data and state"""
        self.tmp_path = Path(tempfile.mkdtemp())
        self.params = make_params(self.tmp_path)
        self.load_ckpt_patch = patch(
            "ng_model_gym.core.utils.export_utils.load_checkpoint",
            new=lambda *a, **k: MockFeedbackModel(self.params),
        )
        self.load_ckpt_patch.start()

    def tearDown(self):
        """Clean up the temporary directory."""
        self.load_ckpt_patch.stop()
        shutil.rmtree(self.tmp_path)

    # Patch the heavy or external dependencies on every test:
    @patch("ng_model_gym.core.utils.export_utils._update_metadata_file", new=DEFAULT)
    @patch("ng_model_gym.core.utils.export_utils.get_dataloader", side_effect=fake_dl)
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

        executorch_vgf_export(
            params=self.params,
            export_type=ExportType.QAT_INT8,
            model_path=Path("checkpoint.pt"),
        )

        # Train/eval mode set correctly.
        self.assertEqual(self.params.model_train_eval_mode, TrainEvalMode.QAT_INT8)

        model_key = get_model_key(self.params.model.name, self.params.model.version)

        # Metadata file should be updated with constants.
        expected_meta = Path(self.params.output.export.vgf_output_dir) / (
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
        self.assertIsInstance(module, BaseNGModel)
        self.assertIsInstance(trace_input, torch.Tensor)
        self.assertEqual(etype, ExportType.QAT_INT8)

        model_key = get_model_key(self.params.model.name, self.params.model.version)

        # Metadata path should match the expected path.
        expected_meta = Path(self.params.output.export.vgf_output_dir) / (
            f"{model_key}-{ExportType.QAT_INT8.name}-metadata.json"
        )

        self.assertEqual(meta_path, expected_meta)

    @patch("ng_model_gym.core.utils.export_utils._update_metadata_file", new=DEFAULT)
    @patch("ng_model_gym.core.utils.export_utils.get_dataloader", side_effect=fake_dl)
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
        executorch_vgf_export(
            params=self.params,
            export_type=ExportType.FP32,
            model_path=Path("checkpoint.pt"),
        )

        # Train/eval mode set correctly.
        self.assertEqual(self.params.model_train_eval_mode, TrainEvalMode.FP32)

        model_key = get_model_key(self.params.model.name, self.params.model.version)

        # Metadata file should be updated with constants.
        expected_meta = Path(self.params.output.export.vgf_output_dir) / (
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
        self.assertIsInstance(module, BaseNGModel)
        self.assertIsInstance(trace_input, torch.Tensor)
        self.assertEqual(etype, ExportType.FP32)

        model_key = get_model_key(self.params.model.name, self.params.model.version)

        # Metadata path should match the expected path.
        expected_meta = Path(self.params.output.export.vgf_output_dir) / (
            f"{model_key}-{ExportType.FP32.name}-metadata.json"
        )
        self.assertEqual(meta_path, expected_meta)

    @patch("ng_model_gym.core.utils.export_utils._check_cuda", new=lambda: None)
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
        # Run with FP32 branch.
        executorch_vgf_export(self.params, ExportType.FP32, Path("doesnt_matter.pt"))

        model_key = get_model_key(self.params.model.name, self.params.model.version)

        # Build expected path.
        meta_path = (
            Path(self.params.output.export.vgf_output_dir)
            / f"{model_key}-{ExportType.FP32.name}-metadata.json"
        )

        # File must now exist.
        self.assertTrue(
            meta_path.exists(), f"Expected metadata file at {meta_path} to exist"
        )

        # And contain exactly the constants dict.
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(data, {"foo": "bar"})

    @patch("ng_model_gym.core.utils.export_utils._check_cuda", new=lambda: None)
    def test_inputs_channels_last_for_4d(self):
        """Ensure 4D tensors become channels_last and non-tensors unchanged."""

        # Build nested input with 4D + non-4D tensors and non-tensors.
        tensor_nchw = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(
            2, 3, 4, 5
        )
        tensor_4d = torch.randn(1, 1, 2, 2)
        tensor_3d = torch.randn(3, 4, 5)
        non_tensor = {"k": 123, "s": "testing"}

        nested_input = (
            {"img": tensor_nchw, "meta": non_tensor},
            [tensor_4d, tensor_3d, 7],
        )

        def local_dl(
            params, num_workers, prefetch_factor, loader_mode, trace_mode
        ):  # pylint: disable=unused-argument
            yield (nested_input, 0)

        # Make tracer return preprocess input unchanged to observe changes from export_utils
        def passthrough_tracer(
            model, preprocess_trace_input
        ):  # pylint: disable=unused-argument
            return preprocess_trace_input

        # Capture model_forward_input passed into _export_module_to_vgf
        captured = {}

        def capture_export(
            params, model, model_forward_input, export_type, metadata_path
        ):  # pylint: disable=unused-argument
            captured["input"] = model_forward_input

        with patch(
            "ng_model_gym.core.utils.export_utils.get_dataloader", side_effect=local_dl
        ), patch(
            "ng_model_gym.core.utils.export_utils.model_tracer", new=passthrough_tracer
        ), patch(
            "ng_model_gym.core.utils.export_utils._export_module_to_vgf",
            new=capture_export,
        ), patch(
            "ng_model_gym.core.utils.export_utils._update_metadata_file",
            new=lambda *a, **k: None,
        ):
            executorch_vgf_export(
                params=self.params,
                export_type=ExportType.FP32,
                model_path=Path("checkpoint.pt"),
            )

        self.assertIn("input", captured, "Export did not reach the VGF step.")
        transformed = captured["input"]

        # Tensor values, dtypes and shapes preserved
        orig_tensors = _flatten(nested_input)
        new_tensors = _flatten(transformed)
        self.assertEqual(len(orig_tensors), len(new_tensors))

        for o, n in zip(orig_tensors, new_tensors):
            self.assertIsInstance(n, torch.Tensor)
            self.assertEqual(o.shape, n.shape)
            self.assertEqual(o.dtype, n.dtype)
            self.assertTrue(
                torch.allclose(o.detach().cpu(), n),
                "Tensor values changed during format move.",
            )

        # 4D tensors must be channels_last
        for t in new_tensors:
            if t.ndim == 4:
                self.assertTrue(
                    (t.is_contiguous(memory_format=torch.channels_last)),
                    "4D tensor was not channels_last.",
                )

        # Non-tensor payload should be unchanged
        self.assertIsInstance(transformed, tuple)
        self.assertEqual(transformed[0]["meta"]["k"], 123)
        self.assertEqual(transformed[0]["meta"]["s"], "testing")
