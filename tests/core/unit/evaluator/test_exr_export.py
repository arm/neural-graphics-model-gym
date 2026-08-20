# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
"""Tests for EXR export during evaluation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, Mock, patch

import torch
from torch import nn

from ng_model_gym.core.evaluator.evaluator import NGModelEvaluator
from ng_model_gym.core.model.base_ng_model import BaseNGModel
from tests.testing_utils import create_simple_params


class _OrdinaryEvaluationModel(BaseNGModel):
    """Minimal model used to exercise the default EXR selection hook."""

    def __init__(self):
        super().__init__(create_simple_params(usecase="nss-v1"))
        self.network = nn.Identity()

    def get_neural_network(self) -> nn.Module:
        return self.network

    def set_neural_network(self, neural_network: nn.Module) -> None:
        self.network = neural_network


class TestEvaluatorEXRExport(unittest.TestCase):
    """Test evaluator integration for model-selected EXR export."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        self.output_dir = Path(self.temp_dir.name, "out")
        self.params = create_simple_params(
            usecase="nss-v1", output_dir=str(self.output_dir)
        )
        self.params.output.export_frame_png = False
        self.params.output.export_frame_exr = True
        self.model = Mock(spec=BaseNGModel)
        self.model.device = torch.device("cpu")
        self.metrics_patch = patch(
            "ng_model_gym.core.evaluator.evaluator.get_metrics", return_value=[]
        )
        self.metrics_patch.start()

    def tearDown(self) -> None:
        self.metrics_patch.stop()
        self.temp_dir.cleanup()

    def _evaluator(self) -> NGModelEvaluator:
        return NGModelEvaluator(self.model, self.params)

    def test_startup_uses_common_lifecycle_and_creates_requested_trees(self) -> None:
        """Create each requested export tree using the common evaluation lifecycle."""
        for exr_enabled in (False, True):
            with self.subTest(exr_enabled=exr_enabled):
                self.model.reset_mock()
                output_dir = self.output_dir / str(exr_enabled)
                self.params.output.dir = str(output_dir)
                self.params.output.export_frame_png = True
                self.params.output.export_frame_exr = exr_enabled
                evaluator = self._evaluator()
                evaluator.prepare_datasets = Mock()
                lifecycle = Mock()
                lifecycle.attach_mock(evaluator.prepare_datasets, "prepare_datasets")
                lifecycle.attach_mock(self.model.eval, "eval")
                lifecycle.attach_mock(
                    self.model.on_evaluation_start, "on_evaluation_start"
                )

                evaluator._test_begin()

                self.assertEqual(
                    lifecycle.mock_calls,
                    [call.prepare_datasets(), call.eval(), call.on_evaluation_start()],
                )
                self.assertTrue((output_dir / "png" / "predicted").is_dir())
                self.assertTrue((output_dir / "png" / "ground_truth").is_dir())
                self.assertEqual(
                    (output_dir / "exr" / "predicted").is_dir(), exr_enabled
                )
                self.assertEqual(
                    (output_dir / "exr" / "ground_truth").is_dir(), exr_enabled
                )

    def test_one_forward_updates_metrics_and_writes_png_and_exr(self) -> None:
        """Pass the complete output mapping and transferred target to the hook."""
        self.params.output.export_frame_png = True
        evaluator = self._evaluator()
        inputs = {"seq": torch.tensor([7])}
        prediction = torch.full((1, 3, 2, 2), 0.25)
        target = torch.full((1, 3, 2, 2), 0.75)
        outputs = {
            "output": prediction,
            "output_linear": torch.full((1, 3, 2, 2), 2.0),
            "diagnostic": object(),
        }
        exr_prediction = torch.full((1, 3, 2, 2), 12.0)
        exr_ground_truth = torch.full((1, 3, 2, 2), 24.0)
        metric = Mock(is_streaming=False)
        evaluator.metrics = [metric]
        evaluator.x_in = inputs
        evaluator.y_true = target
        evaluator.idx = 7
        self.model.return_value = outputs
        self.model.get_evaluation_exr_frames.return_value = (
            exr_prediction,
            exr_ground_truth,
        )
        expected_paths = [
            self.output_dir / "png" / "predicted" / "frame_0007_pred.png",
            self.output_dir / "png" / "ground_truth" / "frame_0007_gt.png",
            self.output_dir / "exr" / "predicted" / "frame_0007_pred.exr",
            self.output_dir / "exr" / "ground_truth" / "frame_0007_gt.exr",
        ]
        writer_events = []

        def save_png(image: torch.Tensor, path: Path) -> None:
            writer_events.append((path, image))

        def save_exr(path: Path, *, frame: torch.Tensor) -> None:
            writer_events.append((path, frame))

        model_outputs = evaluator._run_model()
        with (
            patch(
                "ng_model_gym.core.evaluator.evaluator.torchvision.utils.save_image",
                side_effect=save_png,
            ),
            patch(
                "ng_model_gym.core.evaluator.evaluator.write_rgb_float32_exr",
                side_effect=save_exr,
            ),
        ):
            evaluator._predict_end(model_outputs)

        self.model.assert_called_once_with(inputs)
        self.assertIs(model_outputs, outputs)
        self.assertIs(evaluator.y_pred, prediction)
        metric.update.assert_called_once_with(prediction, target)
        (
            hook_inputs,
            hook_outputs,
            hook_target,
        ) = self.model.get_evaluation_exr_frames.call_args.args
        self.assertIs(hook_inputs, inputs)
        self.assertIs(hook_outputs, outputs)
        self.assertIs(hook_target, target)
        self.assertEqual([event[0] for event in writer_events], expected_paths)
        self.assertTrue(torch.equal(writer_events[0][1], prediction[0]))
        self.assertTrue(torch.equal(writer_events[1][1], target[0]))
        self.assertIs(writer_events[2][1], exr_prediction)
        self.assertIs(writer_events[3][1], exr_ground_truth)

    def test_second_exr_failure_reports_path_and_leaves_first_export(self) -> None:
        """Keep an earlier direct export if the following writer fails."""
        evaluator = self._evaluator()
        evaluator.x_in = {"seq": torch.tensor([1])}
        evaluator.y_pred = torch.zeros(1, 3, 2, 2)
        evaluator.y_true = torch.ones(1, 3, 2, 2)
        model_outputs = {"output": evaluator.y_pred}
        evaluator.idx = 0
        exr_prediction = torch.full((1, 3, 2, 2), 12.0)
        exr_ground_truth = torch.full((1, 3, 2, 2), 24.0)
        self.model.get_evaluation_exr_frames.return_value = (
            exr_prediction,
            exr_ground_truth,
        )
        predicted_path = self.output_dir / "exr/predicted/frame_0000_pred.exr"
        ground_truth_path = self.output_dir / "exr/ground_truth/frame_0000_gt.exr"
        predicted_path.parent.mkdir(parents=True)
        ground_truth_path.parent.mkdir(parents=True)
        attempted_paths = []
        error = OSError("ground-truth EXR write failed")

        def fail_second_write(path: Path, *, frame: torch.Tensor) -> None:
            attempted_paths.append(path)
            if len(attempted_paths) == 1:
                self.assertIs(frame, exr_prediction)
                path.write_bytes(b"complete prediction EXR")
                return
            self.assertIs(frame, exr_ground_truth)
            raise error

        with (
            patch(
                "ng_model_gym.core.evaluator.evaluator.write_rgb_float32_exr",
                side_effect=fail_second_write,
            ),
            self.assertLogs(
                "ng_model_gym.core.evaluator.evaluator", level="ERROR"
            ) as logs,
            self.assertRaises(OSError) as raised,
        ):
            evaluator._predict_end(model_outputs)

        self.assertIs(raised.exception, error)
        self.assertEqual(attempted_paths, [predicted_path, ground_truth_path])
        self.assertEqual(predicted_path.read_bytes(), b"complete prediction EXR")
        self.assertFalse(ground_truth_path.exists())
        self.assertIn("frame 0", "\n".join(logs.output))
        self.assertIn(str(ground_truth_path), "\n".join(logs.output))

    def test_disabled_exr_export_does_not_call_model_hook(self) -> None:
        """Models without EXR support evaluate normally when export is disabled."""
        self.params.output.export_frame_exr = False
        evaluator = self._evaluator()
        evaluator.x_in = {"seq": torch.tensor([1])}
        evaluator.y_pred = torch.zeros(1, 3, 2, 2)
        evaluator.y_true = torch.ones(1, 3, 2, 2)

        evaluator._predict_end(None)

        self.model.get_evaluation_exr_frames.assert_not_called()

    def test_unsupported_model_error_propagates_when_exr_enabled(self) -> None:
        """Propagate the opt-in error on the first requested EXR frame."""
        model = _OrdinaryEvaluationModel()
        evaluator = NGModelEvaluator(model, self.params)
        evaluator.x_in = {"seq": torch.tensor([1])}
        evaluator.y_pred = torch.zeros(1, 3, 2, 2)
        evaluator.y_true = torch.ones(1, 3, 2, 2)

        with (
            patch(
                "ng_model_gym.core.evaluator.evaluator.write_rgb_float32_exr"
            ) as writer,
            self.assertRaisesRegex(
                NotImplementedError,
                "output.export_frame_exr",
            ),
        ):
            evaluator._predict_end({"output": evaluator.y_pred})

        writer.assert_not_called()

    def test_base_hook_rejects_exr_export(self) -> None:
        """Custom models must opt in to EXR export."""
        model = _OrdinaryEvaluationModel()

        with self.assertRaisesRegex(
            NotImplementedError,
            "output.export_frame_exr",
        ):
            model.get_evaluation_exr_frames(
                {"unused": torch.tensor(0)},
                {"output": torch.rand(1, 3, 2, 2)},
                torch.rand(1, 3, 2, 2),
            )
