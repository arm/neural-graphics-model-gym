# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import json
import re
import subprocess
from pathlib import Path

from tests.usecases.nss.integration.base_integration_v1 import NSSV1BaseIntegrationTest

# pylint: disable=duplicate-code

NSS_V1_REFERENCE_METRICS_PATH = (
    Path(__file__).parent / "data" / "nss_v1_reference_eval_metrics.json"
)
NSS_V1_REFERENCE_QAT_METRICS_PATH = (
    Path(__file__).parent / "data" / "nss_v1_reference_qat_eval_metrics.json"
)
NSS_V1_REFERENCE_SCALE_1_5_METRICS_PATH = (
    Path(__file__).parent / "data" / "nss_v1_reference_eval_metrics_scale_1_5.json"
)


class NSSV1EvaluationIntegrationTest(NSSV1BaseIntegrationTest):
    """Tests for NSS v1 evaluation pipeline."""

    _REFERENCE_METRIC_FLOOR_DELTA = 0.0001
    _PSNR_METRIC_FLOOR_DELTA = 0.001

    def _extract_metric_value(self, metric, log_line):
        """Helper function to extract metric values from log line."""
        match = re.search(f"{metric}" + r": (\d+\.\d+),", log_line)
        if match:
            return float(match.group(1))
        return None

    def _read_metric_value(self, metric, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                metric_val = self._extract_metric_value(metric, line)
                if metric_val is not None:
                    return metric_val
        return None

    def _read_final_metric_values(self, output_dir):
        """Read final structured metric values from an evaluation output dir."""
        metrics_paths = sorted(Path(output_dir).glob("eval_metrics_*.json"))
        self.assertTrue(metrics_paths)

        with open(metrics_paths[-1], encoding="utf-8") as f:
            metric_history = json.load(f)

        final_metrics = {}
        for metric, values in metric_history.items():
            final_key = max(values, key=int)
            final_metrics[metric] = values[final_key]
        return final_metrics

    def _assert_metric_at_least_reference(
        self,
        metric_value,
        expected_value,
        *,
        floor_delta,
        msg,
    ):
        """Assert a metric is not materially worse than the reference value."""
        self.assertGreaterEqual(
            metric_value,
            expected_value - floor_delta,
            f"{msg} Delta: {metric_value - expected_value:+.6f}.",
        )

    def _reference_metric_floor_delta(self, model_type, quality, metric):
        """Return the allowed one-sided reference delta for a metric."""
        del model_type, quality
        if metric in ("PSNR", "tPSNRStreaming"):
            return self._PSNR_METRIC_FLOOR_DELTA

        return self._REFERENCE_METRIC_FLOOR_DELTA

    def test_reference_metric_floor_delta_uses_psnr_like_tolerance(self):
        """PSNR-like metrics use a larger floor delta for platform differences."""
        self.assertEqual(
            self._reference_metric_floor_delta("fp32", "mid", "PSNR"),
            self._PSNR_METRIC_FLOOR_DELTA,
        )
        self.assertEqual(
            self._reference_metric_floor_delta("fp32", "mid", "tPSNRStreaming"),
            self._PSNR_METRIC_FLOOR_DELTA,
        )
        self.assertEqual(
            self._reference_metric_floor_delta("qat_int8", "mid", "PSNR"),
            self._PSNR_METRIC_FLOOR_DELTA,
        )
        self.assertEqual(
            self._reference_metric_floor_delta("qat_int8", "mid", "tPSNRStreaming"),
            self._PSNR_METRIC_FLOOR_DELTA,
        )
        self.assertEqual(
            self._reference_metric_floor_delta("fp32", "mid", "SSIM"),
            self._REFERENCE_METRIC_FLOOR_DELTA,
        )

    def _write_quality_eval_config(
        self,
        quality,
        *,
        output_suffix=None,
        export_frame_png=False,
        scale=2.0,
    ):
        """Write an evaluation config for the requested quality mode."""
        cfg_json = json.loads(json.dumps(self.cfg_json))
        cfg_json["model"]["quality"] = quality
        cfg_json["model"]["scale"] = scale
        cfg_json["dataset"]["path"]["train"] = "tests/usecases/nss/datasets/train"
        cfg_json["dataset"]["path"]["validation"] = "tests/usecases/nss/datasets/val"
        cfg_json["dataset"]["path"]["test"] = "tests/usecases/nss/datasets/test"
        cfg_json["metrics"] = ["PSNR", "SSIM", "tPSNR"]
        cfg_json["output"]["export_frame_png"] = export_frame_png

        if output_suffix is None:
            output_suffix = quality
        output_dir = Path(self.test_dir, f"model_output_{output_suffix}")
        cfg_json["output"]["dir"] = str(output_dir)

        config_path = Path(self.test_dir, f"test_eval_{output_suffix}.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f)

        return config_path, output_dir

    @staticmethod
    def _model_path_for_quality(_quality):
        """Return the FP32 checkpoint path for the requested quality mode."""
        return "tests/usecases/nss/weights/nss_v1_high_fp32.pt"

    @staticmethod
    def _qat_model_path_for_quality(quality):
        """Return the quantized checkpoint path for the requested quality mode."""
        if quality == "high":
            return "tests/usecases/nss/weights/nss_v1_high_int8.pt"

        return "tests/usecases/nss/weights/nss_v1_mid_low_int8.pt"

    def _evaluate_from_checkpoint(
        self,
        model_path,
        *,
        output_suffix,
        quality="high",
        model_type="fp32",
        export_frame_png=False,
        scale=2.0,
    ):
        """Evaluate an NSS v1 checkpoint."""
        config_path, output_dir = self._write_quality_eval_config(
            quality,
            output_suffix=output_suffix,
            export_frame_png=export_frame_png,
            scale=scale,
        )

        sub_proc = subprocess.run(
            [
                "ng-model-gym",
                f"--config-path={config_path}",
                "evaluate",
                f"--model-path={model_path}",
                f"--model-type={model_type}",
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            sub_proc.returncode,
            0,
            (
                "NSS v1 evaluation failed.\n"
                f"STDOUT:\n{sub_proc.stdout}\n"
                f"STDERR:\n{sub_proc.stderr}"
            ),
        )
        results_path = Path(output_dir, "results.log")
        self.assertTrue(results_path.exists())
        return output_dir, results_path

    def test_evaluate_from_checkpoints_matches_reference_metrics_by_quality(self):
        """Evaluate NSS v1 quality modes against reference metric values."""
        with open(NSS_V1_REFERENCE_METRICS_PATH, encoding="utf-8") as f:
            reference_metrics = json.load(f)

        for quality, expected_metrics in reference_metrics.items():
            with self.subTest(quality=quality):
                output_dir, _ = self._evaluate_from_checkpoint(
                    self._model_path_for_quality(quality),
                    output_suffix=quality,
                    quality=quality,
                    model_type="fp32",
                )
                metric_values = self._read_final_metric_values(output_dir)

                for metric, expected_value in expected_metrics.items():
                    metric_value = metric_values.get(metric)
                    self.assertIsNotNone(metric_value)
                    self._assert_metric_at_least_reference(
                        metric_value,
                        expected_value,
                        floor_delta=self._reference_metric_floor_delta(
                            "fp32",
                            quality,
                            metric,
                        ),
                        msg=f"{quality} {metric} differs from reference value.",
                    )

    def test_evaluate_from_identifier(self):
        """Evaluate NSS v1 using a remote model identifier."""
        _, results_path = self._evaluate_from_checkpoint(
            "@neural-super-sampling/nss_v1_high_fp32.pt",
            output_suffix="high_identifier",
        )

        for metric in ("PSNR", "SSIM", "tPSNRStreaming"):
            self.assertIsNotNone(self._read_metric_value(metric, results_path))

    def test_evaluate_from_checkpoint_with_non_default_scale(self):
        """Evaluate NSS v1 non-default scale against reference metric values."""
        with open(NSS_V1_REFERENCE_SCALE_1_5_METRICS_PATH, encoding="utf-8") as f:
            expected_metrics = json.load(f)["high"]

        output_dir, _ = self._evaluate_from_checkpoint(
            self._model_path_for_quality("high"),
            output_suffix="high_scale_1_5",
            scale=1.5,
        )
        metric_values = self._read_final_metric_values(output_dir)

        for metric, expected_value in expected_metrics.items():
            metric_value = metric_values.get(metric)
            self.assertIsNotNone(metric_value)
            self._assert_metric_at_least_reference(
                metric_value,
                expected_value,
                floor_delta=self._reference_metric_floor_delta(
                    "fp32",
                    "high",
                    metric,
                ),
                msg=f"high scale 1.5 {metric} differs from reference value.",
            )

    def test_evaluate_from_qat_checkpoint(self):
        """Evaluate NSS v1 quantized quality modes against reference metric values."""
        with open(NSS_V1_REFERENCE_QAT_METRICS_PATH, encoding="utf-8") as f:
            reference_metrics = json.load(f)

        for quality, expected_metrics in reference_metrics.items():
            with self.subTest(quality=quality):
                output_dir, _ = self._evaluate_from_checkpoint(
                    self._qat_model_path_for_quality(quality),
                    output_suffix=f"{quality}_qat",
                    quality=quality,
                    model_type="qat_int8",
                )
                metric_values = self._read_final_metric_values(output_dir)

                for metric, expected_value in expected_metrics.items():
                    metric_value = metric_values.get(metric)
                    self.assertIsNotNone(metric_value)
                    self._assert_metric_at_least_reference(
                        metric_value,
                        expected_value,
                        floor_delta=self._reference_metric_floor_delta(
                            "qat_int8",
                            quality,
                            metric,
                        ),
                        msg=f"{quality} {metric} differs from reference value.",
                    )

    def test_evaluate_exports_png_frames(self):
        """Evaluate NSS v1 and export predicted and ground-truth frames."""
        output_dir, _ = self._evaluate_from_checkpoint(
            self._model_path_for_quality("high"),
            output_suffix="high_export_frames",
            export_frame_png=True,
        )

        exported_prediction = Path(
            output_dir,
            "png",
            "predicted",
            "frame_0000_pred.png",
        )
        exported_ground_truth = Path(
            output_dir,
            "png",
            "ground_truth",
            "frame_0000_gt.png",
        )
        self.assertTrue(exported_prediction.exists())
        self.assertTrue(exported_ground_truth.exists())

    def test_trace_profiler(self):
        """Evaluate NSS v1 with trace profiling enabled."""
        self.run_model_profiler("eval")

    def test_cuda_profiler(self):
        """Evaluate NSS v1 with CUDA memory profiling enabled."""
        self.run_cuda_profiler_test("eval")
