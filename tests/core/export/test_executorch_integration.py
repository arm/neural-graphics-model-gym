# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import glob
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from ng_model_gym.api import do_export
from ng_model_gym.core.utils.export_utils import ExportType
from ng_model_gym.core.utils.types import TrainEvalMode
from tests.testing_utils import create_simple_params


class TestExecuTorchIntegration(unittest.TestCase):
    """Integration tests for exporting via ExecuTorch."""

    fp32_file_size = None

    @classmethod
    def setUpClass(cls):
        """Run only once before executing tests."""
        cls.startTimeTotal = time.time()
        cls.test_dir = tempfile.mkdtemp()
        cls.tosa_out_dir = Path(cls.test_dir) / "tosa"

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory after all tests."""
        t = time.time() - cls.startTimeTotal
        print(f"Total runtime is {t:.3f}s")
        if cls.test_dir and Path(cls.test_dir).exists():
            shutil.rmtree(cls.test_dir)

    @classmethod
    def tearDown(cls):
        """Clean up after every test."""
        if cls.tosa_out_dir and Path(cls.tosa_out_dir).exists():
            shutil.rmtree(cls.tosa_out_dir)

    def validate_vgf_export(self):
        """Check VGF file is produced"""
        with self.subTest("Validate VGF file export"):
            vgf_output_dir = Path(self.tosa_out_dir)
            vgf_files = glob.glob(f"{vgf_output_dir}/*.vgf")
            self.assertTrue(len(vgf_files) == 1)

            size = Path(vgf_files[0]).stat().st_size

            self.assertGreater(
                size, 100_000, f"VGF file size {size / 1000}KB is smaller than 100 KB"
            )

    def test_executorch_tosa_export_fp32(self):
        """Load an NSS model and export it to a TOSA file via ExecuTorch."""
        # Load config and setup for current test
        params = create_simple_params()
        params.dataset.path.test = Path("tests/usecases/nss/datasets/test")

        params.output.export.vgf_output_dir = self.tosa_out_dir
        params.model_train_eval_mode = TrainEvalMode.FP32

        # Ensure TOSA output directory does not exist
        tosa_out_dir = Path(params.output.export.vgf_output_dir)
        self.assertFalse(tosa_out_dir.exists())

        do_export(
            params,
            "./tests/usecases/nss/weights/nss_v0.1.0_fp32.pt",
            ExportType.FP32,
        )

        # Verify TOSA folder and output files created
        output_tosa_files = glob.glob(f"{tosa_out_dir}/*.tosa")
        output_json_file_exists = glob.glob(f"{tosa_out_dir}/*.json")
        self.assertTrue(tosa_out_dir.exists())
        self.assertTrue(output_tosa_files)
        self.assertTrue(output_json_file_exists)

        # Find the largest TOSA file
        largest_file_size = 0
        for tosa_file in output_tosa_files:
            file_size = Path(tosa_file).stat().st_size
            if file_size > largest_file_size:
                largest_file_size = file_size

        self.assertGreater(
            largest_file_size,
            100_000,
            "TOSA file generated incorrectly "
            f"as per the small file size of {largest_file_size / 1000} KB. ",
        )

        self.assertLess(
            largest_file_size,
            1_000_000,
            "TOSA file generated incorrectly "
            f"as per the large file size of {largest_file_size / 1000} KB. ",
        )

        TestExecuTorchIntegration.fp32_file_size = largest_file_size
        self.validate_vgf_export()

    def test_executorch_tosa_export_ptq_int8(self):
        """Load an NSS model and export it to an int8 TOSA file via ExecuTorch."""
        # Load config and setup for current test
        params = create_simple_params()
        params.output.export.vgf_output_dir = self.tosa_out_dir
        params.dataset.path.test = Path("tests/usecases/nss/datasets/test")

        # Ensure TOSA output directory does not exist
        tosa_out_dir = Path(params.output.export.vgf_output_dir)
        self.assertFalse(tosa_out_dir.exists())

        do_export(
            params,
            "./tests/usecases/nss/weights/nss_v0.1.0_fp32.pt",
            ExportType.PTQ_INT8,
        )

        # Verify TOSA folder and output files created
        output_tosa_files = glob.glob(f"{tosa_out_dir}/*.tosa")
        output_json_file_exists = glob.glob(f"{tosa_out_dir}/*.json")
        self.assertTrue(tosa_out_dir.exists())
        self.assertTrue(output_tosa_files)
        self.assertTrue(output_json_file_exists)

        # Find the largest TOSA file
        largest_file_size = 0
        for tosa_file in output_tosa_files:
            file_size = Path(tosa_file).stat().st_size
            if file_size > largest_file_size:
                largest_file_size = file_size

        self.assertGreater(
            largest_file_size,
            100_000,
            "TOSA file generated incorrectly "
            f"as per the small file size of {largest_file_size / 1000} KB. ",
        )

        self.assertLess(
            largest_file_size,
            1_000_000,
            "TOSA file generated incorrectly "
            f"as per the large file size of {largest_file_size / 1000} KB. ",
        )

        # Compare .tosa file sizes between ptq int8 model
        # and fp32 model if fp32 test ran successfully
        if TestExecuTorchIntegration.fp32_file_size is not None:
            ptq_int8_file_size = largest_file_size
            model_size_ratio = (
                ptq_int8_file_size / TestExecuTorchIntegration.fp32_file_size
            )

            # Ensure int8 model is ~1/4 size of fp32 model
            self.assertAlmostEqual(
                model_size_ratio,
                0.25,
                delta=0.1,
                msg="int8 TOSA file from PTQ is not ~1/4 size of fp32 TOSA file.",
            )

        self.validate_vgf_export()

    def test_executorch_tosa_qat_int8_export(self):
        """Load checkpoint produced from QAT training"""
        params = create_simple_params()
        params.output.export.vgf_output_dir = self.tosa_out_dir
        params.dataset.path.train = Path("tests/usecases/nss/datasets/train")

        # Ensure tosa output directory does not exist
        tosa_out_dir = Path(params.output.export.vgf_output_dir)
        self.assertFalse(tosa_out_dir.exists())

        do_export(
            params,
            "./tests/usecases/nss/weights/nss_v0.1.1_int8.pt",
            ExportType.QAT_INT8,
        )

        # Verify tosa folder and output files created
        output_tosa_files = glob.glob(f"{tosa_out_dir}/*.tosa")
        output_json_file_exists = glob.glob(f"{tosa_out_dir}/*.json")
        self.assertTrue(tosa_out_dir.exists())
        self.assertTrue(output_tosa_files)
        self.assertTrue(output_json_file_exists)

        # Find the largest .tosa file to validate correct file output
        largest_file_size = 0
        for tosa_file in output_tosa_files:
            file_size = Path(tosa_file).stat().st_size
            if file_size > largest_file_size:
                largest_file_size = file_size

        self.assertGreater(
            largest_file_size,
            100_000,
            "TOSA file generated incorrectly "
            f"as per the small file size of {largest_file_size / 1000} KB. ",
        )

        self.assertLess(
            largest_file_size,
            1_000_000,
            "TOSA file generated incorrectly  "
            f"as per the large file size of {largest_file_size / 1000} KB. ",
        )

        # Compare .tosa file sizes between qat int8 model and
        # fp32 model if fp32 test ran successfully
        if TestExecuTorchIntegration.fp32_file_size is not None:
            qat_int8_file_size = largest_file_size
            model_size_ratio = (
                qat_int8_file_size / TestExecuTorchIntegration.fp32_file_size
            )

            # Ensure int8 model is ~1/4 size of fp32 model
            self.assertAlmostEqual(
                model_size_ratio,
                0.25,
                delta=0.1,
                msg="int8 TOSA file from QAT is not ~1/4 size of fp32 TOSA file.",
            )

        self.validate_vgf_export()

    def test_export_function_raises_error_missing_dataset_path(self):
        """Test export raises error if missing dataset path"""
        params = create_simple_params()
        params.output.export.vgf_output_dir = self.tosa_out_dir

        # Ensure tosa output directory does not exist
        tosa_out_dir = Path(params.output.export.vgf_output_dir)
        self.assertFalse(tosa_out_dir.exists())

        # Remove all paths in the config file
        params.dataset.path.train = None
        params.dataset.path.validation = None
        params.dataset.path.test = None

        # Point to pretrained weights to load
        qat_model_path = "tests/usecases/nss/weights/nss_v0.1.1_int8.pt"

        fp32_model_path = "tests/usecases/nss/weights/nss_v0.1.0_fp32.pt"

        with self.assertRaises(ValueError) as exc:
            do_export(params, qat_model_path, ExportType.QAT_INT8)

        self.assertIn("Config error", str(exc.exception))

        with self.assertRaises(ValueError) as exc:
            do_export(params, fp32_model_path, ExportType.FP32)

        self.assertIn("Config error", str(exc.exception))

        with self.assertRaises(ValueError) as exc:
            do_export(params, fp32_model_path, ExportType.PTQ_INT8)

        self.assertIn("Config error", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
