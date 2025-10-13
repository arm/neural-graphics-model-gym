# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import argparse
import math
import shutil
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.safetensors_generator.dataset_reader import (
    generic_safetensors_reader,
    NSSEXRDatasetReader,
)
from scripts.safetensors_generator.safetensors_writer import generic_safetensors_writer


class TestSafetensorsWriter(unittest.TestCase):
    """Test output .safetensors file generated from .exr"""

    def setUp(self):
        """Set up test"""
        self.seq_path = Path("0002")
        self.seq_id = 0

        self.args = argparse.Namespace()
        self.args.src = Path("tests/datasets/test_exr")
        self.args.dst = Path("tests/datasets")
        self.args.dst_root = self.args.dst / Path(self.args.src.parts[-1])
        self.args.threads = 1
        self.args.extension = "exr"
        self.args.overwrite = False
        self.args.linear_truth = True
        self.args.reader = "EXRv101"

        self.output_path = Path(self.args.dst_root / f"{self.seq_path}.safetensors")
        self.crop_output_root = Path(self.args.dst_root / f"{self.seq_path}")

        self.reader = NSSEXRDatasetReader(
            src_root=self.args.src,
            dst_root=self.args.dst_root,
            seq_id=self.seq_id,
            seq_path=self.seq_path,
            args=self.args,
        )

    def tearDown(self):
        """Clean up after test"""

        # Remove uncropped safetensors
        if self.output_path.exists():
            self.output_path.unlink()

        # Remove cropped safetensors
        if self.crop_output_root.exists():
            shutil.rmtree(self.crop_output_root)

    def test_compare_saved_tensors(self):
        """Test tensor values, dtype and shape are preserved."""
        generic_safetensors_writer(self.args)

        self.assertTrue(
            self.output_path.exists(), msg=f"{self.output_path} failed to be created."
        )

        iterator = iter(self.reader)
        frame = next(iterator)
        dst_file_path, features = frame[0]

        sf_path = (self.args.dst_root / dst_file_path).with_suffix(".safetensors")

        # Create test reference container and Sequence Descriptor
        test_reference = {k: torch.tensor(v) for k, v in features.items()}

        # Create Reference Dataset from saved .safetensors
        test_target = generic_safetensors_reader(sf_path, 0)

        MSE = lambda y, x: torch.mean((y - x) ** 2)
        for name, test_ref in test_reference.items():
            test_target_parsed = test_target[name]

            # Check loaded tensor has same data type as original
            self.assertEqual(
                test_target_parsed.dtype,
                test_ref.dtype,
                msg=f"{name} data type mismatch.",
            )

            if test_ref.dtype not in [torch.int64, torch.int32, torch.bool]:
                # Check MSE between loaded tensor and original
                mse = MSE(test_target_parsed, test_ref)

                self.assertTrue(
                    np.isclose(0.0, mse.numpy(), atol=1e-6),
                    msg=f"MSE higher than expected for {name}, dtype:{test_ref.dtype} value: {mse}",
                )

            else:
                print(
                    f"Name: {name}, Value: {test_target_parsed} "
                    f"(Reference: {test_ref}), Test Skipped"
                )

    def test_safetensors_crop_number(self):
        """Test cropper produces correct number of cropped safetensors"""

        # Run with normal reader first to produce uncropped safetensors
        generic_safetensors_writer(self.args)

        self.args.reader = "cropper"
        self.args.crop_size = 256
        self.args.extension = "safetensors"

        self.crop_output_path = Path(self.args.dst_root / f"{self.seq_path}")

        width, height = 1920, 1080

        horizontal_crops = math.ceil(width / self.args.crop_size)
        vertical_crops = math.ceil(height / self.args.crop_size)

        total_crops = horizontal_crops * vertical_crops

        # Run with the cropper
        generic_safetensors_writer(self.args)

        # Count subdirectories created in output dir, containing .safetensors
        safetensors_dirs = sum(
            1
            for d in self.crop_output_path.iterdir()
            if d.is_dir() and any(f.suffix == ".safetensors" for f in d.iterdir())
        )

        self.assertEqual(total_crops, safetensors_dirs)

    def test_tensor_shapes_cropped(self):
        """Test tensor shapes of cropped safetensors"""

        # Run with normal reader first to produce uncropped safetensors
        generic_safetensors_writer(self.args)

        self.args.reader = "cropper"
        self.args.crop_size = 256
        self.args.extension = "safetensors"

        self.crop_output_path = Path(self.args.dst_root / f"{self.seq_path}")

        # Run with the cropper
        generic_safetensors_writer(self.args)

        # Check values match expected for first cropped .safetensors
        first_cropped_safetensors = (
            self.crop_output_path / "0000" / f"{self.seq_path}.safetensors"
        )

        data = generic_safetensors_reader(first_cropped_safetensors, 0)

        self.assertEqual(data["crop_id_x"], 0)
        self.assertEqual(data["crop_id_y"], 0)
        self.assertEqual(data["crop_sz"], self.args.crop_size)

        # Check values match expected for last cropped .safetensors
        width, height = 1920, 1080

        horizontal_crops = math.ceil(width / self.args.crop_size)
        vertical_crops = math.ceil(height / self.args.crop_size)

        total_crops = horizontal_crops * vertical_crops

        last_cropped_safetensors = (
            self.crop_output_path
            / f"{total_crops - 1:04d}"
            / f"{self.seq_path}.safetensors"
        )

        data = generic_safetensors_reader(last_cropped_safetensors, 0)

        self.assertEqual(data["crop_id_x"], horizontal_crops - 1)
        self.assertEqual(data["crop_id_y"], vertical_crops - 1)
        self.assertEqual(data["crop_sz"], self.args.crop_size)

        # Check height and width of a full resolution tensor after cropping
        _, gt_h, gt_w = data["ground_truth_linear"].shape

        self.assertEqual(gt_h, self.args.crop_size)
        self.assertEqual(gt_w, self.args.crop_size)

        # Check height and width of scaled tensor after cropping
        _, colour_h, colour_w = data["colour_linear"].shape
        scale = data["scale"]

        self.assertEqual(colour_h, math.ceil(self.args.crop_size / scale))
        self.assertEqual(colour_w, math.ceil(self.args.crop_size / scale))


if __name__ == "__main__":
    unittest.main()
