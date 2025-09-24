# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import argparse
import unittest
from pathlib import Path

import numpy as np
import safetensors
import torch

from scripts.safetensors_generator.dataset_reader import NSSEXRDatasetReader
from scripts.safetensors_generator.safetensors_writer import generic_safetensors_writer


def generic_safetensors_reader(seq_path: Path, idx: int) -> dict:
    """Safetensors reader to return a dictionary of tensors"""
    data_frame = {}
    with safetensors.safe_open(seq_path, framework="numpy", device="cpu") as f:
        for k in f.keys():
            data_frame[k] = torch.from_numpy(f.get_slice(k)[idx])

    return data_frame


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

        self.output_path = Path(self.args.dst_root / f"{self.seq_path}.safetensors")

        self.reader = NSSEXRDatasetReader(
            src_root=self.args.src,
            dst_root=self.args.dst_root,
            seq_id=self.seq_id,
            seq_path=self.seq_path,
            args=self.args,
        )

    def tearDown(self):
        """Clean up after test"""
        if self.output_path.exists():
            self.output_path.unlink()

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


if __name__ == "__main__":
    unittest.main()
