# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import argparse
import copy
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from ng_model_gym.core.data import DataLoaderMode, generic_safetensors_reader
from ng_model_gym.usecases.nss.data.dataset import NSSDataset
from scripts.safetensors_generator.dataset_readers import (
    NFRUEXRDatasetReader,
    NSSEXRDatasetReader,
)
from scripts.safetensors_generator.exr_utils import read_exr_torch
from scripts.safetensors_generator.safetensors_writer import (
    generic_safetensors_writer,
    GrowingTensorBuffer,
)
from tests.testing_utils import create_simple_params
from tests.usecases.nss.unit.data.camera_cut_builders import compute_expected_segments

# pylint:disable=too-many-lines


class TestNSSSafetensorsWriter(unittest.TestCase):
    """Test output .safetensors file generated from .exr"""

    def setUp(self):
        """Set up test"""
        self.seq_path = Path("0002")
        self.seq_id = 0

        self.args = argparse.Namespace()
        self.args.src = Path("tests/datasets/test_nss_exr")
        self.args.dst = Path("tests/datasets")
        self.args.dst_root = self.args.dst / Path(self.args.src.parts[-1])
        self.args.threads = 1
        self.args.extension = "exr"
        self.args.overwrite = False
        self.args.linear_truth = True
        self.args.reader = "NSSv1_0_1"

        self.output_path = Path(self.args.dst_root / f"{self.seq_path}.safetensors")
        self.crop_output_root = Path(self.args.dst_root / f"{self.seq_path}")

        self.nss_reader = NSSEXRDatasetReader(
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

    def _clone_args(self):
        return argparse.Namespace(**vars(self.args))

    def _apply_camera_cut_metadata(
        self, dataset_root: Path, camera_cut_flags, *, seq_path: Path | None = None
    ):
        seq_path = seq_path or self.seq_path
        json_path = dataset_root / f"{seq_path}.json"
        with open(json_path, "r", encoding="utf-8") as src_file:
            metadata = json.load(src_file)
        frames = metadata.get("Frames", [])
        if len(frames) < len(camera_cut_flags):
            raise AssertionError("CameraCut flags exceed available metadata frames")
        for idx, frame in enumerate(frames):
            flag = bool(camera_cut_flags[idx]) if idx < len(camera_cut_flags) else False
            frame["CameraCut"] = flag
        with open(json_path, "w", encoding="utf-8") as dst_file:
            json.dump(metadata, dst_file, indent=2)

    @staticmethod
    def _writer_output_root(args: argparse.Namespace) -> Path:
        """Mirror SafetensorsWriter dst layout."""
        return Path(args.dst) / Path(args.src).name

    def test_compare_saved_tensors(self):
        """Test tensor values, dtype and shape are preserved."""
        generic_safetensors_writer(self.args)

        self.assertTrue(
            self.output_path.exists(), msg=f"{self.output_path} failed to be created."
        )

        iterator = iter(self.nss_reader)
        frame = next(iterator)
        dst_file_path, features = frame[0]

        sf_path = (self.args.dst_root / dst_file_path).with_suffix(".safetensors")

        # Create test reference container and Sequence Descriptor
        test_reference = {k: v.detach().clone() for k, v in features.items()}

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

    def test_reverse_z_depth_is_inverted(self):
        """Ensure ReverseZ datasets store inverted depth."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            reverse_src = tmp_dir_path / "reverse_z_src"
            shutil.copytree(self.args.src, reverse_src)

            metadata_path = reverse_src / f"{self.seq_path}.json"
            with open(metadata_path, "r", encoding="utf-8") as src_file:
                metadata = json.load(src_file)
            metadata["ReverseZ"] = True
            with open(metadata_path, "w", encoding="utf-8") as dst_file:
                json.dump(metadata, dst_file, indent=2)

            reverse_args = argparse.Namespace(**vars(self.args))
            reverse_args.src = reverse_src
            reverse_args.dst = tmp_dir_path / "reverse_z_dst"
            reverse_args.dst.mkdir(parents=True, exist_ok=True)
            reverse_args.dst_root = reverse_args.dst / Path(reverse_args.src.parts[-1])

            generic_safetensors_writer(reverse_args)

            depth_exr_path = sorted(
                (reverse_src / "x2" / "depth" / self.seq_path).glob("*.exr")
            )[0]
            original_depth = read_exr_torch(
                depth_exr_path,
                dtype=np.float32,
                channels="R",
            ).to(torch.float32)
            expected_depth = 1.0 - original_depth

            output_safetensors = reverse_args.dst_root / f"{self.seq_path}.safetensors"
            self.assertTrue(
                output_safetensors.exists(),
                msg=f"{output_safetensors} was not created for ReverseZ dataset.",
            )

            written = generic_safetensors_reader(output_safetensors, 0)
            self.assertTrue(
                bool(written["ReverseZ"].item()),
                msg="ReverseZ flag not persisted in written safetensors.",
            )
            self.assertTrue(
                torch.allclose(written["depth"], expected_depth, atol=1e-6),
                msg="Depth tensor was not inverted for ReverseZ dataset.",
            )

    def test_camera_cut_tensor_persisted(self):
        """Writer copies CameraCut flags from JSON into the uncropped safetensor tensor."""
        flags = [True, False, False, False, False, True]
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "src"
            dst = Path(tmp_dir) / "dst"
            shutil.copytree(self.args.src, src)
            args = self._clone_args()
            args.src = src
            args.dst = dst
            self._apply_camera_cut_metadata(src, flags)

            generic_safetensors_writer(args)

            output_root = self._writer_output_root(args)
            output_path = output_root / f"{self.seq_path}.safetensors"
            with safe_open(output_path, framework="pt") as written:
                self.assertIn("camera_cut", list(written.keys()))
                tensor = written.get_tensor("camera_cut").view(-1)
                self.assertEqual(tensor.dtype, torch.bool)
                self.assertGreaterEqual(tensor.shape[0], 1)
                self.assertLessEqual(tensor.shape[0], len(flags))
                expected = flags[: tensor.shape[0]]
                self.assertEqual(tensor.tolist(), expected)

    def test_camera_cut_tensor_persisted_in_crops(self):
        """Cropper propagates per-frame camera_cut flags even when multiple frames exist."""
        seq_path = self.seq_path
        source_root = self.args.src
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "src"
            dst = Path(tmp_dir) / "dst"
            shutil.copytree(source_root, src)
            camcut_metadata = src / f"{seq_path}_camcut.json"
            seq_metadata = src / f"{seq_path}.json"
            self.assertTrue(
                camcut_metadata.exists(),
                msg=f"{camcut_metadata} missing camera-cut metadata fixture.",
            )
            shutil.copyfile(camcut_metadata, seq_metadata)

            with open(seq_metadata, encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            frames = metadata.get("Frames", [])
            self.assertGreater(
                len(frames),
                0,
                msg="Camera-cut fixtures do not have any frame metadata.",
            )
            expected_flags = [bool(frame.get("CameraCut", False)) for frame in frames]

            args = self._clone_args()
            args.src = src
            args.dst = dst

            generic_safetensors_writer(args)

            output_root = self._writer_output_root(args)

            crop_args = self._clone_args()
            crop_args.src = output_root
            crop_args.dst = args.dst
            crop_args.reader = "cropper"
            crop_args.extension = "safetensors"
            crop_args.crop_size = 256

            generic_safetensors_writer(crop_args)

            crop_root = self._writer_output_root(crop_args)
            seq_crop_root = crop_root / seq_path
            seen_flags: dict[int, bool] = {}
            for crop_dir in sorted(seq_crop_root.iterdir()):
                safepath = crop_dir / f"{seq_path}.safetensors"
                if not safepath.exists():
                    continue
                with safe_open(safepath, framework="pt") as cropped:
                    frame_idx = int(cropped.get_tensor("img").view(-1)[0])
                    if frame_idx in seen_flags:
                        continue
                    seen_flags[frame_idx] = bool(
                        cropped.get_tensor("camera_cut").view(-1)[0]
                    )
                if len(seen_flags) >= len(expected_flags):
                    break

            self.assertGreaterEqual(len(seen_flags), 1)
            self.assertLessEqual(len(seen_flags), len(expected_flags))
            ordered_indices = sorted(seen_flags.keys())
            self.assertEqual(ordered_indices, list(range(len(ordered_indices))))
            ordered = [seen_flags[idx] for idx in ordered_indices]
            expected = expected_flags[: len(ordered)]
            self.assertEqual(ordered, expected)

    def test_nss_dataset_respects_writer_camera_cut(self):
        """Dataset instantiated on writer output obeys the injected camera-cut boundaries."""
        flags = [True]
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "src"
            dst = Path(tmp_dir) / "dst"
            shutil.copytree(self.args.src, src)
            args = self._clone_args()
            args.src = src
            args.dst = dst
            self._apply_camera_cut_metadata(src, flags)

            generic_safetensors_writer(args)

            output_root = self._writer_output_root(args)

            params = create_simple_params(usecase="nss", dataset_path=str(output_root))
            params.dataset.path.test = output_root
            dataset = NSSDataset(params, DataLoaderMode.TEST)

            capture = dataset.captures[0]
            expected_segments = compute_expected_segments(
                flags, dataset.recurrent_samples
            )
            self.assertEqual(dataset.capture_sequences[capture], expected_segments)
            self.assertEqual(dataset.capture_windows[capture], [(0, 1)])

    def test_generated_safetensor_contains_all_features(self):
        """Test that all expected features are present in the generated .safetensors file."""
        generic_safetensors_writer(self.args)

        self.assertTrue(
            self.output_path.exists(), msg=f"{self.output_path} failed to be created."
        )

        expected_features = {
            "EmulatedFramerate",
            "FovX",
            "FovY",
            "ReverseZ",
            "Samples",
            "TargetResolution",
            "X",
            "Y",
            "camera_cut",
            "colour_linear",
            "depth",
            "depth_params",
            "exposure",
            "ground_truth_linear",
            "img",
            "infinite_zFar",
            "jitter",
            "motion",
            "motion_lr",
            "outDims",
            "render_size",
            "scale",
            "seq",
            "viewProj",
            "zFar",
            "zNear",
        }

        with safe_open(self.output_path, framework="pt") as written:
            actual_features = set(written.keys())
            missing = expected_features - actual_features
            unexpected = actual_features - expected_features
            self.assertFalse(
                missing,
                msg=f"Missing expected features in {self.output_path}: {missing}",
            )
            self.assertFalse(
                unexpected,
                msg=f"Unexpected features found in {self.output_path}: {unexpected}",
            )


@unittest.skip("NFRU EXR fixtures removed pending refreshed writer coverage")
class TestNFRUSafetensorsWriter(unittest.TestCase):
    """Test output .safetensors file generated from NFRU .exr"""

    _NFRU_EXR_SUBDIRS = (
        "ground_truth",
        "ground_truth_final",
        "motion_gt_m1",
        "motion_gt_m2",
        "x2/depth",
        "x2/motion_m1",
        "x2/motion_m2",
    )

    @staticmethod
    def _make_writer_args(src: Path, dst: Path) -> argparse.Namespace:
        """Construct default writer arguments for NFRU tests."""
        args = argparse.Namespace()
        args.src = src
        args.dst = dst
        args.dst_root = dst / src.name
        args.threads = 1
        args.extension = "exr"
        args.overwrite = False
        args.linear_truth = True
        args.reader = "NFRUv2_2"
        return args

    @classmethod
    def _trim_nfru_fixture_to_frame_count(cls, src_root: Path, frame_count: int):
        """Keep only the first `frame_count` frames in a copied NFRU fixture."""
        metadata_path = src_root / "0000.json"
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        metadata["Frames"] = metadata["Frames"][:frame_count]
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

        for subdir in cls._NFRU_EXR_SUBDIRS:
            exr_dir = src_root / subdir / "0000"
            for exr_file in exr_dir.glob("*.exr"):
                if int(exr_file.stem) >= frame_count:
                    exr_file.unlink()
                    exr_file.with_name(exr_file.name + ".license").unlink(
                        missing_ok=True
                    )

    @staticmethod
    def _clone_frame(frame: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Detach and clone tensors from a saved frame dict."""
        return {k: v.detach().clone() for k, v in frame.items()}

    @classmethod
    def setUpClass(cls):
        """Build a reusable 1-frame fixture for fast NFRU writer assertions."""
        super().setUpClass()
        cls._fixtures_tmp_root = Path(tempfile.mkdtemp())

        source_root = Path("tests/datasets/test_nfru_exr")

        cls._metadata_src = cls._fixtures_tmp_root / "nfru_src_metadata"
        shutil.copytree(source_root, cls._metadata_src)
        cls._trim_nfru_fixture_to_frame_count(cls._metadata_src, frame_count=1)

        cls._one_frame_args = cls._make_writer_args(
            src=cls._metadata_src, dst=cls._fixtures_tmp_root / "one_frame_dst"
        )

        generic_safetensors_writer(cls._one_frame_args)
        cls._one_frame_output_path = cls._one_frame_args.dst_root / "0000.safetensors"
        cls._one_frame_frame_zero = generic_safetensors_reader(
            cls._one_frame_output_path, 0
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up class-level temporary output."""
        shutil.rmtree(cls._fixtures_tmp_root, ignore_errors=True)
        super().tearDownClass()

    def setUp(self):
        """Set up test"""
        self.seq_path = Path("0000")
        self.seq_id = 0

        self.args = argparse.Namespace(**vars(self._one_frame_args))
        self.args.dst_root = self._writer_output_root(self.args)

        self.nfru_reader = NFRUEXRDatasetReader(
            src_root=self.args.src,
            dst_root=self.args.dst_root,
            seq_id=self.seq_id,
            seq_path=self.seq_path,
            args=self.args,
        )

    def _clone_args(self, src: Path | None = None, dst: Path | None = None):
        args = argparse.Namespace(**vars(self.args))
        if src is not None:
            args.src = src
        if dst is not None:
            args.dst = dst
        args.dst_root = self._writer_output_root(args)
        return args

    @staticmethod
    def _writer_output_root(args: argparse.Namespace) -> Path:
        return Path(args.dst) / Path(args.src).name

    def _write_and_read_frame_zero(
        self, src: Path, dst: Path
    ) -> dict[str, torch.Tensor]:
        """
        Write a Safetensors file based upon the args attribute. Return the
        first frame of the written file, as a dict of the keys and values
        making up the frame.
        """
        args = self._clone_args(src=src, dst=dst)
        generic_safetensors_writer(args)
        output_path = self._writer_output_root(args) / f"{self.seq_path}.safetensors"
        return generic_safetensors_reader(output_path, 0)

    def test_compare_saved_tensors(self):
        """Test tensor values, dtype and shape are preserved for NFRU inputs."""
        self.assertTrue(
            self._one_frame_output_path.exists(),
            msg=f"{self._one_frame_output_path} failed to be created.",
        )

        iterator = iter(self.nfru_reader)
        frame = next(iterator)
        dst_file_path, features = frame[0]

        sf_path = (self.args.dst_root / dst_file_path).with_suffix(".safetensors")

        # Get calculated dataset
        test_reference = {k: v.detach().clone() for k, v in features.items()}
        # For comparison, get dataset from saved .safetensors
        test_target = generic_safetensors_reader(sf_path, 0)

        mse_fn = lambda y, x: torch.mean((y - x) ** 2)
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
                mse = mse_fn(test_target_parsed, test_ref)
                self.assertTrue(
                    np.isclose(0.0, mse.numpy(), atol=1e-6),
                    msg=f"MSE higher than expected for {name}, dtype:{test_ref.dtype} value: {mse}",
                )
            else:
                print(
                    f"Name: {name}, Value: {test_target_parsed} "
                    f"(Reference: {test_ref}), Test Skipped"
                )

    def test_output_omits_offline_flow_and_dilated_hint_tensors(self):
        """NFRU writer should not persist removed offline flow or dilated hint tensors."""
        self.assertTrue(
            self._one_frame_output_path.exists(),
            msg=f"{self._one_frame_output_path} failed to be created.",
        )

        data = generic_safetensors_reader(self._one_frame_output_path, 0)
        removed_tensor_names = [
            "flow_{}_f30_m1@blockmatch_v3",
            "flow_{}_f30_p1@blockmatch_v3",
            "flow_{}_f60_m1@blockmatch_v3",
            "flow_{}_f60_p1@blockmatch_v3",
            "dilated_sy_{}_f30_m1",
            "dilated_sy_{}_f30_p1",
        ]

        for tensor_name in removed_tensor_names:
            self.assertNotIn(
                tensor_name,
                data,
                msg=f"Removed offline tensor '{tensor_name}' should not be written.",
            )

        retained_synthetic_motion_names = [
            "sy_{}_f30_m1",
            "sy_{}_f30_p1",
            "sy_{}_f60_m1",
            "sy_{}_f60_p1",
        ]

        for tensor_name in retained_synthetic_motion_names:
            self.assertIn(
                tensor_name,
                data,
                msg=f"Missing retained synthetic motion tensor '{tensor_name}'.",
            )

    def test_inverse_y_metadata_changes_written_viewproj(self):
        """Toggling JSON InverseY must update InverseY and ViewProj tensors."""
        baseline = self._clone_frame(self._one_frame_frame_zero)
        baseline_inverse_y = int(baseline["InverseY"].item())
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "src"
            shutil.copytree(self.args.src, src)
            metadata_path = src / f"{self.seq_path}.json"

            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)

                metadata["InverseY"] = not bool(baseline_inverse_y)
            with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                json.dump(metadata, metadata_file, indent=2)

            modified_args = self._clone_args(
                src=src,
                dst=Path(tmp_dir) / "dst_modified",
            )
            generic_safetensors_writer(modified_args)
            modified_output = (
                self._writer_output_root(modified_args) / f"{self.seq_path}.safetensors"
            )
            inverse = generic_safetensors_reader(modified_output, 0)

            self.assertIn("InverseY", baseline)
            self.assertIn("InverseY", inverse)
            self.assertEqual(int(baseline["InverseY"].item()), baseline_inverse_y)
            self.assertEqual(
                int(inverse["InverseY"].item()), int(not baseline_inverse_y)
            )
            self.assertFalse(
                torch.allclose(baseline["ViewProj"], inverse["ViewProj"]),
                msg="ViewProj should change when InverseY metadata toggles.",
            )

    def test_upscaling_ratio_index_selects_jitter_entry(self):
        """x2_index must select the matching jitter entry from JSON metadata."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "src"
            shutil.copytree(self.args.src, src)
            metadata_path = src / f"{self.seq_path}.json"

            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)

            target_x = 0.25
            target_y = -0.5
            metadata["UpscalingRatiosIndices"]["x2_index"] = 1
            for frame in metadata["Frames"]:
                frame["NormalizedPerRatioJitter"] = [
                    {"X": 0.0, "Y": 0.0},
                    {"X": target_x, "Y": target_y},
                ]
            with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                json.dump(metadata, metadata_file, indent=2)

            args = self._clone_args(src=src, dst=Path(tmp_dir) / "dst")
            generic_safetensors_writer(args)

            output_path = (
                self._writer_output_root(args) / f"{self.seq_path}.safetensors"
            )
            data = generic_safetensors_reader(output_path, 0)
            self.assertIn("jitter", data)
            _, render_h, render_w = data["rgb_linear"].shape
            expected_jitter = torch.tensor(
                [target_y * render_h, target_x * render_w], dtype=torch.float32
            ).reshape(2, 1, 1)
            torch.testing.assert_close(data["jitter"], expected_jitter)

    def test_root_depth_plane_metadata_overrides_frame_planes(self):
        """Root NearPlane/FarPlane metadata must drive written plane tensors."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "src"
            shutil.copytree(self.args.src, src)
            metadata_path = src / f"{self.seq_path}.json"

            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)

            metadata["NearPlane"] = 7.0
            metadata["FarPlane"] = 1234.0
            metadata["Frames"][0]["CameraNearPlane"] = 0.25
            metadata["Frames"][0]["CameraFarPlane"] = 0.5
            with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                json.dump(metadata, metadata_file, indent=2)

            args = self._clone_args(src=src, dst=Path(tmp_dir) / "dst")
            generic_safetensors_writer(args)

            output_path = (
                self._writer_output_root(args) / f"{self.seq_path}.safetensors"
            )
            data = generic_safetensors_reader(output_path, 0)
            self.assertIn("NearPlane", data)
            self.assertIn("FarPlane", data)
            self.assertIn("infinite_zFar", data)
            self.assertEqual(float(data["NearPlane"].item()), 7.0)
            self.assertEqual(float(data["FarPlane"].item()), 1234.0)
            self.assertFalse(bool(data["infinite_zFar"].item()))

    def test_frame_depth_plane_metadata_used_when_root_planes_missing(self):
        """Frame CameraNear/FarPlane metadata should be used when root planes are absent."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "src"
            shutil.copytree(self.args.src, src)
            metadata_path = src / f"{self.seq_path}.json"

            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)

            metadata.pop("NearPlane", None)
            metadata.pop("FarPlane", None)
            metadata["Frames"][0]["CameraNearPlane"] = 3.0
            metadata["Frames"][0]["CameraFarPlane"] = -1
            with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                json.dump(metadata, metadata_file, indent=2)

            args = self._clone_args(src=src, dst=Path(tmp_dir) / "dst")
            generic_safetensors_writer(args)

            output_path = (
                self._writer_output_root(args) / f"{self.seq_path}.safetensors"
            )
            data = generic_safetensors_reader(output_path, 0)
            self.assertIn("NearPlane", data)
            self.assertIn("FarPlane", data)
            self.assertIn("infinite_zFar", data)
            self.assertEqual(float(data["NearPlane"].item()), 3.0)
            self.assertEqual(float(data["FarPlane"].item()), 5000.0)
            self.assertTrue(bool(data["infinite_zFar"].item()))

    def test_fovx_metadata_changes_written_fovx(self):
        """Changing frame FovX in JSON must change the written FovX tensor."""
        baseline = self._clone_frame(self._one_frame_frame_zero)
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "src"
            shutil.copytree(self.args.src, src)
            metadata_path = src / f"{self.seq_path}.json"
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)

            modified_metadata = copy.deepcopy(metadata)
            modified_metadata["Frames"][0]["FovX"] = 0.9
            with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                json.dump(modified_metadata, metadata_file, indent=2)

            modified = self._write_and_read_frame_zero(src, Path(tmp_dir) / "dst_mod")

            self.assertIn("FovX", baseline)
            self.assertIn("FovX", modified)
            self.assertFalse(
                torch.allclose(baseline["FovX"], modified["FovX"]),
                msg="FovX should change when frame FovX metadata changes.",
            )

    def test_fovy_metadata_changes_written_fovy_and_depth_params(self):
        """Changing frame FovY in JSON must change written FovY and DepthParams tensors."""
        baseline = self._clone_frame(self._one_frame_frame_zero)
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "src"
            shutil.copytree(self.args.src, src)
            metadata_path = src / f"{self.seq_path}.json"
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)

            modified_metadata = copy.deepcopy(metadata)
            modified_metadata["Frames"][0]["FovY"] = 0.75
            with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                json.dump(modified_metadata, metadata_file, indent=2)

            modified = self._write_and_read_frame_zero(src, Path(tmp_dir) / "dst_mod")

            self.assertIn("FovY", baseline)
            self.assertIn("FovY", modified)
            self.assertFalse(
                torch.allclose(baseline["FovY"], modified["FovY"]),
                msg="FovY should change when frame FovY metadata changes.",
            )
            self.assertIn("DepthParams", baseline)
            self.assertIn("DepthParams", modified)
            self.assertFalse(
                torch.allclose(baseline["DepthParams"], modified["DepthParams"]),
                msg="DepthParams should change when frame FovY metadata changes.",
            )

    def test_view_projection_metadata_changes_written_viewproj(self):
        """Changing frame ViewProjection in JSON must change the written ViewProj tensor."""
        baseline = self._clone_frame(self._one_frame_frame_zero)
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "src"
            shutil.copytree(self.args.src, src)
            metadata_path = src / f"{self.seq_path}.json"
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)

            modified_metadata = copy.deepcopy(metadata)
            modified_metadata["Frames"][0]["ViewProjection"][0] += 0.25
            with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                json.dump(modified_metadata, metadata_file, indent=2)

            modified = self._write_and_read_frame_zero(src, Path(tmp_dir) / "dst_mod")

            self.assertIn("ViewProj", baseline)
            self.assertIn("ViewProj", modified)
            self.assertFalse(
                torch.allclose(baseline["ViewProj"], modified["ViewProj"]),
                msg="ViewProj should change when frame ViewProjection metadata changes.",
            )

    def test_exposure_metadata_changes_written_exposure_only(self):
        """Changing frame Exposure in JSON must change exposure tensor but not tonemapped rgb."""
        baseline = self._clone_frame(self._one_frame_frame_zero)
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "src"
            shutil.copytree(self.args.src, src)
            metadata_path = src / f"{self.seq_path}.json"
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)

            modified_metadata = copy.deepcopy(metadata)
            modified_metadata["Frames"][0]["Exposure"] = 2.75
            with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                json.dump(modified_metadata, metadata_file, indent=2)

            modified = self._write_and_read_frame_zero(src, Path(tmp_dir) / "dst_mod")

            self.assertIn("exposure", baseline)
            self.assertIn("exposure", modified)
            self.assertFalse(
                torch.allclose(baseline["exposure"], modified["exposure"]),
                msg="exposure tensor should change when frame Exposure metadata changes.",
            )
            self.assertIn("rgb_reinhard", baseline)
            self.assertIn("rgb_reinhard", modified)
            self.assertTrue(
                torch.allclose(baseline["rgb_reinhard"], modified["rgb_reinhard"]),
                msg=(
                    "Tonemapped rgb is expected to remain unchanged: "
                    "NFRUEXRDatasetReader applies a fixed internal exposure value."
                ),
            )


class TestGrowingTensorBuffer(unittest.TestCase):
    """Unit tests for tensor append and growth behavior."""

    def test_init_rejects_scalar_tensor(self):
        """A scalar tensor cannot be appended on dim 0."""
        with self.assertRaises(ValueError) as e:
            GrowingTensorBuffer(torch.tensor(1.0))
        self.assertIn(
            "Expected tensor with at least 1 dimension for dim-0 append",
            str(e.exception),
        )

    def test_append_grows_and_preserves_content(self):
        """Appending beyond capacity should grow storage and keep all values."""
        first = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        to_append = torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)

        buffer = GrowingTensorBuffer(first)
        self.assertEqual(buffer.length, 1)
        self.assertEqual(buffer.capacity, 1)

        buffer.append(to_append)
        self.assertEqual(buffer.length, 3)
        # Buffer capacity should double each time that it grows
        # After appending 2 more rows to a capacity of 1, it should have grown to at least 3,
        # which means it should have doubled twice (1 -> 2 -> 4)
        self.assertEqual(buffer.capacity, 4)

        expected = torch.cat([first, to_append], dim=0)
        torch.testing.assert_close(buffer.as_tensor(), expected)

    def test_append_zero_length_tensor_keeps_existing_data(self):
        """Appending a 0-length batch should be a no-op."""
        first = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        empty = torch.empty((0, 2), dtype=torch.float32)

        buffer = GrowingTensorBuffer(first)
        before_length = buffer.length
        before_capacity = buffer.capacity
        before = buffer.as_tensor().clone()

        buffer.append(empty)

        self.assertEqual(buffer.length, before_length)
        self.assertEqual(buffer.capacity, before_capacity)
        torch.testing.assert_close(buffer.as_tensor(), before)

    def test_append_rejects_mismatched_dtype(self):
        """Appending with a different dtype should fail fast."""
        buffer = GrowingTensorBuffer(torch.ones((1, 2), dtype=torch.float32))
        with self.assertRaises(ValueError) as e:
            buffer.append(torch.ones((1, 2), dtype=torch.float16))
        self.assertIn(
            "Mismatched dtype while appending tensor buffer", str(e.exception)
        )

    def test_append_rejects_mismatched_shape(self):
        """Appending with a different tail shape should fail fast."""
        buffer = GrowingTensorBuffer(torch.ones((1, 2), dtype=torch.float32))
        with self.assertRaises(ValueError) as e:
            buffer.append(torch.ones((1, 3), dtype=torch.float32))
        self.assertIn(
            "Mismatched shape while appending tensor buffer", str(e.exception)
        )

    def test_append_rejects_mismatched_device(self):
        """Appending with a different device should fail fast."""
        buffer = GrowingTensorBuffer(
            torch.ones((1, 2), device="cpu", dtype=torch.float32)
        )

        try:
            cuda_tensor = torch.empty((1, 2), device="cuda", dtype=torch.float32)
        except (RuntimeError, AssertionError):
            self.skipTest("CUDA device not available in this torch build")

        with self.assertRaises(ValueError) as e:
            buffer.append(cuda_tensor)
        self.assertIn(
            "Mismatched device while appending tensor buffer", str(e.exception)
        )

    def test_initial_zero_length_input_supported(self):
        """A zero-length first tensor should initialize and append correctly."""
        buffer = GrowingTensorBuffer(torch.empty((0, 2), dtype=torch.float32))
        self.assertEqual(buffer.length, 0)
        self.assertEqual(buffer.capacity, 1)
        self.assertEqual(tuple(buffer.as_tensor().shape), (0, 2))

        later = torch.tensor([[7.0, 8.0], [9.0, 10.0]], dtype=torch.float32)
        buffer.append(later)

        self.assertEqual(buffer.length, 2)
        self.assertEqual(buffer.capacity, 2)
        torch.testing.assert_close(buffer.as_tensor(), later)
