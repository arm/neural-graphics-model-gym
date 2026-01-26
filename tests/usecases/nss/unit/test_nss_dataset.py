# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import random
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ng_model_gym.core.data import DataLoaderMode, DatasetType
from ng_model_gym.core.utils.general_utils import create_directory
from ng_model_gym.usecases.nss.data.dataset import NSSDataset
from tests.testing_utils import create_simple_params
from tests.usecases.nss.unit.data.camera_cut_builders import write_camera_cut_fixture


class TestNSSDataset(unittest.TestCase):
    """Test NSSDataset class"""

    def setUp(self):
        """Set up test"""
        self.params = create_simple_params(
            dataset="./tests/usecases/nss/datasets/train"
        )
        self.params.dataset.recurrent_samples = 4

    def test_existing_safetensors_file(self):
        """Test loading existing Safetensors file"""
        dataset = NSSDataset(self.params, DataLoaderMode.TRAIN)
        # Check if two Safetensors files have been found
        self.assertEqual(len(dataset.captures), 1)
        x, y = dataset[0]

        # Check outputs exist
        self.assertIsInstance(x, dict)
        self.assertEqual(len(x), 13)
        self.assertIsInstance(y, torch.Tensor)

    def test_len_matches_frame_indexes(self):
        """Test dataset length matches total windows"""
        dataset = NSSDataset(self.params, DataLoaderMode.TRAIN)
        # Calculate number of sliding windows
        total_windows = 0
        for safetensor_file in dataset.captures:
            with safe_open(safetensor_file, framework="pt") as f:
                length = int(f.metadata()["Length"])
            total_windows += length - (self.params.dataset.recurrent_samples - 1)
        self.assertEqual(len(dataset), total_windows)

    def test_dataloader_data_transformation(self):
        """Check data from raw to training transformation is correct"""

        params = create_simple_params(dataset="./tests/usecases/nss/datasets/train")
        params.dataset.recurrent_samples = 2
        params.train.batch_size = 2
        params.dataset.gt_augmentation = False

        dataset = NSSDataset(
            params, loader_mode=DataLoaderMode.TRAIN, extension=DatasetType.SAFETENSOR
        )

        def seed_worker(_):
            torch.manual_seed(params.train.seed)
            worker_seed = torch.initial_seed() % 2**32
            random.seed(worker_seed)

        expected_tensor = torch.Generator()
        expected_tensor.manual_seed(params.train.seed)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=params.train.batch_size,
            num_workers=0,
            pin_memory=True,
            prefetch_factor=None,
            drop_last=True,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=expected_tensor,
        )

        data = next(iter(dataloader))[0]

        golden_data = torch.load(
            "tests/usecases/nss/unit/data/nss_v1_golden_values/dataloader_output_fp32.pt",
            map_location=torch.device("cpu"),
            weights_only=True,
        )

        data.pop("seq", None)
        data.pop("colour", None)
        golden_data.pop("seq", None)

        for key in data:
            tensor = data[key]
            expected_tensor = golden_data[key]

            torch.testing.assert_close(
                tensor,
                expected_tensor,
                atol=1e-8,
                rtol=1e-5,
                msg=f"Mismatch in tensor '{key}'",
            )

            self.assertTrue(
                torch.equal(tensor, expected_tensor), f"Tensors {key} not equal"
            )

    def test_missing_exposure_field(self):
        """Test handling of missing exposure field in Safetensors file"""
        # Create a new Safetensors file without exposure field for testing
        original_safetensor_path = (
            "./tests/usecases/nss/datasets/train/train_cropped_sample.safetensors"
        )
        original_metadata = {}
        tensors = {}

        with safe_open(original_safetensor_path, framework="pt") as f:
            # Save original metadata so that it can be used in the new safetensors file
            original_metadata = f.metadata()
            # Copy all tensors except exposure
            for key in f.keys():
                if key != "exposure":
                    tensors[key] = f.get_tensor(key)

        # Create new safetensors file
        new_safetensor_path = "./tests/usecases/nss/datasets/missing_exposure_field/missing_exposure.safetensors"  # pylint: disable=line-too-long
        create_directory("./tests/usecases/nss/datasets/missing_exposure_field")
        save_file(tensors, new_safetensor_path, metadata=original_metadata)

        # Create params pointing to SafeTensors dataset with missing exposure field
        params = create_simple_params(
            dataset="tests/usecases/nss/datasets/missing_exposure_field"
        )

        # Set exposure to None to match missing dataset field
        params.dataset.exposure = None
        params.dataset.recurrent_samples = 4

        dataset = NSSDataset(params, DataLoaderMode.TRAIN)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            prefetch_factor=None,
            drop_last=True,
            persistent_workers=False,
        )
        data = next(iter(dataloader))[0]

        # Check that exposure field exists in returned `data` and is correctly set
        self.assertIsInstance(data, dict)
        self.assertIn("exposure", data)
        # Check exposure tensor is all ones (exp(0) = 1.0)
        expected_exposure = torch.ones_like(data["exposure"])
        self.assertTrue(
            torch.equal(data["exposure"], expected_exposure),
            "Exposure tensor should be filled with ones.",
        )

    def test_windows_skip_mid_sequence_cuts(self):
        """Sliding windows stop before mid-span cuts but still start exactly on the cut."""
        params = create_simple_params()
        params.dataset.recurrent_samples = 4

        flags = [False, False, False, False, True, False, False, False, False, False]
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = write_camera_cut_fixture(Path(tmp_dir) / "train", flags)
            params.dataset.path.train = Path(dataset_dir)
            dataset = NSSDataset(params, DataLoaderMode.TRAIN)  # NB: TRAIN mode

            windows = dataset.capture_windows[dataset.captures[0]]
            # All windows must end before the cut index (4) unless they start exactly at it.
            for start_idx, stop_idx in windows:
                if start_idx < 4:
                    self.assertLessEqual(stop_idx, 4)
            # Ensure we still generate windows that start on the cut frame itself.
            self.assertTrue(any(start == 4 for start, _ in windows))

    def test_seq_id_changes_per_segment_in_test_mode(self):
        """Sequence hashes change at each cut when iterating in TEST mode."""
        params = create_simple_params()
        params.dataset.recurrent_samples = 4

        flags = [False, False, False, False, True, False, False, False]
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = write_camera_cut_fixture(Path(tmp_dir) / "test", flags)
            params.dataset.path.test = Path(dataset_dir)
            dataset = NSSDataset(params, DataLoaderMode.TEST)  # NB: TEST mode

        capture = dataset.captures[0]
        seen_hashes = []
        for start_idx, _ in dataset.capture_windows[capture]:
            seq_id = dataset._get_seq_id(capture, start_idx)[0][0].item()
            seen_hashes.append((start_idx, seq_id))

        self.assertNotEqual(seen_hashes[0][1], seen_hashes[-1][1])
        for idx in range(1, len(seen_hashes)):
            prev_seq = dataset.window_sequence_map[capture][seen_hashes[idx - 1][0]]
            curr_seq = dataset.window_sequence_map[capture][seen_hashes[idx][0]]
            if prev_seq == curr_seq:
                self.assertEqual(seen_hashes[idx - 1][1], seen_hashes[idx][1])
            else:
                self.assertNotEqual(seen_hashes[idx - 1][1], seen_hashes[idx][1])

    def test_seq_tensor_resets_when_camera_cut_true(self):
        """`seq` tensor visible to the model flips only when the camera_cut flag is set."""
        params = create_simple_params()
        params.dataset.recurrent_samples = 4

        flags = [False, False, True, False, False]
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = write_camera_cut_fixture(Path(tmp_dir) / "test", flags)
            params.dataset.path.test = Path(dataset_dir)
            dataset = NSSDataset(params, DataLoaderMode.TEST)  # NB: TEST mode

            seq_values = []
            for idx in range(len(flags)):
                sample, _ = dataset[idx]
                seq_values.append(int(sample["seq"].view(-1)[0].item()))

        self.assertGreaterEqual(len(seq_values), 4)
        self.assertEqual(seq_values[0], seq_values[1])
        self.assertNotEqual(seq_values[1], seq_values[2])
        self.assertEqual(seq_values[2], seq_values[3])

    def test_short_camera_cut_segments_are_dropped(self):
        """Segments shorter than recurrent_samples should not emit any sliding windows."""
        params = create_simple_params()
        params.dataset.recurrent_samples = 4

        # 2 valid sequences: frames 0-3 and 7-10 (inclusive)
        flags = [
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            False,
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = write_camera_cut_fixture(Path(tmp_dir) / "train", flags)
            params.dataset.path.train = Path(dataset_dir)
            dataset = NSSDataset(params, DataLoaderMode.TRAIN)  # NB: TRAIN mode

        capture = dataset.captures[0]
        windows = dataset.capture_windows[capture]

        self.assertEqual(
            len(windows),
            2,
            msg="Expected only two valid sliding windows from the two valid segments.",
        )
        self.assertEqual(
            windows, [(0, 4), (7, 11)], "Unexpected sliding window ranges."
        )

    def test_legacy_file_without_camera_cut(self):
        """Test handling of legacy files without camera cut flags."""
        params = create_simple_params()
        params.dataset.recurrent_samples = 4

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = write_camera_cut_fixture(
                Path(tmp_dir) / "train",
                camera_cut_flags=[False] * 10,
                include_camera_cut=False,
            )
            params.dataset.path.train = Path(dataset_dir)
            dataset = NSSDataset(params, DataLoaderMode.TRAIN)

            capture = dataset.captures[0]
            self.assertEqual(dataset.capture_sequences[capture], [(0, 10)])
