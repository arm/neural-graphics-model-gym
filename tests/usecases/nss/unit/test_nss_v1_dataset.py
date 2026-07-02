# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from ng_model_gym.core.data import DataLoaderMode, DatasetType
from ng_model_gym.core.utils.io.file_utils import create_directory
from ng_model_gym.usecases.nss.data.dataset import NSSDataset
from tests.testing_utils import create_simple_params, validate_params
from tests.usecases.nss.unit.data.camera_cut_builders import write_camera_cut_fixture


class TestNSSV1Dataset(unittest.TestCase):
    """Tests for the NSS v1 dataset."""

    def setUp(self):
        """Set up a default NSS v1 training config."""
        self.params = create_simple_params(
            usecase="nss_v1", dataset_path=Path("tests/usecases/nss/datasets/train")
        )
        self.params.model.recurrent_samples = 4
        self.params.dataset.gt_augmentation = False

    def test_existing_safetensors_file(self):
        """Test loading an existing Safetensors file."""
        dataset = NSSDataset(self.params, DataLoaderMode.TRAIN)
        self.assertEqual(len(dataset.captures), 1)

        x, y = dataset[0]

        self.assertIsInstance(x, dict)
        self.assertEqual(len(x), 13)
        self.assertIsInstance(y, torch.Tensor)

    def test_len_matches_frame_indexes(self):
        """Test dataset length matches the total number of sliding windows."""
        dataset = NSSDataset(self.params, DataLoaderMode.TRAIN)

        total_windows = 0
        for safetensor_file in dataset.captures:
            with safe_open(safetensor_file, framework="pt") as f:
                length = int(f.metadata()["Length"])
            total_windows += length - (self.params.model.recurrent_samples - 1)

        self.assertEqual(len(dataset), total_windows)

    def test_missing_exposure_field(self):
        """Missing exposure fields should default to all ones."""
        original_safetensor_path = Path(
            "tests/usecases/nss/datasets/train/train_cropped_sample.safetensors"
        )
        tensors = {}

        with safe_open(original_safetensor_path, framework="pt") as f:
            original_metadata = f.metadata()
            for key in f.keys():
                if key != "exposure":
                    tensors[key] = f.get_tensor(key)

        new_safetensor_dir = Path("tests/usecases/nss/datasets/missing_exposure_field")
        new_safetensor_path = new_safetensor_dir / "missing_exposure.safetensors"
        create_directory(str(new_safetensor_dir))
        save_file(tensors, new_safetensor_path, metadata=original_metadata)

        params = create_simple_params(
            usecase="nss_v1",
            dataset_path=new_safetensor_dir,
        )
        params.dataset.exposure = None
        params.model.recurrent_samples = 4

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

        self.assertIsInstance(data, dict)
        self.assertIn("exposure", data)
        expected_exposure = torch.ones_like(data["exposure"])
        self.assertTrue(torch.equal(data["exposure"], expected_exposure))

    def test_leaves_low_res_motion_unnormalized_by_default(self):
        """NSS v1 keeps raw low-res motion by default."""
        dataset = NSSDataset(self.params, DataLoaderMode.TRAIN)
        x, _ = dataset[0]

        with safe_open(dataset.captures[0], framework="pt", device="cpu") as f:
            expected_motion_lr = f.get_slice("motion_lr")[
                0 : self.params.model.recurrent_samples
            ].to(torch.float32)

        torch.testing.assert_close(x["motion_lr"], expected_motion_lr)

    def test_omitted_normalize_lr_motion_defaults_to_raw_motion(self):
        """Omitting normalize_lr_motion should still preserve raw low-res motion."""
        params_json = create_simple_params(
            usecase="nss_v1",
            dataset_path=Path("tests/usecases/nss/datasets/train"),
        ).model_dump(mode="json")
        params_json["model"].pop("normalize_lr_motion", None)
        params = validate_params(params_json)
        params.model.recurrent_samples = 4
        params.dataset.gt_augmentation = False

        dataset = NSSDataset(params, DataLoaderMode.TRAIN)
        x, _ = dataset[0]

        self.assertFalse(dataset.normalize_lr_motion)
        with safe_open(dataset.captures[0], framework="pt", device="cpu") as f:
            expected_motion_lr = f.get_slice("motion_lr")[
                0 : params.model.recurrent_samples
            ].to(torch.float32)

        torch.testing.assert_close(x["motion_lr"], expected_motion_lr)

    def test_derives_motion_lr_without_legacy_normalization(self):
        """NSS v1 derives raw low-res motion from motion when motion_lr is absent."""
        original_safetensor_path = Path(
            "tests/usecases/nss/datasets/train/train_cropped_sample.safetensors"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "train"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            new_safetensor_path = dataset_dir / "missing_motion_lr.safetensors"

            tensors = {}
            with safe_open(original_safetensor_path, framework="pt") as f:
                original_metadata = f.metadata()
                for key in f.keys():
                    if key != "motion_lr":
                        tensors[key] = f.get_tensor(key)

            save_file(tensors, new_safetensor_path, metadata=original_metadata)

            params_json = create_simple_params(
                usecase="nss_v1", dataset_path=dataset_dir
            ).model_dump(mode="json")
            params_json["model"].pop("normalize_lr_motion", None)
            params = validate_params(params_json)
            params.model.recurrent_samples = 4

            dataset = NSSDataset(params, DataLoaderMode.TRAIN)
            data_frame = dataset._load_data(
                dataset.captures[0],
                start=0,
                stop=params.model.recurrent_samples,
            )

            expected_motion_lr = (
                F.interpolate(
                    tensors["motion"][: params.model.recurrent_samples],
                    scale_factor=0.5,
                    mode="nearest",
                )
                * 0.5
            )

            self.assertFalse(dataset.normalize_lr_motion)
            torch.testing.assert_close(data_frame["motion_lr"], expected_motion_lr)

    def test_windows_skip_mid_sequence_cuts(self):
        """Sliding windows stop before mid-span cuts but still start on the cut."""
        params = create_simple_params(usecase="nss_v1")
        params.model.recurrent_samples = 4

        flags = [False, False, False, False, True, False, False, False, False, False]
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = write_camera_cut_fixture(Path(tmp_dir) / "train", flags)
            params.dataset.path.train = Path(dataset_dir)
            dataset = NSSDataset(params, DataLoaderMode.TRAIN)

            windows = dataset.capture_windows[dataset.captures[0]]
            for start_idx, stop_idx in windows:
                if start_idx < 4:
                    self.assertLessEqual(stop_idx, 4)

            self.assertTrue(any(start == 4 for start, _ in windows))

    def test_seq_id_changes_per_segment_in_test_mode(self):
        """Sequence hashes change at each cut when iterating in TEST mode."""
        params = create_simple_params(usecase="nss_v1")
        params.model.recurrent_samples = 4

        flags = [False, False, False, False, True, False, False, False]
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = write_camera_cut_fixture(Path(tmp_dir) / "test", flags)
            params.dataset.path.test = Path(dataset_dir)
            dataset = NSSDataset(params, DataLoaderMode.TEST)

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
        """`seq` should flip only when the camera_cut flag is set."""
        params = create_simple_params(usecase="nss_v1")
        params.model.recurrent_samples = 4

        flags = [False, False, True, False, False]
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = write_camera_cut_fixture(Path(tmp_dir) / "test", flags)
            params.dataset.path.test = Path(dataset_dir)
            dataset = NSSDataset(params, DataLoaderMode.TEST)

            seq_values = []
            for idx in range(len(flags)):
                sample, _ = dataset[idx]
                seq_values.append(int(sample["seq"].view(-1)[0].item()))

        self.assertGreaterEqual(len(seq_values), 4)
        self.assertEqual(seq_values[0], seq_values[1])
        self.assertNotEqual(seq_values[1], seq_values[2])
        self.assertEqual(seq_values[2], seq_values[3])

    def test_short_camera_cut_segments_are_dropped(self):
        """Segments shorter than recurrent_samples should not emit windows."""
        params = create_simple_params(usecase="nss_v1")
        params.model.recurrent_samples = 4

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
            dataset = NSSDataset(params, DataLoaderMode.TRAIN)

        capture = dataset.captures[0]
        windows = dataset.capture_windows[capture]

        self.assertEqual(len(windows), 2)
        self.assertEqual(windows, [(0, 4), (7, 11)])

    def test_legacy_file_without_camera_cut(self):
        """Legacy files without camera_cut should be treated as a single sequence."""
        params = create_simple_params(usecase="nss_v1")
        params.model.recurrent_samples = 4

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

    def test_short_capture_logs_warning_and_is_skipped(self):
        """Captures shorter than recurrent_samples emit a warning and are ignored."""
        params = create_simple_params(usecase="nss_v1")
        params.model.recurrent_samples = 4

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "train"
            write_camera_cut_fixture(
                dataset_dir,
                camera_cut_flags=[False, False, False],
                file_name="short_capture",
            )
            write_camera_cut_fixture(
                dataset_dir,
                camera_cut_flags=[False] * 6,
                file_name="valid_capture",
            )
            params.dataset.path.train = dataset_dir

            with self.assertLogs(
                "ng_model_gym.usecases.nss.data.dataset", level="WARNING"
            ) as logs:
                dataset = NSSDataset(params, DataLoaderMode.TRAIN)

            joined_logs = "\n".join(logs.output)
            self.assertIn("Skipping capture", joined_logs)
            self.assertIn(
                "Capture length (3) must be >= recurrent_samples (4)", joined_logs
            )
            self.assertEqual(len(dataset.captures), 2)
            self.assertEqual(len(dataset.capture_windows), 1)

    def test_recurrent_samples(self):
        """Test recurrent_samples validation and mode-specific behavior."""
        with self.subTest("train_mode_two_recurrent_samples"):
            params = create_simple_params(
                usecase="nss_v1",
                dataset_path=Path("tests/usecases/nss/datasets/train"),
            )
            params.model.recurrent_samples = 2
            dataset = NSSDataset(
                params,
                loader_mode=DataLoaderMode.TRAIN,
                extension=DatasetType.SAFETENSOR,
            )
            self.assertEqual(dataset.recurrent_samples, 2)

        with self.subTest("train_mode_missing_recurrent_samples"):
            config_json = create_simple_params(
                usecase="nss_v1",
                dataset_path=Path("tests/usecases/nss/datasets/train"),
            ).model_dump(mode="json")
            config_json["model"] = {
                "name": "my_custom_model",
                "model_source": "custom",
                "version": "1",
                "quality": "high",
            }
            params = validate_params(config_json)

            with self.assertRaises(ValueError) as exc:
                NSSDataset(
                    params,
                    loader_mode=DataLoaderMode.TRAIN,
                    extension=DatasetType.SAFETENSOR,
                )
            self.assertIn("model.recurrent_samples", str(exc.exception))
            self.assertIn("train/validation", str(exc.exception))

        with self.subTest("train_mode_raise_recurrent_samples_less_than_two"):
            params = create_simple_params(
                usecase="nss_v1",
                dataset_path=Path("tests/usecases/nss/datasets/train"),
            )
            params.model.recurrent_samples = 1
            with self.assertRaises(ValueError) as exc:
                NSSDataset(
                    params,
                    loader_mode=DataLoaderMode.TRAIN,
                    extension=DatasetType.SAFETENSOR,
                )
            self.assertIn("model.recurrent_samples >= 2", str(exc.exception))

        with self.subTest("test_mode_sets_single_sample"):
            params = create_simple_params(
                usecase="nss_v1",
                dataset_path=Path("tests/usecases/nss/datasets/train"),
            )
            params.model.recurrent_samples = 10
            dataset = NSSDataset(
                params,
                loader_mode=DataLoaderMode.TEST,
                extension=DatasetType.SAFETENSOR,
            )
            self.assertEqual(dataset.recurrent_samples, 1)


class TestNSSV1DatasetGolden(unittest.TestCase):
    """Test NSS v1 dataset output against known golden values."""

    def test_dataset_first_batch_golden_values(self):
        """Test first deterministic NSS v1 dataset batch."""
        expected_data = torch.load(
            "tests/usecases/nss/unit/data/nss_v1_golden_values/"
            + "nss_v1_high_dataset_output_golden.pt",
            map_location=torch.device("cpu"),
            weights_only=True,
        )

        params = create_simple_params(
            usecase="nss_v1", dataset_path=Path("tests/usecases/nss/datasets/train")
        )
        params.model.quality = "high"
        params.model.recurrent_samples = 2
        params.train.batch_size = 2
        params.dataset.gt_augmentation = False
        params.dataset.exposure = 2.0
        params.dataset.tonemapper = "reinhard"
        params.dataset.num_workers = 0

        dataset = NSSDataset(
            params,
            loader_mode=DataLoaderMode.TRAIN,
            extension=DatasetType.SAFETENSOR,
        )
        dataloader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=params.train.batch_size,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        x, y = next(iter(dataloader))
        actual_data = {**x, "ground_truth": y}

        expected_seq = expected_data.pop("seq")
        actual_seq = actual_data.pop("seq")
        self.assertEqual(actual_seq.shape, expected_seq.shape)
        self.assertEqual(actual_seq.dtype, expected_seq.dtype)

        for metadata_key in ("batch_size", "recurrent_samples", "quality"):
            expected_data.pop(metadata_key, None)

        for key, expected_tensor in expected_data.items():
            with self.subTest(key=key):
                torch.testing.assert_close(
                    actual_data[key],
                    expected_tensor,
                    rtol=1e-5,
                    atol=1e-8,
                    msg=f"Mismatch in dataset tensor '{key}'",
                )
