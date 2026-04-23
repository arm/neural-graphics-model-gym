# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from shutil import copy2

import torch
from pydantic import ValidationError
from safetensors import safe_open
from safetensors.torch import save_file

from ng_model_gym.core.config.config_model import ConfigModel
from ng_model_gym.core.data import DataLoaderMode
from ng_model_gym.usecases.nfru.data.dataset import (
    _round_up_to_odd_int,
    NFRU_MAX_OFFSET,
    NFRU_MIN_OFFSET,
    NFRU_OPTIONAL_FLOW_KEY,
    NFRUDataset,
    NFRUDatasetWrapper,
)
from ng_model_gym.usecases.nfru.data.processing import (
    NFRU_DEFAULT_BRIGHTNESS_SHAPE_AUG_PROBABILITY,
    NFRU_DEFAULT_SHAPE_AUG_MAX_DISPLACEMENT,
    NFRU_DEFAULT_SHAPE_AUG_MAX_SIZE,
    NFRU_DEFAULT_SHAPE_AUG_NUM_SHAPES,
    NFRU_DEFAULT_SHAPE_AUG_PROBABILITY,
)
from tests.testing_utils import create_simple_params

NFRU_TEMPLATE_PATH = Path("src/ng_model_gym/usecases/nfru/configs/nfru_template.json")
NFRU_SAMPLE_DIR = Path("tests/usecases/nfru/data/nfru_sample")
NFRU_SAMPLE_FILE = NFRU_SAMPLE_DIR / "0000.safetensors"
NFRU_GOLDEN_OUTPUT_PATH = Path(
    "tests/usecases/nfru/unit/data/nfru_v1_golden_values/dataloader_output_fp32.pt"
)
_LEGACY_NFRU_OPTIONAL_FLOW_KEY = "flow_m1_f30_p1@blockmatch_v3"


def _build_nfru_config(
    dataset_dir: Path,
    dataset_overrides: dict | None = None,
    model_overrides: dict | None = None,
) -> ConfigModel:
    """Build a validated NFRU config pointed at a test dataset directory."""
    raw_config = json.loads(NFRU_TEMPLATE_PATH.read_text(encoding="utf-8"))
    dataset_path = str(dataset_dir)
    raw_config["dataset"]["path"] = {
        "train": dataset_path,
        "validation": dataset_path,
        "test": dataset_path,
    }
    raw_config["dataset"]["num_workers"] = 0
    raw_config["dataset"]["prefetch_factor"] = 1
    if dataset_overrides:
        raw_config["dataset"].update(dataset_overrides)
    if model_overrides:
        raw_config["model"].update(model_overrides)
    return ConfigModel.model_validate(raw_config)


def _clone_safetensor_without_keys(
    source_path: Path,
    target_dir: Path,
    drop_keys: set[str],
) -> Path:
    """Clone a safetensors file into target_dir while dropping selected keys."""
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / source_path.name
    tensors = {}
    metadata = {}
    with safe_open(source_path, framework="pt", device="cpu") as source_file:
        metadata = source_file.metadata()
        for key in source_file.keys():
            if key not in drop_keys:
                tensors[key] = source_file.get_tensor(key)
    save_file(tensors, str(destination), metadata=metadata)
    return destination


@unittest.skip("NFRU CI/assets disabled for now")
class TestNFRUDataset(unittest.TestCase):  # pylint: disable=too-many-public-methods
    """Unit tests for the NFRU dataset."""

    def _dataset(
        self,
        loader_mode: DataLoaderMode | str = DataLoaderMode.TRAIN,
        dataset_dir: Path | None = None,
        dataset_overrides: dict | None = None,
        model_overrides: dict | None = None,
    ) -> NFRUDataset:
        """Construct an NFRUDataset instance for the requested mode and overrides."""
        config = _build_nfru_config(
            dataset_dir or NFRU_SAMPLE_DIR, dataset_overrides, model_overrides
        )
        return NFRUDataset(config, loader_mode=loader_mode)

    def test_existing_safetensors_file(self):
        """Loads sample safetensors data and returns the expected tuple contract."""
        dataset = self._dataset(DataLoaderMode.TRAIN)
        self.assertEqual(len(dataset.sequences), 1)
        self.assertGreater(len(dataset), 0)

        x, y = dataset[0]
        self.assertIsInstance(x, dict)
        self.assertIsInstance(y, torch.Tensor)
        self.assertIn("seq", x)
        self.assertIn("MotionMat", x)
        self.assertIn("y_true", x)

    def test_nfru_config_accepts_template_without_offset_fields(self):
        """NFRU template should validate without dataset offset config fields."""
        raw_config = json.loads(NFRU_TEMPLATE_PATH.read_text(encoding="utf-8"))
        raw_config["dataset"]["path"] = {
            "train": str(NFRU_SAMPLE_DIR),
            "validation": str(NFRU_SAMPLE_DIR),
            "test": str(NFRU_SAMPLE_DIR),
        }

        config = ConfigModel.model_validate(raw_config)
        self.assertFalse(hasattr(config.model, "new_dynamic_mask"))

    def test_nfru_config_rejects_removed_offset_fields(self):
        """Dataset offset fields were removed from config and must be rejected."""
        raw_config = json.loads(NFRU_TEMPLATE_PATH.read_text(encoding="utf-8"))
        raw_config["dataset"]["path"] = {
            "train": str(NFRU_SAMPLE_DIR),
            "validation": str(NFRU_SAMPLE_DIR),
            "test": str(NFRU_SAMPLE_DIR),
        }
        raw_config["dataset"]["nfru_min_offset"] = 3
        with self.assertRaises(ValidationError) as exc:
            ConfigModel.model_validate(raw_config)
        self.assertIn("nfru_min_offset", str(exc.exception))

    def test_nfru_config_rejects_model_scale_field(self):
        """NFRU should reject model.scale because the field is no longer supported."""
        raw_config = json.loads(NFRU_TEMPLATE_PATH.read_text(encoding="utf-8"))
        raw_config["dataset"]["path"] = {
            "train": str(NFRU_SAMPLE_DIR),
            "validation": str(NFRU_SAMPLE_DIR),
            "test": str(NFRU_SAMPLE_DIR),
        }
        raw_config["model"]["scale"] = 1.0
        with self.assertRaises(ValidationError) as exc:
            ConfigModel.model_validate(raw_config)
        self.assertIn("scale", str(exc.exception))
        self.assertIn("Extra inputs are not permitted", str(exc.exception))

    def test_nfru_config_rejects_unsupported_model_scale_factor(self):
        """model.scale_factor must stay pinned to 2.0 for the current NFRU model."""
        raw_config = json.loads(NFRU_TEMPLATE_PATH.read_text(encoding="utf-8"))
        raw_config["dataset"]["path"] = {
            "train": str(NFRU_SAMPLE_DIR),
            "validation": str(NFRU_SAMPLE_DIR),
            "test": str(NFRU_SAMPLE_DIR),
        }
        raw_config["model"]["scale_factor"] = 1.0

        with self.assertRaises(ValidationError) as exc:
            ConfigModel.model_validate(raw_config)

        self.assertIn("scale_factor", str(exc.exception))
        self.assertIn("must be 2 or 2.0", str(exc.exception))

    def test_nfru_config_accepts_integer_model_scale_factor(self):
        """model.scale_factor may be provided as 2 or 2.0 and is normalized to 2.0."""
        raw_config = json.loads(NFRU_TEMPLATE_PATH.read_text(encoding="utf-8"))
        raw_config["dataset"]["path"] = {
            "train": str(NFRU_SAMPLE_DIR),
            "validation": str(NFRU_SAMPLE_DIR),
            "test": str(NFRU_SAMPLE_DIR),
        }
        raw_config["model"]["scale_factor"] = 2

        config = ConfigModel.model_validate(raw_config)
        self.assertEqual(config.model.scale_factor, 2.0)

    def test_nfru_config_requires_colour_preprocessing(self):
        """NFRU v1 configs must provide dataset.colour_preprocessing."""
        raw_config = json.loads(NFRU_TEMPLATE_PATH.read_text(encoding="utf-8"))
        raw_config["dataset"]["path"] = {
            "train": str(NFRU_SAMPLE_DIR),
            "validation": str(NFRU_SAMPLE_DIR),
            "test": str(NFRU_SAMPLE_DIR),
        }
        raw_config["dataset"].pop("colour_preprocessing")
        raw_config["dataset"]["tonemapper"] = "reinhard"
        raw_config["dataset"]["exposure"] = 2.0

        with self.assertRaises(ValidationError) as exc:
            ConfigModel.model_validate(raw_config)

        self.assertIn("colour_preprocessing", str(exc.exception))
        self.assertIn("dataset.colour_preprocessing.train", str(exc.exception))

    def test_nfru_config_requires_all_colour_preprocessing_splits(self):
        """NFRU v1 configs must define train/validation/test preprocessing."""
        raw_config = json.loads(NFRU_TEMPLATE_PATH.read_text(encoding="utf-8"))
        raw_config["dataset"]["path"] = {
            "train": str(NFRU_SAMPLE_DIR),
            "validation": str(NFRU_SAMPLE_DIR),
            "test": str(NFRU_SAMPLE_DIR),
        }
        raw_config["dataset"]["colour_preprocessing"].pop("validation")

        with self.assertRaises(ValidationError) as exc:
            ConfigModel.model_validate(raw_config)

        self.assertIn("colour_preprocessing", str(exc.exception))
        self.assertIn("Missing or invalid splits: validation", str(exc.exception))

    def test_nfru_config_defaults_missing_colour_preprocessing_exposure(self):
        """NFRU colour-preprocessing splits may omit exposure and use the schema default."""
        raw_config = json.loads(NFRU_TEMPLATE_PATH.read_text(encoding="utf-8"))
        raw_config["dataset"]["path"] = {
            "train": str(NFRU_SAMPLE_DIR),
            "validation": str(NFRU_SAMPLE_DIR),
            "test": str(NFRU_SAMPLE_DIR),
        }
        raw_config["dataset"]["colour_preprocessing"]["validation"].pop("exposure")

        config = ConfigModel.model_validate(raw_config)
        if config.dataset.colour_preprocessing is None:
            self.fail("Expected colour_preprocessing config for NFRU")
        validation_split = config.dataset.colour_preprocessing.validation
        if validation_split is None:
            self.fail("Expected validation colour-preprocessing split for NFRU")
        self.assertEqual(validation_split.exposure, 2.0)

    def test_nfru_config_defaults_align_data_when_missing(self):
        """NFRU configs should default dataset.align_data to true when omitted."""
        raw_config = json.loads(NFRU_TEMPLATE_PATH.read_text(encoding="utf-8"))
        raw_config["dataset"]["path"] = {
            "train": str(NFRU_SAMPLE_DIR),
            "validation": str(NFRU_SAMPLE_DIR),
            "test": str(NFRU_SAMPLE_DIR),
        }
        raw_config["dataset"].pop("align_data")

        config = ConfigModel.model_validate(raw_config)
        self.assertTrue(config.dataset.align_data)

    def test_nfru_config_rejects_processing_scale_factor(self):
        """processing.scale_factor should be rejected as an unknown field."""
        raw_config = json.loads(NFRU_TEMPLATE_PATH.read_text(encoding="utf-8"))
        raw_config["dataset"]["path"] = {
            "train": str(NFRU_SAMPLE_DIR),
            "validation": str(NFRU_SAMPLE_DIR),
            "test": str(NFRU_SAMPLE_DIR),
        }
        raw_config["processing"]["scale_factor"] = 3

        with self.assertRaises(ValidationError) as exc:
            ConfigModel.model_validate(raw_config)

        self.assertIn("scale_factor", str(exc.exception))
        self.assertIn("Extra inputs are not permitted", str(exc.exception))

    def test_len_matches_expected_window_count_train(self):
        """Train mode should produce 3 windows for Length=10 with default offsets/stride."""
        dataset = self._dataset(DataLoaderMode.TRAIN)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.min_offset, NFRU_MIN_OFFSET)
        self.assertEqual(dataset.max_offset, NFRU_MAX_OFFSET)

    def test_len_matches_expected_window_count_test(self):
        """Test mode should produce 3 windows for Length=10 with fixed stride=2."""
        dataset = self._dataset(DataLoaderMode.TEST)
        self.assertEqual(len(dataset), 3)

    def test_getitem_out_of_bounds_raises_stop_iteration(self):
        """Accessing one-past-the-end index raises StopIteration."""
        dataset = self._dataset(DataLoaderMode.TRAIN)
        with self.assertRaises(StopIteration):
            _ = dataset[len(dataset)]

    def test_loader_mode_accepts_enum_and_string_equivalently(self):
        """Enum and string loader modes normalize to equivalent dataset behavior."""
        enum_dataset = self._dataset(DataLoaderMode.TRAIN)
        string_dataset = self._dataset("train")
        self.assertEqual(enum_dataset.frame_indexes, string_dataset.frame_indexes)
        self.assertEqual(enum_dataset.loader_mode, "train")
        self.assertEqual(string_dataset.loader_mode, "train")
        self.assertEqual(enum_dataset.loader_mode_enum, DataLoaderMode.TRAIN)
        self.assertEqual(string_dataset.loader_mode_enum, DataLoaderMode.TRAIN)

    def test_augmentation_disabled_in_validation_and_test(self):
        """Validation and test modes force augmentation off."""
        val_dataset = self._dataset(
            DataLoaderMode.VAL, dataset_overrides={"gt_augmentation": True}
        )
        test_dataset = self._dataset(
            DataLoaderMode.TEST, dataset_overrides={"gt_augmentation": True}
        )
        self.assertFalse(val_dataset.processor_params["augment"])
        self.assertFalse(test_dataset.processor_params["augment"])
        self.assertFalse(val_dataset.processor_params["shape_aug"])
        self.assertFalse(test_dataset.processor_params["shape_aug"])

    def test_augmentation_enabled_in_train_when_config_true(self):
        """Train mode keeps augmentation enabled when configured."""
        train_dataset = self._dataset(
            DataLoaderMode.TRAIN, dataset_overrides={"gt_augmentation": True}
        )
        self.assertTrue(train_dataset.processor_params["augment"])
        self.assertTrue(train_dataset.processor_params["shape_aug"])
        self.assertEqual(
            train_dataset.processor_params["shape_aug_num_shapes"],
            list(NFRU_DEFAULT_SHAPE_AUG_NUM_SHAPES),
        )
        self.assertEqual(
            train_dataset.processor_params["shape_aug_max_size"],
            list(NFRU_DEFAULT_SHAPE_AUG_MAX_SIZE),
        )
        self.assertEqual(
            train_dataset.processor_params["shape_aug_max_displacement"],
            list(NFRU_DEFAULT_SHAPE_AUG_MAX_DISPLACEMENT),
        )
        self.assertEqual(
            train_dataset.processor_params["shape_aug_probability"],
            NFRU_DEFAULT_SHAPE_AUG_PROBABILITY,
        )
        self.assertEqual(
            train_dataset.processor_params["brightness_shape_aug_probability"],
            NFRU_DEFAULT_BRIGHTNESS_SHAPE_AUG_PROBABILITY,
        )

    def test_missing_loader_mode_path_raises_value_error(self):
        """Missing mode-specific path in config should fail with ValueError."""
        config = _build_nfru_config(
            NFRU_SAMPLE_DIR,
            dataset_overrides={
                "path": {
                    "train": str(NFRU_SAMPLE_DIR),
                    "validation": str(NFRU_SAMPLE_DIR),
                    "test": None,
                }
            },
        )
        with self.assertRaises(ValueError) as exc:
            NFRUDataset(config, loader_mode=DataLoaderMode.TEST)
        self.assertIn("loader mode 'test'", str(exc.exception))

    def test_non_config_model_raises_type_error(self):
        """Passing a non-ConfigModel object should raise TypeError."""
        with self.assertRaises(TypeError):
            NFRUDataset({"dataset": {"path": {"train": "unused"}}}, "train")

    def test_missing_dataset_directory_raises_file_not_found(self):
        """A missing dataset directory should raise FileNotFoundError."""
        missing_path = NFRU_SAMPLE_DIR / "does_not_exist"
        config = _build_nfru_config(missing_path)
        with self.assertRaises(FileNotFoundError):
            NFRUDataset(config, loader_mode=DataLoaderMode.TRAIN)

    def test_invalid_extension_value_raises_value_error(self):
        """Invalid extension values should fail DatasetType conversion."""
        config = _build_nfru_config(NFRU_SAMPLE_DIR)
        with self.assertRaises(ValueError):
            NFRUDataset(config, loader_mode=DataLoaderMode.TRAIN, extension=".xyz")

    def test_empty_dataset_directory_current_behavior(self):
        """Current empty-dir behavior raises IndexError during first sequence read."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _build_nfru_config(Path(tmp_dir))
            with self.assertRaises(IndexError):
                NFRUDataset(config, loader_mode=DataLoaderMode.TRAIN)

    def test_generate_frame_indexes_train_default_step_is_two_for_non_legacy(self):
        """Non-legacy train mode should step by 2."""
        dataset = self._dataset(DataLoaderMode.TRAIN)
        starts = [start for _, start, _ in dataset.frame_indexes]
        self.assertEqual(starts, [0, 2, 4])

    def test_generate_frame_indexes_test_step_is_two(self):
        """Test mode should step by 2."""
        dataset = self._dataset(DataLoaderMode.TEST)
        starts = [start for _, start, _ in dataset.frame_indexes]
        self.assertEqual(starts, [0, 2, 4])

    def test_generate_frame_indexes_legacy_capture_paths_restore_train_step_one_legacy(
        self,
    ):
        """Legacy capture path matching should restore train step=1."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            legacy_dir = Path(tmp_dir) / "legacy_nfru_sample"
            legacy_dir.mkdir(parents=True, exist_ok=True)
            copy2(NFRU_SAMPLE_FILE, legacy_dir / NFRU_SAMPLE_FILE.name)
            dataset = self._dataset(
                DataLoaderMode.TRAIN,
                dataset_dir=legacy_dir,
                model_overrides={"legacy_nfru_capture_paths": ["nfru_sample"]},
            )
            starts = [start for _, start, _ in dataset.frame_indexes]
            self.assertEqual(starts, [0, 1, 2, 3, 4, 5])

    def test_generate_frame_indexes_legacy_tokens_case_insensitive_matches_legacy(self):
        """Legacy token matching is case-insensitive substring matching."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            legacy_dir = Path(tmp_dir) / "nfru_sample_capture"
            legacy_dir.mkdir(parents=True, exist_ok=True)
            copy2(NFRU_SAMPLE_FILE, legacy_dir / NFRU_SAMPLE_FILE.name)
            dataset = self._dataset(
                DataLoaderMode.TRAIN,
                dataset_dir=legacy_dir,
                model_overrides={"legacy_nfru_capture_paths": ["NFRU_SAMPLE"]},
            )
            starts = [start for _, start, _ in dataset.frame_indexes]
            self.assertEqual(starts, [0, 1, 2, 3, 4, 5])

    def test_round_up_to_odd_int_helper(self):
        """Odd inputs are unchanged and even inputs are incremented by one."""
        self.assertEqual(_round_up_to_odd_int(3), 3)
        self.assertEqual(_round_up_to_odd_int(4), 5)

    def test_dataset_uses_fixed_temporal_offsets(self):
        """Dataset should always use fixed NFRU temporal offsets."""
        dataset = self._dataset(DataLoaderMode.TRAIN)
        self.assertEqual(dataset.min_offset, NFRU_MIN_OFFSET)
        self.assertEqual(dataset.max_offset, NFRU_MAX_OFFSET)

        starts = [start for _, start, _ in dataset.frame_indexes]
        self.assertEqual(starts, [0, 2, 4])
        x, _ = dataset[0]
        self.assertIn("ViewProj_m3", x)

    def test_train_mode_seq_hash_changes_per_window(self):
        """Train mode should use a distinct sequence hash per window."""
        dataset = self._dataset(DataLoaderMode.TRAIN)
        seq_ids = []
        for idx, _ in enumerate(dataset.frame_indexes):
            x, _ = dataset[idx]
            seq_tensor = x["seq"]
            self.assertIsInstance(seq_tensor, torch.Tensor)
            self.assertEqual(seq_tensor.shape, torch.Size([1, 1]))
            seq_ids.append(int(seq_tensor.view(-1)[0].item()))
        self.assertEqual(len(seq_ids), len(set(seq_ids)))

    def test_test_mode_seq_hash_constant_per_sequence(self):
        """Test mode should keep the same sequence hash for all windows of one sequence."""
        dataset = self._dataset(DataLoaderMode.TEST)
        seq_ids = []
        for idx, _ in enumerate(dataset.frame_indexes):
            x, _ = dataset[idx]
            seq_tensor = x["seq"]
            self.assertIsInstance(seq_tensor, torch.Tensor)
            self.assertEqual(seq_tensor.shape, torch.Size([1, 1]))
            seq_ids.append(int(seq_tensor.view(-1)[0].item()))
        self.assertEqual(len(set(seq_ids)), 1)

    def test_seq_id_non_legacy_train_uses_halved_start_index_mapping(self):
        """Non-legacy train path maps seq id by start_idx // 2."""
        dataset = self._dataset(DataLoaderMode.TRAIN)
        seq_path = dataset.sequences[0]
        self.assertTrue(
            torch.equal(dataset._get_seq_id(seq_path, 0), dataset.hashes[seq_path][0])
        )
        self.assertTrue(
            torch.equal(dataset._get_seq_id(seq_path, 2), dataset.hashes[seq_path][1])
        )
        self.assertTrue(
            torch.equal(dataset._get_seq_id(seq_path, 4), dataset.hashes[seq_path][2])
        )

    def test_seq_id_legacy_train_uses_direct_start_index_mapping_legacy(self):
        """Legacy train path maps seq id by direct start index."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            legacy_dir = Path(tmp_dir) / "legacy_nfru_sample"
            legacy_dir.mkdir(parents=True, exist_ok=True)
            copied_path = legacy_dir / NFRU_SAMPLE_FILE.name
            copy2(NFRU_SAMPLE_FILE, copied_path)

            dataset = self._dataset(
                DataLoaderMode.TRAIN,
                dataset_dir=legacy_dir,
                model_overrides={"legacy_nfru_capture_paths": ["legacy_nfru_sample"]},
            )

            seq_path = dataset.sequences[0]
            for start_idx in range(6):
                seq_id = dataset._get_seq_id(seq_path, start_idx)
                self.assertTrue(
                    torch.equal(seq_id, dataset.hashes[seq_path][start_idx])
                )

    def test_is_legacy_capture_path_behavior_legacy(self):
        """Legacy path detection should match configured tokens and reject non-matches."""
        plain_dataset = self._dataset(DataLoaderMode.TRAIN)
        seq_path = plain_dataset.sequences[0]
        self.assertFalse(plain_dataset._is_legacy_capture_path(seq_path))

        with tempfile.TemporaryDirectory() as tmp_dir:
            legacy_dir = Path(tmp_dir) / "legacy_nfru_sample"
            legacy_dir.mkdir(parents=True, exist_ok=True)
            copy2(NFRU_SAMPLE_FILE, legacy_dir / NFRU_SAMPLE_FILE.name)

            matching = self._dataset(
                DataLoaderMode.TRAIN,
                dataset_dir=legacy_dir,
                model_overrides={"legacy_nfru_capture_paths": ["nfru_sample"]},
            )
            self.assertTrue(matching._is_legacy_capture_path(matching.sequences[0]))

            non_matching = self._dataset(
                DataLoaderMode.TRAIN,
                dataset_dir=legacy_dir,
                model_overrides={"legacy_nfru_capture_paths": ["no_match_token"]},
            )
            self.assertFalse(
                non_matching._is_legacy_capture_path(non_matching.sequences[0])
            )

    def test_to_read_contains_required_model_inputs_and_keep_keys(self):
        """`to_read` should include key model inputs and explicit keep keys."""
        dataset = self._dataset(DataLoaderMode.TRAIN)
        self.assertIn("ViewProj_m1", dataset.to_read)
        self.assertIn("ViewProj_p1", dataset.to_read)
        self.assertIn("rgb_linear_t", dataset.to_read)

    def test_align_data_to_model_logs_missing_features_without_crash(self):
        """Aligning with missing features should log errors and still return present keys."""
        dataset = self._dataset(DataLoaderMode.TRAIN)
        reduced = {"rgb_linear": torch.float32}
        with self.assertLogs(
            "ng_model_gym.usecases.nfru.data.dataset", level="ERROR"
        ) as captured:
            aligned = dataset.align_data_to_model(reduced, load_all=False)
        self.assertIn("rgb_linear_m1", aligned)
        self.assertIn("rgb_linear_p1", aligned)
        self.assertTrue(
            any("not in safetensors file" in msg for msg in captured.output)
        )

    def test_align_data_to_model_load_all_true_returns_concrete_keys(self):
        """`load_all=True` should return concrete offset keys without raising."""
        dataset = self._dataset(DataLoaderMode.TRAIN)
        feature_dict = {"rgb_linear": torch.float32, "ViewProj": torch.float32}
        aligned = dataset.align_data_to_model(feature_dict, load_all=True, keep_keys=[])
        expected = {
            "rgb_linear_m3",
            "rgb_linear_m2",
            "rgb_linear_m1",
            "rgb_linear_t",
            "rgb_linear_p1",
            "ViewProj_m3",
            "ViewProj_m2",
            "ViewProj_m1",
            "ViewProj_t",
            "ViewProj_p1",
        }
        self.assertTrue(expected.issubset(set(aligned.keys())))

    def test_align_data_to_model_load_all_true_includes_keep_keys(self):
        """Keep keys should be present in `load_all=True` output when source features exist."""
        dataset = self._dataset(DataLoaderMode.TRAIN)
        feature_dict = {"rgb_linear": torch.float32, "ViewProj": torch.float32}
        aligned = dataset.align_data_to_model(
            feature_dict,
            load_all=True,
            keep_keys=["ViewProj_m1", "rgb_linear_t"],
        )
        self.assertIn("ViewProj_m1", aligned)
        self.assertIn("rgb_linear_t", aligned)

    def test_dataloader_data_transformation_golden_values(self):
        """Dataset output should remain numerically stable against committed golden tensors."""
        dataset = self._dataset(
            DataLoaderMode.TRAIN, dataset_overrides={"gt_augmentation": False}
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            prefetch_factor=None,
            drop_last=False,
            persistent_workers=False,
            generator=torch.Generator().manual_seed(123456),
        )

        batch = next(iter(dataloader))
        data = batch[0]
        target = batch[1]
        data.pop("seq", None)

        golden_data = torch.load(
            NFRU_GOLDEN_OUTPUT_PATH,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        golden_x = golden_data.get("inputs")
        golden_y = golden_data.get("targets")
        self.assertIsNotNone(golden_x)
        self.assertIsNotNone(golden_y)
        golden_x = dict(golden_x)
        golden_x.pop("seq", None)
        self.assertNotIn(NFRU_OPTIONAL_FLOW_KEY, data)
        # We compute flow at runtime so remove from goldens for comparison
        golden_x.pop(NFRU_OPTIONAL_FLOW_KEY, None)
        golden_x.pop(_LEGACY_NFRU_OPTIONAL_FLOW_KEY, None)

        self.assertEqual(set(data.keys()), set(golden_x.keys()))
        for key in data:
            golden_tensor = golden_x[key].to(data[key].dtype)
            if key == "MotionMat":
                # MotionMat is derived from matrix inverse/matmul and can vary across
                # CPU/BLAS backends, so assert invariants instead of fixed goldens.
                self.assertEqual(data[key].shape, golden_tensor.shape)
                self.assertTrue(
                    torch.isfinite(data[key]).all(),
                    "MotionMat contains non-finite values",
                )
                motion = data[key]
                if motion.ndim == 4:
                    motion_forward = motion[0, 0]
                    motion_backward = motion[0, 1]
                else:
                    motion_forward = motion[0]
                    motion_backward = motion[1]
                identity = torch.eye(
                    motion_forward.shape[-1],
                    dtype=motion_forward.dtype,
                    device=motion_forward.device,
                )
                torch.testing.assert_close(
                    motion_forward @ motion_backward,
                    identity,
                    atol=1e-3,
                    rtol=1e-3,
                    msg="MotionMat forward/backward transforms are inconsistent",
                )
                continue
            torch.testing.assert_close(
                data[key],
                golden_tensor,
                atol=1e-8,
                rtol=1e-5,
                msg=f"Mismatch in tensor '{key}'",
            )
            self.assertTrue(
                torch.equal(data[key], golden_tensor),
                f"Tensors {key} are not exactly equal",
            )
        torch.testing.assert_close(target, golden_y, atol=1e-8, rtol=1e-5)
        self.assertTrue(torch.equal(target, golden_y), "Target tensor is not equal")

    def test_missing_rgb_linear_target_fails_with_clear_error(self):
        """Missing `rgb_linear` should fail with a clear target-frame error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mutated_dir = Path(tmp_dir) / "missing_rgb_linear"
            _clone_safetensor_without_keys(
                NFRU_SAMPLE_FILE, mutated_dir, drop_keys={"rgb_linear"}
            )
            dataset = self._dataset(DataLoaderMode.TRAIN, dataset_dir=mutated_dir)
            with self.assertRaises(ValueError) as exc:
                _ = dataset[0]
            self.assertIn("Missing rgb target frame", str(exc.exception))

    def test_missing_required_motion_feature_current_behavior(self):
        """Missing motion tensors are logged and omitted from output in current behavior."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mutated_dir = Path(tmp_dir) / "missing_mv"
            _clone_safetensor_without_keys(
                NFRU_SAMPLE_FILE, mutated_dir, drop_keys={"mv_{}_f30_m1"}
            )
            with self.assertLogs(
                "ng_model_gym.usecases.nfru.data.dataset", level="ERROR"
            ) as captured:
                dataset = self._dataset(DataLoaderMode.TRAIN, dataset_dir=mutated_dir)
            x, _ = dataset[0]
            self.assertNotIn("mv_p1_f30_m1", x)
            self.assertNotIn("mv_m1_f30_m3", x)
            self.assertTrue(any("mv_{}_f30_m1" in msg for msg in captured.output))

    def test_missing_exposure_current_behavior(self):
        """Missing exposure is logged and omitted from output in current behavior."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mutated_dir = Path(tmp_dir) / "missing_exposure"
            _clone_safetensor_without_keys(
                NFRU_SAMPLE_FILE, mutated_dir, drop_keys={"exposure"}
            )
            with self.assertLogs(
                "ng_model_gym.usecases.nfru.data.dataset", level="ERROR"
            ) as captured:
                dataset = self._dataset(DataLoaderMode.TRAIN, dataset_dir=mutated_dir)
            x, _ = dataset[0]
            self.assertNotIn("exposure_p1", x)
            self.assertTrue(any("key exposure" in msg for msg in captured.output))

    def test_test_mode_seq_hash_constant_within_sequence_and_distinct_across_sequences(
        self,
    ):
        """In test mode, hash stays constant per sequence and differs across sequences."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "two_sequences"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            copy2(NFRU_SAMPLE_FILE, dataset_dir / "0000.safetensors")
            copy2(NFRU_SAMPLE_FILE, dataset_dir / "0001.safetensors")

            dataset = self._dataset(DataLoaderMode.TEST, dataset_dir=dataset_dir)
            seq_ids_by_sequence = defaultdict(set)
            for idx, (seq_path, _, _) in enumerate(dataset.frame_indexes):
                x, _ = dataset[idx]
                seq_ids_by_sequence[seq_path].add(int(x["seq"].view(-1)[0].item()))

            self.assertEqual(len(seq_ids_by_sequence), 2)
            for seq_ids in seq_ids_by_sequence.values():
                self.assertEqual(len(seq_ids), 1)
            distinct_values = {
                next(iter(seq_ids)) for seq_ids in seq_ids_by_sequence.values()
            }
            self.assertEqual(len(distinct_values), 2)

    def test_load_data_applies_offsets_and_shapes(self):
        """`_load_data` should apply offsets and emit single-frame unsqueezed tensors."""
        dataset = self._dataset(DataLoaderMode.TRAIN)
        seq_path = dataset.sequences[0]
        loaded = dataset._load_data(seq_path, start=0)

        self.assertEqual(set(loaded.keys()), set(dataset.to_read.keys()))
        self.assertEqual(loaded["exposure_p1"].shape, torch.Size([1]))
        self.assertEqual(loaded["mv_p1_f30_m1"].shape, torch.Size([1, 2, 540, 960]))

        with safe_open(seq_path, framework="pt", device="cpu") as f:
            expected_exposure = f.get_slice("exposure")[4].unsqueeze(0)
            expected_mv = f.get_slice("mv_{}_f30_m1")[4].unsqueeze(0)

        torch.testing.assert_close(loaded["exposure_p1"], expected_exposure)
        torch.testing.assert_close(loaded["mv_p1_f30_m1"], expected_mv)

    def test_getitem_output_schema(self):
        """`__getitem__` should return expected keys, float32 tensors, and y==y_true."""
        dataset = self._dataset(DataLoaderMode.TEST)
        x, y = dataset[0]

        required_keys = {
            "MotionMat",
            "seq",
            "y_true",
            "ViewProj_m1",
            "ViewProj_p1",
            "rgb_linear_m1",
            "rgb_linear_p1",
        }
        self.assertTrue(required_keys.issubset(set(x.keys())))

        for tensor in x.values():
            self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.float32)
        torch.testing.assert_close(y, x["y_true"])

    def test_wrapper_behaves_like_base_dataset(self):
        """Wrapper dataset should be behaviorally equivalent to base dataset."""
        config = _build_nfru_config(
            NFRU_SAMPLE_DIR, dataset_overrides={"gt_augmentation": True}
        )
        base_dataset = NFRUDataset(config, loader_mode=DataLoaderMode.TEST)
        wrapper_dataset = NFRUDatasetWrapper(config, loader_mode=DataLoaderMode.TEST)

        self.assertEqual(len(base_dataset), len(wrapper_dataset))
        base_x, base_y = base_dataset[0]
        wrapper_x, wrapper_y = wrapper_dataset[0]
        self.assertEqual(set(base_x.keys()), set(wrapper_x.keys()))
        for key in base_x:
            torch.testing.assert_close(base_x[key], wrapper_x[key])
        torch.testing.assert_close(base_y, wrapper_y)

    @unittest.skip("Short NFRU safetensors fixture removed pending replacement")
    def test_raises_when_sequence_has_fewer_than_minimum_frames(self):
        """A 4-frame sequence cannot satisfy the 5-frame sliding-window minimum."""
        dataset_path = Path("tests/datasets/test_nfru_4f_short")
        params = create_simple_params(usecase="nfru", dataset_path=str(dataset_path))
        params.dataset.path.test = dataset_path
        params.dataset.health_check = False

        with self.assertRaises(RuntimeError) as exc_info:
            NFRUDataset(params, loader_mode=DataLoaderMode.TEST)

        message = str(exc_info.exception)
        self.assertIn("Couldn't find sufficient frame data in dataset", message)
        self.assertIn("NFRU requires at least 5 frame(s) per sequence.", message)
        self.assertIn("0000.safetensors: 4 frame(s)", message)
