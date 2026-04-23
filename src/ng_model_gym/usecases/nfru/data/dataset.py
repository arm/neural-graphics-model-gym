# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import safetensors
import torch

from ng_model_gym.core.config.config_model import ConfigModel
from ng_model_gym.core.data import (
    DataLoaderMode,
    DatasetType,
    generic_safetensors_reader,
)
from ng_model_gym.core.data.dataset_registry import register_dataset
from ng_model_gym.usecases.nfru.data.naming import (
    convert_str_offset_to_int,
    DataVariable,
)
from ng_model_gym.usecases.nfru.data.processing import (
    NFRU_DEFAULT_BRIGHTNESS_SHAPE_AUG_PROBABILITY,
    NFRU_DEFAULT_SHAPE_AUG_MAX_DISPLACEMENT,
    NFRU_DEFAULT_SHAPE_AUG_MAX_SIZE,
    NFRU_DEFAULT_SHAPE_AUG_NUM_SHAPES,
    NFRU_DEFAULT_SHAPE_AUG_PROBABILITY,
    process_data,
)


def _round_up_to_odd_int(n):
    """Return `n` if odd, otherwise the next odd integer."""
    return n if n % 2 == 1 else n + 1


logger = logging.getLogger(__name__)
# Non-legacy captures store 30 fps inputs on a 60 fps timeline, so sliding windows
# advance by two timeline frames unless a path is configured to use legacy behavior.
NFRU_REFERENCE_FPS = 60
NFRU_NON_LEGACY_CAPTURE_FPS = 30
NFRU_NON_LEGACY_FRAME_STEP = NFRU_REFERENCE_FPS // NFRU_NON_LEGACY_CAPTURE_FPS
NFRU_MIN_OFFSET = 3
NFRU_MAX_OFFSET = 1
NFRU_OPTIONAL_FLOW_KEY = "flow_m1_f30_p1@blockmatch_v311"


class NFRUDataset(torch.utils.data.Dataset):
    """Dataset loader for NFRU safetensors."""

    def __init__(
        self,
        config_params: ConfigModel,
        loader_mode: Union[DataLoaderMode, str] = DataLoaderMode.TEST,
        extension: DatasetType = DatasetType.SAFETENSOR,
    ) -> None:
        if not isinstance(config_params, ConfigModel):
            raise TypeError("NFRUDataset expects a ConfigModel configuration.")

        self.config_params = config_params
        self.extension = (
            extension if isinstance(extension, DatasetType) else DatasetType(extension)
        )

        if isinstance(loader_mode, DataLoaderMode):
            self.loader_mode_enum = loader_mode
            self.loader_mode = loader_mode.value
        else:
            self.loader_mode = loader_mode
            try:
                self.loader_mode_enum = DataLoaderMode(loader_mode)
            except ValueError:
                self.loader_mode_enum = None

        dataset_cfg = self.config_params.dataset

        dataset_path = getattr(dataset_cfg.path, self.loader_mode, None)
        if dataset_path is None:
            raise ValueError(
                f"No dataset path configured for loader mode '{self.loader_mode}'."
            )

        self.dset_path = Path(dataset_path)
        if not self.dset_path.exists():
            raise FileNotFoundError(f"Dataset is missing: {self.dset_path}")

        model_cfg = self.config_params.model
        legacy_capture_paths = getattr(model_cfg, "legacy_nfru_capture_paths", []) or []
        self.legacy_nfru_capture_paths = tuple(
            str(path_part).lower() for path_part in legacy_capture_paths if path_part
        )

        # Temporal context window for NFRU model inputs.
        self.min_offset = NFRU_MIN_OFFSET
        self.max_offset = NFRU_MAX_OFFSET

        augment_enabled = bool(dataset_cfg.gt_augmentation)
        if self.loader_mode_enum in (DataLoaderMode.VAL, DataLoaderMode.TEST):
            augment_enabled = False

        self.processor_params = {
            "augment": augment_enabled,
            "shape_aug": augment_enabled,
            "shape_aug_num_shapes": list(NFRU_DEFAULT_SHAPE_AUG_NUM_SHAPES),
            "shape_aug_max_size": list(NFRU_DEFAULT_SHAPE_AUG_MAX_SIZE),
            "shape_aug_max_displacement": list(NFRU_DEFAULT_SHAPE_AUG_MAX_DISPLACEMENT),
            "shape_aug_probability": NFRU_DEFAULT_SHAPE_AUG_PROBABILITY,
            "brightness_shape_aug_probability": NFRU_DEFAULT_BRIGHTNESS_SHAPE_AUG_PROBABILITY,
        }

        # Find all the files inside the dataset
        sequence_paths = set(self.dset_path.rglob(f"*{self.extension.value}"))
        self.sequences = self.all_sequences = sorted(sequence_paths)

        # Filter features to read to only those specified by the model inputs
        data_frame = generic_safetensors_reader(self.sequences[0], 0)
        self.to_read = {k: data_frame[k].dtype for k in list(data_frame.keys())}
        logger.info(f"NFRU dataloader available features: {self.to_read}")
        should_align_data_to_model = dataset_cfg.align_data
        keep_keys = [
            "ViewProj_m1",
            "ViewProj_p1",
            "rgb_linear_t",
        ]

        self.to_read = self.align_data_to_model(
            self.to_read, load_all=not should_align_data_to_model, keep_keys=keep_keys
        )
        logger.info(f"NFRU dataloader model-aligned features: {self.to_read}")

        # Configure self.sequence_data and self.frame_indexes
        self._configure_sliding_window_data_structures()

        # Processor configuration
        logger.info(f"NFRU dataloader processor params: {self.processor_params}")

        # Generate a hash map for sequences, providing a unique integer for each
        # temporal window to inform the model when to flush and reset history buffers.
        # pylint: disable=duplicate-code
        to_hash = lambda x: torch.tensor(hash(x)).view(1, 1).to("cpu")
        if self.loader_mode != "test":
            # Hash per sequence and starting frame to guarantee a reset after each window
            self.hashes = {
                k: [to_hash(str(k) + str(b)) for (b, _) in v]
                for k, v in self.sequence_data.items()
            }
            all_hashes = [h for hash_list in self.hashes.values() for h in hash_list]
        else:
            # Hash only needed per sequence
            self.hashes = {k: to_hash(k) for k in self.sequence_data.keys()}
            all_hashes = list(self.hashes.values())
        assert len(all_hashes) == len(set(all_hashes)), "Duplicate hashes found!"

        # Only used for test, store data
        self.existing_seq_path = None
        self.data_frame = None

        # Validate data
        if self.loader_mode_enum == DataLoaderMode.TRAIN and dataset_cfg.health_check:
            logger.info(
                "NFRU dataset health check requested but not yet implemented."
            )
        # pylint: enable=duplicate-code

    def _configure_sliding_window_data_structures(self):
        """
        Configures the sliding window index data structures for the dataset:
        - self.sequence_data: dict with one entry per sequence; this maps to a
          list of frame index ranges for the sequence in question
        - self.frame_indexes: list of tuples with one entry per sequence /
          frame index combination.
        """
        self.sequence_data = self._generate_frame_indexes(
            self.sequences,
            self.loader_mode,
            min_offset=self.min_offset,
            max_offset=self.max_offset,
        )
        self.frame_indexes = [
            (k, *indices) for k, v in self.sequence_data.items() for indices in v
        ]
        if len(self.frame_indexes) == 0:
            first_valid_center = _round_up_to_odd_int(self.min_offset)
            min_required_length = first_valid_center + self.max_offset + 1
            sequence_lengths = {}
            for seq in self.sequences:
                with safetensors.safe_open(seq, framework="pt") as f:
                    sequence_lengths[str(seq)] = int(f.metadata()["Length"])

            length_summary = ", ".join(
                f"{Path(k).name}: {v} frame(s)" for k, v in sequence_lengths.items()
            )
            raise RuntimeError(
                "Couldn't find sufficient frame data in dataset "
                f"{self.dset_path}. NFRU requires at least {min_required_length} "
                f"frame(s) per sequence. File contents: {length_summary}"
            )

    def _is_legacy_capture_path(self, seq_path: Path) -> bool:
        """Return True when sequence path matches configured legacy capture path tokens."""
        if not self.legacy_nfru_capture_paths:
            return False
        seq_path_str = str(seq_path).lower()
        return any(token in seq_path_str for token in self.legacy_nfru_capture_paths)

    def _load_data(self, seq_path: Path, start: int) -> Dict[str, torch.Tensor]:
        """Lazily open a safetensors file, only load data we need based on `self.to_read`
        and extract a temporal window between start:stop indexes.
        """
        data_frame = {}

        # Load data for every new sequence
        with safetensors.safe_open(seq_path, framework="pt", device="cpu") as f:
            for key in self.to_read:
                key_data_variable = DataVariable(key)
                key_for_safetensor = key_data_variable.generate_non_concrete_variable(
                    timeline_fps=NFRU_REFERENCE_FPS
                )

                if key_data_variable.is_mv:
                    offset = key_data_variable.ivec_from
                else:
                    key_split = key.split("_")
                    offset = convert_str_offset_to_int(key_split[-1])
                    key_for_safetensor = "_".join(key_split[:-1])

                data_frame[key] = f.get_slice(key_for_safetensor)[
                    start + self.min_offset + offset
                ].unsqueeze(dim=0)

        return data_frame

    def _get_seq_id(self, seq_path: Path, start_idx: int) -> torch.Tensor:
        """Return a deterministic sequence identifier for the requested window."""
        if self.loader_mode != "test":
            if not self._is_legacy_capture_path(seq_path):
                start_idx //= 2
            return self.hashes[seq_path][start_idx]
        return self.hashes[seq_path]

    def get_uncropped_model_input_names(self) -> List[str]:
        """Input tensor names expected by the model prior to cropping."""
        flow = [NFRU_OPTIONAL_FLOW_KEY]

        # pylint: disable=duplicate-code
        uncropped_inputs = [
            "rgb_linear_m1",
            "rgb_linear_p1",
            "depth_m1",
            "depth_p1",
            "mv_p1_f30_m1",
            "sy_m1_f30_p1",
            "mv_m1_f30_m3",
            "sy_m1_f30_m3",
            "exposure_p1",
            "DepthParams_p1",
            # Scenario Evaluator inputs
            "NearPlane_p1",
            "FarPlane_p1",
            "FovY_p1",
            "infinite_zFar_p1",
            "ViewProj_m3",
        ] + flow
        # pylint: enable=duplicate-code
        return uncropped_inputs

    def get_additional_crop_model_input_names(self) -> List[str]:
        """Additional tensor names emitted when crops are requested."""
        return [
            "crop_id_x",
            "crop_id_y",
            "crop_sz",
        ]

    def get_model_input_names(self) -> List[str]:
        """Return the ordered list of model input tensors."""
        input_names = self.get_uncropped_model_input_names()
        return input_names

    def get_model_output_names(self) -> List[str]:
        """Return the tensor names exposed by the dataset."""
        return ["output"]

    def align_data_to_model(
        self,
        feature_dict: Dict[str, torch.dtype],
        load_all: bool,
        keep_keys: List[str] | None = None,
    ) -> Dict[str, torch.dtype]:
        """Filter features from safetensors to the subset required by the model."""
        if keep_keys is None:
            keep_keys = []
        if not load_all:
            model_inputs = self.get_model_input_names()
        else:
            model_inputs = []
            for i in range(-self.min_offset, self.max_offset + 1):
                for k in feature_dict:
                    model_inputs.append(
                        DataVariable(k).generate_concrete_variable(
                            i, timeline_fps=NFRU_REFERENCE_FPS
                        )
                    )

        model_inputs += keep_keys

        output_feature_dict = {}
        for k in model_inputs:
            k_short = DataVariable(k).generate_non_concrete_variable(
                timeline_fps=NFRU_REFERENCE_FPS
            )
            if k_short in feature_dict.keys():
                output_feature_dict[k] = feature_dict[k_short]
            else:
                if k == NFRU_OPTIONAL_FLOW_KEY:
                    logger.info(
                        f"Optional precomputed flow '{k}' not present in safetensors "
                        "file. NFRU will recompute flow internally."
                    )
                else:
                    logger.error(
                        f"key {k_short} (from: {k}) not in safetensors file, but "
                        "requested by the model"
                    )

        return output_feature_dict

    def _generate_frame_indexes(
        self,
        sequences: List[Path],
        loader_mode: str = "train",
        min_offset: int = 1,
        max_offset: int = 1,
    ) -> Dict[Path, List[Tuple[int, int]]]:
        """Generate sliding-window frame index ranges for each sequence."""
        frame_indexes: Dict[Path, List[Tuple[int, int]]] = {}
        for seq in sequences:
            with safetensors.safe_open(seq, framework="pt") as f:
                metadata = f.metadata()
                seq_length = int(metadata["Length"])
                step = NFRU_NON_LEGACY_FRAME_STEP if ("test" in loader_mode) else 1
                if "test" not in loader_mode and not self._is_legacy_capture_path(seq):
                    step = NFRU_NON_LEGACY_FRAME_STEP
                idx_range = range(
                    _round_up_to_odd_int(min_offset), seq_length - max_offset, step
                )

                frame_indexes[seq] = [
                    (n - min_offset, n + max_offset + 1) for n in idx_range
                ]
        return frame_indexes

    def __len__(self) -> int:
        """Return the number of sliding windows available."""
        return len(self.frame_indexes)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Load and process a temporal window, returning inputs and targets."""
        # pylint: disable=duplicate-code
        if idx >= len(self):
            raise StopIteration

        # Extract dataset and sliding window indices
        seq_path, start_idx, _ = self.frame_indexes[idx]

        # Load data from safetensors file
        data_frame = self._load_data(seq_path, start=start_idx)

        # Append the unique sequence identifier
        data_frame["seq"] = self._get_seq_id(seq_path, start_idx)

        # Process data
        data_frame = process_data(
            data_frame,
            **self.processor_params,
        )
        # pylint: enable=duplicate-code

        return data_frame


@register_dataset(name="NFRU", version="1")
class NFRUDatasetWrapper(NFRUDataset):
    """
    Thin wrapper that registers the NFRU dataset with the ng-model-gym
    dataset registry while preserving legacy loader behaviour.
    """

    def __init__(
        self,
        config_params: ConfigModel,
        loader_mode: Union[DataLoaderMode, str] = "test",
        extension: DatasetType = DatasetType.SAFETENSOR,
    ):
        super().__init__(config_params, loader_mode=loader_mode, extension=extension)

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, idx: int):
        return super().__getitem__(idx)
