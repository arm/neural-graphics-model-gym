# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import safetensors
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ng_model_gym.core.data.dataset_registry import register_dataset
from ng_model_gym.core.data.utils import DataLoaderMode, DatasetType
from ng_model_gym.core.model.graphics_utils import fixed_normalize_mvs
from ng_model_gym.core.utils.config_model import ConfigModel
from ng_model_gym.usecases.nss.dataloader.health_check import health_check_dataset
from ng_model_gym.usecases.nss.dataloader.process_functions import process_nss_data

logger = logging.getLogger(__name__)


@register_dataset(name="NSS", version="1")
class NSSDataset(Dataset):
    """
    Dataset object providing training data for the Neural Super Sampling use-case.
    """

    def __init__(
        self,
        config_params: ConfigModel,
        loader_mode: DataLoaderMode,
        extension=DatasetType.SAFETENSOR,
    ):
        self.extension = extension
        self.config_params = config_params
        self.loader_mode = loader_mode
        self.recurrent_samples = (
            int(self.config_params.dataset.recurrent_samples)
            if loader_mode != DataLoaderMode.TEST
            else 1
        )
        self.data_path = getattr(self.config_params.dataset.path, self.loader_mode)

        if self.recurrent_samples == 1 and loader_mode != DataLoaderMode.TEST:
            self.features_to_read = [
                "colour_linear",
                "ground_truth_linear",
                "exposure",
                "jitter",
            ]
        else:
            self.features_to_read = [
                "colour_linear",
                "ground_truth_linear",
                "motion",
                "depth",
                "depth_params",
                "exposure",
                "jitter",
                "render_size",
                "zNear",
                "zFar",
                "motion_lr",
            ]

        # Safetensor loading
        self.sequences = sorted(set(self.data_path.rglob("*.safetensors")))
        self.sequence_data = self.generate_frame_indexes(
            self.sequences, self.recurrent_samples
        )
        self.frame_indexes = [
            (k, *indices) for k, v in self.sequence_data.items() for indices in v
        ]

        # Augmentations disabled for test and validation, otherwise use config
        self.augment = (
            False
            if loader_mode != DataLoaderMode.TRAIN
            else self.config_params.dataset.gt_augmentation
        )

        if not self.frame_indexes:
            logger.error(f"Empty list of file paths returned from {self.data_path}")
            raise ValueError(
                f"No {extension} files found. "
                f"Potential causes: Empty dataset directory or overly-strict exclude parameter."
            )

        # Generate a Hash-map for sequences, unique integer for each sequence / recurrent
        # window, to inform the model on when to flush and reset history buffers
        to_hash = lambda x: torch.tensor(hash(x)).view(1, 1)
        if self.loader_mode != DataLoaderMode.TEST:
            # Hash per sequence and per starting frame of recurrent window
            # Guarantees a reset after each window
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

    def generate_frame_indexes(
        self, sequences: List[Path], n_frames: int
    ) -> Dict[str, Tuple[int, int]]:
        """Generate sliding window frame index ranges for each sequence."""
        frame_indexes = {}
        for seq in sequences:
            with safetensors.safe_open(seq, framework="pt") as f:
                metadata = f.metadata()
                seq_length = int(metadata["Length"])
                frame_indexes[seq] = [
                    (n, n + n_frames) for n in range(seq_length - (n_frames - 1))
                ]
        return frame_indexes

    def _get_seq_id(self, seq_path: Path, start_idx: int) -> torch.Tensor:
        if self.loader_mode != DataLoaderMode.TEST:
            return self.hashes[seq_path][start_idx].expand(self.recurrent_samples, -1)
        return self.hashes[seq_path].expand(self.recurrent_samples, -1)

    def __len__(self):
        return len(self.frame_indexes)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # Extract dataset and sliding window indices
        seq_path, start_idx, stop_idx = self.frame_indexes[idx]

        # Load data from Safetensors file
        data_frame = self._load_data(seq_path, start=start_idx, stop=stop_idx)

        # Append the unique sequence identifier
        data_frame["seq"] = self._get_seq_id(seq_path, start_idx)

        # Process the data to apply tonemapping, augmentations, etc.
        x, y = process_nss_data(
            data_frame,
            augment=self.augment,
            exposure=self.config_params.dataset.exposure,
            tonemapper=self.config_params.dataset.tonemapper,
        )

        return x, y

    def _load_data(
        self, seq_path: Path, start: int, stop: int
    ) -> Dict[str, torch.Tensor]:
        """Lazily open Safetensors file, only load data we need based on `self.features_to_read`
        and extract recurrent window between start:stop indexes.
        """
        data_frame = {}
        with safetensors.safe_open(seq_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in self.features_to_read:
                    data_frame[k] = f.get_slice(k)[start:stop]
            if "motion_lr" not in data_frame:
                data_frame["motion_lr"] = (
                    F.interpolate(
                        data_frame["motion"], scale_factor=0.5, mode="nearest"
                    )
                    * 0.5
                )

        # Height and weight are hardcoded to match the slang shaders
        data_frame["motion_lr"] = fixed_normalize_mvs(
            data_frame["motion_lr"], height=544.0, width=960.0
        )
        return data_frame


def seed_worker(worker_id):  # pylint: disable=unused-argument
    """
    Make random functions that are run from workers be deterministic -
    see https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    # Torch will automatically set a seed for this worker based on a random number generated
    # from the torch instance running in the main process, but it doesn't handle seeds for other
    # libraries that we use such as numpy.
    # So we copy the auto-generated torch seed to these other libraries.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(
    config_params: ConfigModel,
    num_workers=1,
    prefetch_factor=1,
    loader_mode: DataLoaderMode = DataLoaderMode.TRAIN,
    trace_mode: bool = False,  # Note: should be False for QAT
):
    """Return the desired DataLoader.

    Args:
        config_params: Configuration parameters.
        num_workers: Number of workers for the DataLoader to use.
        prefetch_factor: How many batches each worker should try to preload.
        loader_mode: What mode the dataloader should be set to.
        trace_mode: Whether the dataloader returned is for tracing at export.
            Should be False for QAT.

    Returns:
        PyTorch DataLoader object ready to produce data as configured by passed in parameters.
    """

    dataset = NSSDataset(
        config_params, loader_mode=loader_mode, extension=(DatasetType.SAFETENSOR)
    )
    if trace_mode:
        logger.debug(
            "Loading data in trace mode with batch_size of 2 (for FP32/PTQ export)"
        )
        batch_size = 2  # Avoid 0/1 Specialization Problem in PT2 Export
    else:
        batch_size = (
            1 if loader_mode == DataLoaderMode.TEST else config_params.train.batch_size
        )

    # Shuffle only when training.
    shuffle = loader_mode == DataLoaderMode.TRAIN

    if (
        config_params.dataset.recurrent_samples < 2
        and loader_mode == DataLoaderMode.TRAIN
    ):
        raise ValueError(
            "Number of recurrent samples must be greater than 1 for training."
        )

    g = torch.Generator()
    g.manual_seed(config_params.train.seed)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=g,
    )

    if config_params.dataset.health_check:
        health_check_dataset(dataloader, loader_mode)

    return dataloader
