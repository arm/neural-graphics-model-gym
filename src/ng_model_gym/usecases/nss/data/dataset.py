# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import safetensors
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torcheval.metrics.functional import peak_signal_noise_ratio as psnr
from tqdm.auto import tqdm

from ng_model_gym.core.data.dataset_registry import register_dataset
from ng_model_gym.core.data.utils import DataLoaderMode, DatasetType
from ng_model_gym.core.model.graphics_utils import fixed_normalize_mvs
from ng_model_gym.core.model.layers.dense_warp import DenseWarp
from ng_model_gym.core.model.layers.resampling import DownSampling2D
from ng_model_gym.core.utils.config_model import ConfigModel
from ng_model_gym.usecases.nss.data.process_functions import process_nss_data

logger = logging.getLogger(__name__)

HIGH_PSNR_FOR_STATIC_WARP = 70.0


@register_dataset(name="NSS", version="1")
class NSSDataset(Dataset):
    """
    Dataset object providing training data for the Neural Super Sampling use-case.
    """

    def __init__(
        self,
        config_params: ConfigModel,
        loader_mode: DataLoaderMode,
        extension: DatasetType = DatasetType.SAFETENSOR,
    ):
        logger.info("Creating dataloader")
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
        self.captures = sorted(set(self.data_path.rglob(f"*{self.extension.value}")))
        (
            self.capture_windows,
            self.window_sequence_map,
            self.capture_sequences,
        ) = self._generate_frame_indexes(self.captures, self.recurrent_samples)
        self.frame_indexes = [
            (k, *indices) for k, v in self.capture_windows.items() for indices in v
        ]

        # Augmentations disabled for test and validation, otherwise use config
        self.augment = (
            False
            if loader_mode != DataLoaderMode.TRAIN
            else self.config_params.dataset.gt_augmentation
        )

        # TODO: Bug if a st exists but has less than recurrent_samples frames, it will raise missing
        if not self.frame_indexes:
            raise ValueError(
                f"No {extension.value} files found at path {self.data_path} "
                f"Potential causes: Empty dataset directory or incorrect file extension used."
            )

        # Generate hash-maps keyed by sequence-aware identifiers so the model knows when
        # to flush history buffers. Training keeps the existing per-window salt to force
        # resets even when augmentation shuffles windows; evaluation hashes only by sequence.
        to_hash = lambda x: torch.tensor(hash(x), dtype=torch.int64).view(1, 1)
        if self.loader_mode != DataLoaderMode.TEST:
            # Training: combine the sequence hash with the window start index so buffers
            # reset for every sampled window, even though the underlying sequence ID is preserved.
            self.hashes = {
                k: {
                    start_idx: to_hash(
                        (
                            self._sequence_hash_value(
                                k, self.window_sequence_map[k][start_idx]
                            ),
                            start_idx,
                        )
                    )
                    for (start_idx, _) in v
                }
                for k, v in self.capture_windows.items()
            }
            all_hashes = [
                h for hash_dict in self.hashes.values() for h in hash_dict.values()
            ]
        else:
            # Eval/Test: use only the sequence hash so history persists within a sequence
            # but resets exactly at camera cuts.
            self.hashes = {
                k: {
                    start_idx: to_hash(
                        self._sequence_hash_value(
                            k, self.window_sequence_map[k][start_idx]
                        )
                    )
                    for (start_idx, _) in v
                }
                for k, v in self.capture_windows.items()
            }
            all_hashes = [
                h for hash_dict in self.hashes.values() for h in hash_dict.values()
            ]

        # Validate no duplicate hashes in training data
        if self.loader_mode != DataLoaderMode.TEST:
            hash_values = [int(h.item()) for h in all_hashes]
            if len(hash_values) != len(set(hash_values)):
                msg = "Duplicate sequence hashes found in training data"
                logger.error(msg)
                raise ValueError(msg)

        # Only used for test, store data
        self.existing_seq_path = None
        self.data_frame = None

    def __len__(self):
        return len(self.frame_indexes)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # Extract dataset and sliding window indices
        capture_path, start_idx, stop_idx = self.frame_indexes[idx]

        # Load data from Safetensors file
        data_frame = self._load_data(capture_path, start=start_idx, stop=stop_idx)

        # Append the unique sequence identifier
        data_frame["seq"] = self._get_seq_id(capture_path, start_idx)

        # Process the data to apply tonemapping, augmentations, etc.
        x, y = process_nss_data(
            data_frame,
            augment=self.augment,
            exposure=self.config_params.dataset.exposure,
            tonemapper=self.config_params.dataset.tonemapper,
        )

        return x, y

    def health_check(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_passes_required: float = 0.9,
    ):
        """Health check mvs and jitter on training dataset
        Args:
            dataloader: PyTorch dataloader containing a dataset
            num_passes_required (float): percentage expected to pass before allowing training
        """
        if self.loader_mode != DataLoaderMode.TRAIN:
            logger.info("DATASET: Health check is only supported for the train dataset")
            return

        logger.info("DATASET: Running Health Check...")

        (x, y) = next(iter(dataloader))

        batch_size = y.shape[0]
        time_steps = y.shape[1]
        total_examples = (time_steps - 1) * batch_size
        required_passes = int(total_examples * num_passes_required)

        test_results = {"flow_passed": 0, "jitter_passed": 0}

        with tqdm(total=time_steps - 1) as pbar:
            for t in range(time_steps):
                if t == 0:
                    gt_tm1 = y[:, 0, ...]
                    continue

                # Grab Assets to test
                gt = y[:, t, ...]
                motion = x["motion"][:, t, ...]
                jittered = x["colour_linear"][:, t, ...]
                jitter = x["jitter"][:, t, ...]

                # Check Flow by warping GT t-1 -> t and comparing with t
                warped = DenseWarp()([gt_tm1, motion])

                psnr_before_warp = []
                psnr_after_warp = []

                for i in range(batch_size):
                    psnr_before_warp.append(psnr(gt[i], gt_tm1[i], data_range=1.0))
                    psnr_after_warp.append(psnr(gt[i], warped[i], data_range=1.0))

                psnr_before_warp = torch.stack(psnr_before_warp, dim=0)
                psnr_after_warp = torch.stack(psnr_after_warp, dim=0)

                # Handle static scene special case where sequences PSNR before is `inf`
                # but due to mv noise `psnr_after_warp` is very high value and not `inf`
                static_pass_cond = torch.logical_and(
                    torch.isinf(psnr_before_warp),
                    psnr_after_warp > HIGH_PSNR_FOR_STATIC_WARP,
                )

                passed_cases = torch.where(
                    torch.logical_or(
                        psnr_after_warp >= psnr_before_warp, static_pass_cond
                    ),
                    1.0,
                    0.0,
                )

                test_results["flow_passed"] += torch.sum(passed_cases).numpy()

                # Check Jitter by unjittering low-res frame and compare to downsampled GT
                gt_lr = DownSampling2D(size=2.0, interpolation="bilinear")(gt)
                unjittered = DenseWarp()([jittered, jitter])

                psnr_jittered = []
                psnr_unjittered = []

                for i in range(batch_size):
                    psnr_jittered.append(psnr(gt_lr[i], jittered[i], data_range=1.0))
                    psnr_unjittered.append(
                        psnr(gt_lr[i], unjittered[i], data_range=1.0)
                    )

                psnr_jittered = torch.stack(psnr_jittered, dim=0)
                psnr_unjittered = torch.stack(psnr_unjittered, dim=0)

                passed_cases = torch.where(psnr_unjittered >= psnr_jittered, 1.0, 0.0)
                test_results["jitter_passed"] += torch.sum(passed_cases).numpy()

                # Store previous GT
                gt_tm1 = gt

                # Update TQDM
                pbar.update(1)
                pbar.set_postfix_str(test_results)

        assert (
            test_results["flow_passed"] > required_passes
        ), "Flow failed to pass all warping tests"

        assert (
            test_results["jitter_passed"] > required_passes
        ), "Jitter failed to pass all warping tests"

    def _generate_frame_indexes(
        self, captures: List[Path], n_frames: int
    ) -> Tuple[
        Dict[Path, List[Tuple[int, int]]],
        Dict[Path, Dict[int, int]],
        Dict[Path, List[Tuple[int, int]]],
    ]:
        """Generate sliding windows plus sequence metadata for each safetensor capture."""
        capture_windows: Dict[Path, List[Tuple[int, int]]] = {}
        window_sequence_map: Dict[Path, Dict[int, int]] = {}
        capture_sequences: Dict[Path, List[Tuple[int, int]]] = {}
        for capture in captures:
            with safetensors.safe_open(capture, framework="pt") as f:
                metadata = f.metadata()
                seq_length = int(metadata["Length"])
                camera_cut_sequences = self._compute_sequences(
                    capture, f, seq_length, n_frames
                )
                capture_sequences[capture] = camera_cut_sequences
                capture_windows[capture] = []
                window_sequence_map[capture] = {}
                for seq_idx, (seq_start, seq_end) in enumerate(camera_cut_sequences):
                    max_start = seq_end - (n_frames - 1)
                    for start in range(seq_start, max_start):
                        stop = start + n_frames
                        capture_windows[capture].append((start, stop))
                        window_sequence_map[capture][start] = seq_idx
        return capture_windows, window_sequence_map, capture_sequences

    def _compute_sequences(
        self, capture_path: Path, safetensor_file, seq_length: int, n_frames: int
    ) -> List[Tuple[int, int]]:
        """Return contiguous sequences without mid-sequence camera cuts."""
        if "camera_cut" not in safetensor_file.keys():
            # Legacy safetensors lack camera cut metadata; treat the entire capture as a
            # single sequence.
            return [(0, seq_length)]
        camera_cut_slice = safetensor_file.get_slice("camera_cut")
        camera_cut_tensor = torch.tensor(camera_cut_slice[:], dtype=torch.bool)
        flags = camera_cut_tensor.view(camera_cut_tensor.shape[0], -1)[:, 0]
        sequences: List[Tuple[int, int]] = []
        start = 0
        for idx in range(seq_length):
            if idx == start:
                continue
            if bool(flags[idx]):
                if idx - start >= n_frames:
                    sequences.append((start, idx))
                start = idx
        if seq_length - start >= n_frames:
            sequences.append((start, seq_length))
        if not sequences:
            # If we saw camera_cut metadata but never found a long-enough sequence, the capture
            # must be entirely shorter than recurrent_samples between cuts. Log and skip it.
            logger.error(
                f"Capture {capture_path} has no sequences satisfying recurrent_samples={n_frames}",
            )
        return sequences

    @staticmethod
    def _sequence_hash_value(capture_path: Path, sequence_idx: int) -> int:
        return hash((str(capture_path), sequence_idx))

    def _get_seq_id(self, capture_path: Path, start_idx: int) -> torch.Tensor:
        hash_tensor = self.hashes[capture_path][start_idx]
        return hash_tensor.expand(self.recurrent_samples, -1)

    def _load_data(
        self, capture_path: Path, start: int, stop: int
    ) -> Dict[str, torch.Tensor]:
        """Lazily open Safetensors file, only load data we need based on `self.features_to_read`
        and extract recurrent window between start:stop indexes.
        """
        data_frame = {}
        with safetensors.safe_open(capture_path, framework="pt", device="cpu") as f:
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
            if "exposure" not in data_frame:
                # If exposure missing, set to 0 (i.e. exp(0) = 1.0)
                data_frame["exposure"] = 0.0 * torch.ones(
                    (stop - start, 1), dtype=torch.float32
                )

        # Height and weight are hardcoded to match the slang shaders
        data_frame["motion_lr"] = fixed_normalize_mvs(
            data_frame["motion_lr"], height=544.0, width=960.0
        )
        return data_frame
