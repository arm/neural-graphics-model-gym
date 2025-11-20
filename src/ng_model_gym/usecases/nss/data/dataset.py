# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
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
        self.sequences = sorted(set(self.data_path.rglob(f"*{self.extension.value}")))
        self.sequence_data = self._generate_frame_indexes(
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

        # TODO: Bug if a st exists but has less than recurrent_samples frames, it will raise missing
        if not self.frame_indexes:
            raise ValueError(
                f"No {extension.value} files found at path {self.data_path} "
                f"Potential causes: Empty dataset directory or incorrect file extension used."
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
