# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path
from typing import List

import safetensors
import torch

from ng_model_gym.core.data import generic_safetensors_reader
from scripts.safetensors_generator.dataset_readers.safetensors_feature_iterator import (
    SafetensorsFeatureIterator,
)


class CropSafetensorsNSS(SafetensorsFeatureIterator):
    """Pre-crop NSS safetensors files for training"""

    def __init__(
        self,
        src_root: Path,
        dst_root: Path,
        seq_id: int,
        seq_path: Path,
        args: argparse.Namespace = None,
    ):
        # pylint: disable=duplicate-code
        super().__init__(
            src_root=src_root,
            dst_root=dst_root,
            seq_id=seq_id,
            seq_path=seq_path,
            args=args,
        )

        self.crop_sz = args.crop_size

    @staticmethod
    def _generic_safetensors_indexer(sequences: List[Path]):
        """
        Returns a list of indexes for a given list of sequence paths
        """
        frame_indexes = {}
        num_frames = 0
        for seq in sequences:
            with safetensors.safe_open(seq, framework="pt") as f:
                metadata = f.metadata()
                seq_length = int(metadata["Length"])
                frame_indexes[seq] = list(range(0, seq_length))
            num_frames += seq_length
        return [
            (k, indices) for k, v in frame_indexes.items() for indices in v
        ], num_frames

    def __iter__(self):
        self.file_indexes, self.num_frames = self._generic_safetensors_indexer(
            [self.seq_path]
        )
        self.idx = 0

        return self

    def __next__(self):
        if self.idx >= self.num_frames:
            raise StopIteration

        seq_path, file_idx = self.file_indexes[self.idx]

        parsed_feature = generic_safetensors_reader(seq_path, file_idx)

        # Read Image Dimensions
        inDims = parsed_feature["render_size"]
        in_height, in_width = inDims[0], inDims[1]
        min_spatial_size = int(sum(inDims).numpy())

        outDims = parsed_feature["outDims"]
        out_height, out_width = outDims[0], outDims[1]

        scale = float(parsed_feature["scale"].numpy().squeeze())

        # Derive Crop Sizes for HxW
        def derive_crops(dim_size, crop_sz):
            # Find all non-overlapping crops
            crops = [
                (i * crop_sz, i * crop_sz + crop_sz) for i in range(dim_size // crop_sz)
            ]
            # Overlap the final crop if there's a remainder
            if dim_size % crop_sz != 0:
                crops.append((dim_size - crop_sz, dim_size))
            return crops

        h_crops = derive_crops(out_height, self.crop_sz)
        w_crops = derive_crops(out_width, self.crop_sz)
        total_crops = len(h_crops) * len(w_crops)

        safetensor_path = Path(seq_path).stem

        crop_id = self.seq_id * total_crops
        if self.max_frames == 0:
            self.max_frames = total_crops

        output_list = []
        for id_y, (h_start, h_end) in enumerate(h_crops):
            for id_x, (w_start, w_end) in enumerate(w_crops):
                # Crop the tensors that can be
                out_features = {}
                for name, tensor in parsed_feature.items():
                    if sum(tensor.shape) >= min_spatial_size:
                        # high-res feature
                        if (
                            tensor.shape[1] == out_height
                            and tensor.shape[2] == out_width
                        ):
                            out_features[name] = tensor[
                                :, h_start:h_end, w_start:w_end
                            ].unsqueeze(0)
                        # half-res feature (e.g. depth, etc.)
                        elif (
                            tensor.shape[1] == in_height and tensor.shape[2] == in_width
                        ):
                            out_features[name] = tensor[
                                :,
                                int(h_start // scale) : int(h_end // scale),
                                int(w_start // scale) : int(w_end // scale),
                            ].unsqueeze(0)
                        # NOTE: Assuming there is no quarter res data (flow is now scaled to 540p)
                        else:
                            out_features[name] = tensor.unsqueeze(0)
                    else:
                        out_features[name] = tensor.unsqueeze(0)

                # Construct file path for crop
                crop_relative_seq_dir = self.dst_root / f"{crop_id:04d}"

                # Written to file in format seq_number/crop_id/seq_number.safetensors
                # This is so we can easily filter sequences for training
                crop_path = (
                    Path(safetensor_path)
                    / f"{crop_id:04d}"
                    / f"{safetensor_path}.safetensors"
                )

                # Overwrite `seq_id` with unique ID for this particular crop index
                # NOTE: `hash` is not consistent across python versions
                seq_id = hash(str(crop_relative_seq_dir))
                out_features["seq_id"] = torch.tensor(
                    seq_id, dtype=torch.int64
                ).reshape((1, 1))

                out_features["crop_id_y"] = torch.tensor(
                    id_y, dtype=torch.int64
                ).reshape((1, 1))
                out_features["crop_id_x"] = torch.tensor(
                    id_x, dtype=torch.int64
                ).reshape((1, 1))
                out_features["crop_sz"] = torch.tensor(
                    self.crop_sz, dtype=torch.float32
                ).reshape((1, 1))

                # Path and crops to write
                output_list.append((crop_path, out_features))

                crop_id += 1

        self.idx += 1

        return output_list
