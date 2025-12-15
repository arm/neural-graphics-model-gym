# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import safetensors
import torch

from ng_model_gym.core.data.utils import tonemap_inverse
from ng_model_gym.core.model.graphics_utils import swizzle
from ng_model_gym.core.utils.exr_utils import read_exr_torch
from scripts.safetensors_generator.fsr2_methods import depth_to_view_space_params


def validate_exr_dataset_structure(
    src_root: Path, seq_path: Path, scale: float
) -> None:
    """Check the dataset is in the expected format."""
    scale_str = str(float(scale)).replace(".", "_").replace("_0", "")

    required_dirs = [
        src_root / "ground_truth" / seq_path,
        src_root / "motion_gt" / seq_path,
        src_root / f"x{scale_str}" / "color" / seq_path,
        src_root / f"x{scale_str}" / "depth" / seq_path,
        src_root / f"x{scale_str}" / "motion" / seq_path,
    ]

    for required_dir in required_dirs:
        if not required_dir.is_dir():
            raise FileNotFoundError(f"Missing {required_dir} in expected structure.")
        if not any(required_dir.iterdir()):
            raise ValueError(f"{required_dir} is empty.")


def create_depth_params(data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Create the depth parameters"""
    make_image_like = lambda t: t.unsqueeze(-1).unsqueeze(-1)
    depth_params = depth_to_view_space_params(
        zNear=make_image_like(data["zNear"]),
        zFar=make_image_like(data["zFar"]),
        FovY=make_image_like(data["FovY"]),
        infinite=make_image_like(data["infinite_zFar"]).to(torch.bool),
        renderSizeWidth=make_image_like(data["render_size"])[:, 1:2, ...],
        renderSizeHeight=make_image_like(data["render_size"])[:, 0:1, ...],
        inverted=data["ReverseZ"],
    ).squeeze()
    return depth_params


def load_metadata(path: Union[str, Path]) -> Dict[Any, Any]:
    """
    Load the JSON metadata, parse and clean.
    """
    # 1) Read the raw text from the file
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Define a helper function for the regex replacement
    def replace_nan_or_inf_literals(match: str) -> Union[None, float, None]:
        """
        Replaces unquoted 'nan', 'infinity', '-infinity' literals in JSON with
        their quoted string equivalents ("NaN", "Infinity", "-Infinity").
        This makes the JSON valid for initial parsing.
        """
        prefix = match.group(1)  # E.g., ": "
        value_literal = match.group(2).lower()  # E.g., "nan", "-infinity"
        suffix = match.group(3)  # E.g., ", ", "}"

        if "nan" in value_literal:
            return f'{prefix}"NaN"{suffix}'
        if "inf" in value_literal:
            # Handle +/- infinity correctly
            if value_literal.startswith("-"):
                return f'{prefix}"-Infinity"{suffix}'
            return f'{prefix}"Infinity"{suffix}'
        return match.group(0)  # Fallback, should not be hit if regex is precise

    # 2) Pre-process the raw JSON string using regex.
    # This specifically targets unquoted 'nan', 'inf', 'infinity' (and their signed versions)
    # that appear as JSON values. It replaces them with *quoted* strings.
    # The regex avoids the "look-behind requires fixed-width pattern" error by
    # capturing the preceding context (like '": ') and re-inserting it.
    clean_json_string = re.sub(
        r"([:,\[]\s*)([+-]?(?:nan|inf(?:inity)?))\b(\s*(?:,|\}))",
        replace_nan_or_inf_literals,
        raw,
        flags=re.IGNORECASE,
    )

    # 3) Define an object hook to convert the *parsed string* constants back into Python floats.
    def convert_string_constants_to_floats(obj: Any) -> Any:
        """
        Converts specific string values (like "NaN", "Infinity") from JSON parsing
        into their respective Python float equivalents (float('nan'), float('inf')).
        """
        if isinstance(obj, str):
            obj_lower = obj.lower()
            if obj_lower == "nan":
                return float("nan")
            if obj_lower == "infinity":
                return float("inf")
            if obj_lower == "-infinity":
                return float("-inf")
        return obj

    # 4) Load the cleaned JSON string, applying the object hook for final type conversion.
    return json.loads(clean_json_string, object_hook=convert_string_constants_to_floats)


def generic_safetensors_reader(seq_path: Path, idx: int) -> dict:
    """Safetensors reader to return a dictionary of tensors"""
    data_frame = {}
    with safetensors.safe_open(seq_path, framework="numpy", device="cpu") as f:
        for k in f.keys():
            data_frame[k] = torch.from_numpy(f.get_slice(k)[idx])

    return data_frame


def generic_safetensors_indexer(sequences: List[Path]):
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
    return [(k, indices) for k, v in frame_indexes.items() for indices in v], num_frames


class FeatureIterator:
    """
    Class for reading (and processing if needed) data we write to Safetensors
    It's formed as an iterator, where:
    - `__init__` is used to give top-level configuration.
    - `__iter__` is expected to find all data that needs to be written and construct iterators
    - `__next__` returns a list of tuples packed as `(dst_file_path, features)`.
        This method is expected to perform:
        - Reading of data
        - Processing of data, e.g., cropping, tonemapping, etc.,
        - Deriving the destination to write each `feature` (relative to `dst_root`)
    """

    def __init__(
        self,
        src_root: Path,
        dst_root: Path,
        seq_id: int,
        seq_path: Path,
        args: argparse.Namespace = None,
        version: str = "unknown",
    ):
        self.src_root = src_root
        self.dst_root = dst_root
        self.seq_id = seq_id
        self.seq_path = seq_path
        self.args = args
        self.version = version
        self.max_frames = 0

    def __iter__(self):
        """Initialises iterator, e.g., finds all file paths
        Typically, returns `self`
        """
        raise NotImplementedError

    def __next__(self) -> List[Tuple[Path, dict]]:
        """Reads all relevant data, performs any processing,
        returns a list of `(dst_file_path, features)` where `dst_file_path` is expected
        to contain `seq_path` and `features` is a `dict` of tensors.
        """
        raise NotImplementedError


class NSSEXRDatasetReader(FeatureIterator):
    """Dataset reader for EXR files v1.0.1 spec."""

    def __init__(
        self,
        src_root: Path,
        dst_root: Path,
        seq_id: int,
        seq_path: Path,
        args: argparse.Namespace = None,
        scale: float = 2.0,
    ):
        super().__init__(
            src_root=src_root,
            dst_root=dst_root,
            seq_id=seq_id,
            seq_path=seq_path,
            args=args,
            version=type(self).__name__,
        )
        self.scale = scale
        # 2.0 -> 2, 1.5 -> 1_5, etc.
        self.scale_str = str(float(self.scale)).replace(".", "_").replace("_0", "")

        validate_exr_dataset_structure(self.src_root, self.seq_path, self.scale_str)

    def __iter__(self):
        # Dataset Binaries
        self.seq_data = {
            f"x{self.scale_str}/depth": None,
            f"x{self.scale_str}/motion": None,
            f"x{self.scale_str}/color": None,
            "motion_gt": None,
            "ground_truth": None,
        }

        # Create Dataset Paths for each data-types
        self.seq_data = {
            k: sorted((self.src_root / k / self.seq_path).rglob("*.exr"))
            for k in self.seq_data.keys()
        }

        self.max_frames = min(len(self.seq_data[k]) for k in self.seq_data)

        # Metadata
        json_metadata = self.src_root / f"{self.seq_path}.json"
        self.metadata = load_metadata(json_metadata)

        # Create Unique Seq ID
        self.unique_seq_id = hash(
            str(f"{self.src_root}_{self.seq_path}_{self.seq_id}_{self.max_frames}")
        )
        assert (
            self.unique_seq_id != 0
        ), f"Generated a `0` seq ID, which is reserved as special case, on seq: {self.seq_path}"

        # Create the iterator
        self.count = 0

        return self

    def __next__(self):
        if self.count >= self.max_frames:
            raise StopIteration

        # Feature to return
        out_features = {}

        # Read EXR data for current frame
        depth_raw = read_exr_torch(
            self.seq_data[f"x{self.scale_str}/depth"][self.count],
            dtype=np.float32,
            channels="R",
        )
        mv_lr_raw = read_exr_torch(
            self.seq_data[f"x{self.scale_str}/motion"][self.count],
            dtype=np.float16,
            channels="RG",
        )
        colour_raw = read_exr_torch(
            self.seq_data[f"x{self.scale_str}/color"][self.count],
            dtype=np.float16,
            channels="RGB",
        )
        mv_raw = read_exr_torch(
            self.seq_data["motion_gt"][self.count], dtype=np.float16, channels="RG"
        )
        truth_raw = read_exr_torch(
            self.seq_data["ground_truth"][self.count], dtype=np.float16, channels="RGB"
        )
        frame_meta_data = self.metadata["Frames"][self.count]

        # Render sizes
        out_features["render_size"] = render_size = torch.tensor(
            [colour_raw.shape[2], colour_raw.shape[3]], dtype=torch.int32
        ).reshape((1, 2))
        out_features["outDims"] = torch.tensor(
            [truth_raw.shape[2], truth_raw.shape[3]], dtype=torch.int32
        ).reshape((1, 2))

        # View Projection matrix is correct, no need for transposition (fixed after 24_10)
        out_features["viewProj"] = torch.reshape(
            torch.tensor(frame_meta_data["ViewProjection"]), (1, 1, 4, 4)
        )

        # Transform motion vectors to PyTorch format
        def process_motion(motion: torch.tensor) -> torch.tensor:
            # clamp to [-1, 1]
            motion = torch.clamp(
                motion, -torch.ones_like(motion), torch.ones_like(motion)
            )
            # switch uv -> vu
            motion = swizzle(motion, "yx")
            # scale from uv to pixels
            size = torch.tensor(
                [motion.shape[2], motion.shape[3]], dtype=torch.float16
            ).reshape((1, 2, 1, 1))
            motion = motion * size
            return motion

        out_features["motion_lr"] = process_motion(mv_lr_raw)
        out_features["motion"] = process_motion(mv_raw)

        reverse_z = self.metadata["ReverseZ"]
        out_features["ReverseZ"] = torch.tensor(reverse_z, dtype=torch.bool).reshape(
            (1, 1)
        )
        if reverse_z:
            # Invert depth
            depth_raw = 1.0 - depth_raw.to(torch.float32)
            out_features["depth"] = depth_raw
        else:
            # Depth is usable as-is, **should not be inverted**
            out_features["depth"] = depth_raw.to(torch.float32)

        # Transform colour data
        out_features["colour_linear"] = colour_raw
        if not self.args.linear_truth:
            truth_raw = tonemap_inverse(truth_raw, mode="karis")
        out_features["ground_truth_linear"] = truth_raw

        # Depth planes
        out_features["infinite_zFar"] = torch.tensor(
            frame_meta_data["CameraFarPlane"] == -1, dtype=torch.bool
        ).reshape((1, 1))

        # This is done for backwards compatability sake, for when
        # we are confronted with certain game engines' infinite FarPlane.
        z_far = (
            5000.0
            if out_features["infinite_zFar"]
            else frame_meta_data["CameraFarPlane"]
        )
        out_features["zFar"] = torch.tensor(z_far, dtype=torch.float32).reshape((1, 1))
        out_features["zNear"] = torch.tensor(
            frame_meta_data["CameraNearPlane"], dtype=torch.float32
        ).reshape((1, 1))

        # Add additional MetaData features
        out_features["seq"] = torch.tensor(
            self.unique_seq_id, dtype=torch.int64
        ).reshape((1, 1))
        out_features["img"] = torch.tensor(self.count, dtype=torch.int64).reshape(
            (1, 1)
        )
        out_features["FovX"] = torch.tensor(
            frame_meta_data["FovX"], dtype=torch.float32
        ).reshape((1, 1))
        out_features["FovY"] = torch.tensor(
            frame_meta_data["FovY"], dtype=torch.float32
        ).reshape((1, 1))
        out_features["scale"] = torch.tensor(self.scale, dtype=torch.float32).reshape(
            (1, 1)
        )

        # Exposure is an optional field, so we must ensure it exists before using it
        if "Exposure" in frame_meta_data.keys():
            # Exposure output from Engine is assumed to already be exponential.
            # We take a natural log here,
            # so this plays nicely with our codebase, which will do exp(exposure)
            exposure_log = torch.log(torch.tensor(float(frame_meta_data["Exposure"])))
            out_features["exposure"] = exposure_log.to(torch.float32).reshape((1, 1))

        # Convert scale factor to relevant index
        scale_idx = self.metadata["UpscalingRatiosIndices"][f"x{self.scale_str}_index"]

        # Jitter Offsets
        x, y = (
            frame_meta_data["NormalizedPerRatioJitter"][scale_idx]["X"],
            frame_meta_data["NormalizedPerRatioJitter"][scale_idx]["Y"],
        )
        jitter = torch.tensor([y, x], dtype=torch.float32).reshape((1, 2, 1, 1))
        out_features["jitter"] = jitter * render_size.reshape((1, 2, 1, 1))

        # Historic Dataset Correction
        # Originally datasets were captured from game engines
        # which did not correct for display space Y-orientation
        # Game engines we use now correct for this, so we align by undoing this.
        out_features["Y"] = torch.tensor(-y, dtype=torch.float32).reshape((1, 1))
        out_features["X"] = torch.tensor(x, dtype=torch.float32).reshape((1, 1))

        # Depth params for FSR-style disocclusion mask
        out_features["depth_params"] = (
            create_depth_params(out_features).to(torch.float32).reshape((1, 4))
        )

        # Output data
        out_filepath = f"{Path(self.seq_path)}.safetensors"

        # Update Count
        self.count += 1

        return [(out_filepath, out_features)]


class CropSafetensors(FeatureIterator):
    """Pre-crop NSS safetensors files for training"""

    def __init__(
        self,
        src_root: Path,
        dst_root: Path,
        seq_id: int,
        seq_path: Path,
        args: argparse.Namespace = None,
    ):
        super().__init__(
            src_root=src_root,
            dst_root=dst_root,
            seq_id=seq_id,
            seq_path=seq_path,
            args=args,
        )

        self.crop_sz = args.crop_size

    def __iter__(self):
        self.file_indexes, self.num_frames = generic_safetensors_indexer(
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


Dataset_Readers = {
    "EXRv101": NSSEXRDatasetReader,
    "cropper": CropSafetensors,
}
