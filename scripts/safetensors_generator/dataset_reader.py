# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from ng_model_gym.usecases.nss.dataloader.utils import tonemap_inverse
from ng_model_gym.usecases.nss.model.graphics_utils import swizzle
from scripts.safetensors_generator.exr_utils import (
    create_depth_params,
    read_exr_torch,
    validate_exr_dataset_structure,
)


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
