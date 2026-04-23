# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


class SafetensorsFeatureIterator:
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

    @staticmethod
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
        return json.loads(
            clean_json_string, object_hook=convert_string_constants_to_floats
        )


class EXRDatasetReader(SafetensorsFeatureIterator):
    """
    A SafetensorsFeatureIterator which handles EXR files.

    This is a container for common code shared between the EXR-based dataset
    readers. Not intended to be instantiated directly.
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
        super().__init__(src_root, dst_root, seq_id, seq_path, args, version)

        # EXR-based dataset readers are controlled using JSON files
        json_metadata = self.src_root / f"{self.seq_path}.json"
        self.metadata = self.load_metadata(json_metadata)

    def __iter__(self):
        return super().__iter__()

    def __next__(self) -> List[Tuple[Path, dict]]:
        return super().__next__()

    def make_image_dict(
        self, subdir_paths: List[Path], glob: str = "*.exr"
    ) -> Dict[Path, List[Path]]:
        """
        Takes a list of subdirectories (subdir_paths). Each is assumed:
        (a) to be rooted in a particular directory (src_root passed to ctor)
        (b) to have a common trailing path (seq_path passed to ctor)

        Returns a dict mapping:
        - FROM subdirectory names as supplied in subdir_paths
        - TO a list of paths representing EXR files in the directory in question.
        """
        return {
            subdir_path: sorted(
                (self.src_root / subdir_path / self.seq_path).rglob(glob)
            )
            for subdir_path in subdir_paths
        }

    def make_unique_seq_id(self, max_frames: int) -> str:
        """
        Returns a unique sequence ID based upon the maximum number of frames to
        be processed (max_frames).
        """
        unique_seq_id = hash(
            str(f"{self.src_root}_{self.seq_path}_{self.seq_id}_{max_frames}")
        )
        assert (
            unique_seq_id != 0
        ), f"Generated a `0` seq ID, which is reserved as special case, on seq: {self.seq_path}"

        return unique_seq_id
