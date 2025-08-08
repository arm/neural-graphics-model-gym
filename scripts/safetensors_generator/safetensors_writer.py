# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging
import os
import traceback
from datetime import datetime
from functools import partial
from multiprocessing import Pool, RLock
from pathlib import Path
from typing import Tuple, Union

import torch
from safetensors.torch import save_file
from tqdm import tqdm

from ng_model_gym.utils.logging import setup_logging
from scripts.safetensors_generator.dataset_reader import (
    FeatureIterator,
    NSSEXRDatasetReader,
)

logger = logging.getLogger("safetensors_writer")


class SafeAsync:
    """Wrapper to safely execute functions and log errors"""

    def __init__(self, func, seq):
        self.func = func
        self.seq = seq[1]

    def __call__(self, *args, **kwargs):
        result = None
        try:
            result = self.func(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Failed for sequence {self.seq}, with error: {e}")
            traceback.print_exc()

        return result


class SafetensorsWriter:
    """
    Class for writing Safetensors

    Most notably, it relies on auto-detection of sequence paths,
    which is heuristically driven based on `file_ext`

    Usually dumped sequences are in order of:
      src_root/asset_type/seq_name/file_name.extension (where, typically `extension = "exr"`)

    Existing Safetensors are:
      src_root/file_name.safetensors

    Pre-Cropped are:
      src_root/crop_id/file_name.safetensors

    For now, we will treat existing Safetensors as a special case, as <crop_id> is not guaranteed
    to be unique across existing datasets, this means:
     - For `extension == "safetensors"` we take directory relative to <src_root>
     - For `extension != "safetensors` we take .stem

    Args:
        src_dir: Top-level source directory where data to write into Safetensors exists
        dst_dir: Destination directory to write Safetensors
        file_ext: file extension for the source content
        sequence_iterator: `FeatureIterator` inherited class used to read data that we write
        args: Any extra arguments that will pass down to help configure `sequence_iterator`
        threads: Number of parallel threads to spawn when writing data, default is `os.cpu_count`
    """

    def __init__(
        self,
        src_dir: Union[str, Path],
        dst_dir: Union[str, Path],
        file_ext: str,
        sequence_iterator: FeatureIterator,
        args: argparse.Namespace = None,
        threads: int = None,
    ):
        self.src_root = Path(src_dir)
        if not self.src_root.exists():
            raise RuntimeError(f"Missing `args.src`: {self.src_root}")
        self.dst_root = Path(dst_dir) / Path(self.src_root.parts[-1])
        self.dst_root.mkdir(exist_ok=True, parents=True)
        self.file_ext = file_ext
        self.sequence_iterator = sequence_iterator

        # For now, all surplus arguments are stored to pass down to
        # `sequence_iterator` later if needed
        self.args = args

        # Number of threads to parallelize across
        self.threads = threads if threads is not None else os.cpu_count()

    def get_sequence_list_from_root(self) -> list:
        """
        By default expects
        For safetensors:
            src_root/file.safetensors
        For other extensions:
            src_root/asset/seq_name/file.extension
        """
        if self.file_ext == "safetensors":
            sequences = list(self.src_root.rglob("*.safetensors"))
        else:
            sequences = [
                f.parent.stem for f in self.src_root.rglob(f"*.{self.file_ext}")
            ]
        return sequences

    def write_data(self):
        """
        Dispatch Parallelized Safetensors writing
        Multi Threads writing across sequences,
            we do this at sequence level as each safetensor is per sequence
        """
        sequences = self.get_sequence_list_from_root()

        # Ensure sequences are unique with `list -> set -> list` cast, sorted for readability
        sequences = sorted(list(set(sequences)))

        # Inputs to Generic sequence generator are:
        #   (seq_num, seq_name), SafetensorWriter (assumed immutable)
        parallel_inps = enumerate(sequences)

        # Sequence Generator for parallel work
        seq_generator = partial(
            generic_sequence_writer,
            writer=self,
        )

        # Multi-Thread writing across sequences
        if self.threads > 1:
            # Multiprocessing tqdm: https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
            with Pool(
                processes=self.threads,
                initargs=(RLock(),),
                initializer=tqdm_initialiser,
            ) as seq_pool:
                for i, seq in enumerate(parallel_inps):
                    seq_pool.apply_async(
                        SafeAsync(seq_generator, seq),
                        args=(seq, i),
                    )
        else:
            # Single Threaded, one sequence at a time
            for seq_n in parallel_inps:
                seq_generator(seq_n, 0)


def tqdm_initialiser(*args, **kwargs):
    """Initialise tqdm with multiprocessing lock"""
    tqdm.set_lock(*args, **kwargs)


def generic_safetensors_writer(args):
    """Initialise and run SafetensorsWriter"""
    reader = NSSEXRDatasetReader

    writer = SafetensorsWriter(
        src_dir=args.src,
        dst_dir=args.dst,
        file_ext=args.extension,
        sequence_iterator=reader,
        args=args,
        threads=args.threads,
    )

    writer.write_data()


def write_safetensors(
    iterator: FeatureIterator, pid: int, writer: SafetensorsWriter, seq_id: int
):
    """Iterate over features and write to .safetensors"""
    # Allows for writing multiple 'sequences' from a single source,
    # used for separating crops into crop id directories
    sub_seq_dict = {}
    with tqdm(
        total=iterator.max_frames,
        desc=f"Sequence ID: {seq_id:04d}",
        colour="#249d3c",
        dynamic_ncols=True,
        position=pid + 1,
    ) as pbar:
        for features in iterator:
            for dst_file_path, feature in features:
                if dst_file_path not in sub_seq_dict:
                    sub_seq_dict[dst_file_path] = {}
                    sub_seq_dict[dst_file_path]["data"] = {
                        k: [feature[k]] for k in feature.keys()
                    }
                    sub_seq_dict[dst_file_path]["length"] = 0
                else:
                    for k in sub_seq_dict[dst_file_path]["data"].keys():
                        sub_seq_dict[dst_file_path]["data"][k].append(feature[k])
                pbar.update(1)
                sub_seq_dict[dst_file_path]["length"] += 1

    for sub_seq, sub_data in sub_seq_dict.items():
        logger.info(f"Writing {sub_seq} to file. This may take a while.")
        sf_file = (writer.dst_root / sub_seq).with_suffix(".safetensors")
        sf_file.parent.mkdir(parents=True, exist_ok=True)
        # TODO: Bit of a bottleneck atm, threads will hang on this
        raw_data = {
            k: torch.cat(sub_data["data"][k], axis=0) for k in sub_data["data"].keys()
        }
        metadata = {
            "Length": str(sub_data["length"]),
            "Created": datetime.now().strftime("%d-%m-%Y"),
            "Version": iterator.version,
        }
        save_file(raw_data, sf_file, metadata)


def check_if_safetensors_exist(writer, feature_iterator):
    """Check if Safetensors file already exists"""
    dst_file_path = next(iter(feature_iterator))[0][0]
    safetensors_path = (writer.dst_root / dst_file_path).with_suffix(".safetensors")
    return safetensors_path.is_file()


def generic_sequence_writer(
    seq_bundle: Tuple[int, str],
    pid: int,
    writer: SafetensorsWriter,
):
    """Worker method for writing Safetensors"""
    # Unpack Sequence information for this thread
    (seq_id, seq_path) = seq_bundle

    # Construct a `DataReader` iterator which will provide (dst_file_name, features_dict)
    feature_iterator = writer.sequence_iterator(
        writer.src_root, writer.dst_root, seq_id, seq_path, writer.args
    )

    # Check if safetensors exists
    if check_if_safetensors_exist(writer, feature_iterator):
        if writer.args.overwrite:
            logger.warning("Overwrite existing sequence")
        else:
            logger.warning(
                "Safetensors already exist - "
                f"Safetensors generation is skipped for sequence {seq_id:04d}"
            )
            return

    write_safetensors(feature_iterator, pid, writer, seq_id)


def main(args):
    """Entry point for Safetensors writer"""
    setup_logging(
        "safetensors_writer",
        logging.INFO,
        args.logging_output_dir,
        "safetensors_writer.log",
    )

    logger.info("Starting writer")

    generic_safetensors_writer(args)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    parser = argparse.ArgumentParser()
    parser.add_argument("-src", help="Path to source files", required=True)
    parser.add_argument(
        "-dst",
        help="Path to root folder of destination",
        default=Path("./output/safetensors"),
    )
    parser.add_argument(
        "-threads",
        help="Number of parallel threads",
        type=int,
        default=1,
        choices=list(range(1, os.cpu_count() + 1)),
    )
    parser.add_argument(
        "-extension", help="File Extension of src data", type=str, default="exr"
    )
    parser.add_argument(
        "-overwrite",
        help="Overwrite data in destination path",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-linear-truth",
        help="Whether the ground truth is linear or not - assumes Karis TM otherwise",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "-logging_output_dir", help="Path to output dir for logging", default="./output"
    )

    parsed_args = parser.parse_args()

    main(parsed_args)
