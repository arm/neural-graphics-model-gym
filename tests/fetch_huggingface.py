# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import shutil
import sys
from pathlib import Path

import huggingface_hub as hf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from scripts.safetensors_generator.safetensor_truncate import truncate_safetensor_file

MINI_DATASET_DESIRED_FRAMES = 20
DATASET_SPLITS = ("train", "test", "val")
NFRU_DATASET_PREFIX = "nfru"


def download_pretrained_nss_weights():
    """Download pretrained NSS weights .pt files."""
    weights_dir = Path("tests/usecases/nss/weights")

    hf.snapshot_download(
        repo_id="Arm/neural-super-sampling",
        allow_patterns=["nss_*_*.pt", "config.json"],
        local_dir=weights_dir,
        revision="3dd6c4f054827a3d018330d5dfcd0b92e7d37974",
    )
    print(f"Downloaded pretrained NSS weights to {weights_dir}")


def download_nss_datasets(datasets_dir):
    """Download NSS test datasets .safetensors files."""

    hf.snapshot_download(
        repo_id="Arm/neural-graphics-dataset",
        allow_patterns=["*.safetensors"],
        repo_type="dataset",
        local_dir=datasets_dir,
        revision="42c92f5193aead4fd4281ce8ff5258f66b03ef10",
    )
    print(f"Downloaded datasets to {datasets_dir}")


def validate_nss_downloads(datasets_dir):
    """Validate NSS downloads from HF"""

    try:
        # Validate pretrained weights
        weights_dir = Path("tests/usecases/nss/weights")
        expected_weights = [
            Path("nss_v1_0_1_high_fp32.pt"),
            Path("nss_v1_0_1_high_int8.pt"),
            Path("nss_v1_0_1_mid_low_int8.pt"),
        ]
        for file_path in expected_weights:
            weights_path = weights_dir / file_path
            assert weights_path.exists(), f"Missing weight file: {file_path}"
            size = weights_path.stat().st_size
            assert (
                size > 100 * 1024
            ), f"Weight file {file_path} is less than 100KB ({size:.1f} bytes)"

        # Validate datasets
        folders = ["train", "test", "val"]
        for folder in folders:
            dataset_path = datasets_dir / folder
            assert (
                dataset_path.exists() and dataset_path.is_dir()
            ), f"Missing dataset directory: {folder}"
            safetensors = list(dataset_path.glob("*.safetensors"))
            assert safetensors, f"No .safetensors files found in {folder}"
            for safetensor in safetensors:
                size = safetensor.stat().st_size
                assert (
                    size > 25 * 1024 * 1024
                ), f"Dataset file {safetensor.name} in {folder} is less than 100KB ({size:.1f} KB)"

    except AssertionError as e:
        raise type(e)(
            f"{e}\n\nRun 'hatch run test:download' to fetch test assets."
        ) from e


def download_pretrained_nfru_weights():
    """Download pretrained NFRU v1 weights files."""
    weights_dir = Path("tests/usecases/nfru/weights")

    hf.snapshot_download(
        repo_id="Arm/neural-frame-rate-upscaling",
        allow_patterns=["*.pt", "config.json"],
        local_dir=weights_dir,
        revision="57eb1caaa98338b25d23450802f4adaac63010ce",
    )
    print(f"Downloaded pretrained NFRU v1 weights to {weights_dir}")


def download_nfru_datasets(datasets_dir):
    """Download NFRU test datasets .safetensors files."""

    if _nfru_flattened_dataset_exists(datasets_dir):
        print(f"NFRU v1 datasets already exist at {datasets_dir}, skipping download")
        return

    hf.snapshot_download(
        repo_id="Arm/neural-graphics-dataset",
        allow_patterns=[
            f"{NFRU_DATASET_PREFIX}/{split}/*.safetensors" for split in DATASET_SPLITS
        ],
        repo_type="dataset",
        local_dir=datasets_dir,
        revision="927bc07a1ad28f2a59c3d957d6b2f9e0f036fadb",
    )
    _flatten_nfru_dataset_download(datasets_dir)
    print(f"Downloaded NFRU v1 datasets to {datasets_dir}")


def _nfru_flattened_dataset_exists(datasets_dir: Path) -> bool:
    """Return True when flattened NFRU split directories already contain safetensors."""
    return all(
        (datasets_dir / split).is_dir()
        and any((datasets_dir / split).glob("*.safetensors"))
        for split in DATASET_SPLITS
    )


def _flatten_nfru_dataset_download(datasets_dir: Path):
    """
    Move downloaded nfru v1 dataset files into datasets_dir/{split}.

    E.g.    tests/usecases/nfru/datasets/nfru/train/0000.safetensors becomes
            tests/usecases/nfru/datasets/test/0000.safetensors
    """
    nfru_root = datasets_dir / NFRU_DATASET_PREFIX
    for split in DATASET_SPLITS:
        source_split_dir = nfru_root / split
        if not source_split_dir.is_dir():
            continue

        target_split_dir = datasets_dir / split
        target_split_dir.mkdir(parents=True, exist_ok=True)
        for source_file in source_split_dir.glob("*.safetensors"):
            source_file.replace(target_split_dir / source_file.name)

    if nfru_root.exists():
        shutil.rmtree(nfru_root)


def validate_nfru_downloads(dataset_dir: Path):
    """Validate local NFRU safetensor datasets provisioned via Hugging Face."""
    try:
        # Validate pretrained weights
        weights_dir = Path("tests/usecases/nfru/weights")
        expected_weights = [
            Path("nfru_v1_fp32.pt"),
            Path("nfru_v1_int8.pt"),
        ]
        for file_path in expected_weights:
            weights_path = weights_dir / file_path
            assert weights_path.exists(), f"Missing weight file: {file_path}"
            size = weights_path.stat().st_size
            assert (
                size > 100 * 1024
            ), f"Weight file {file_path} is less than 100KB ({size:.1f} bytes)"

        # Validate datasets
        folders = ["train", "test", "val"]
        for folder in folders:
            dataset_path = dataset_dir / folder
            assert (
                dataset_path.exists() and dataset_path.is_dir()
            ), f"Missing dataset directory: {folder}"
            safetensors = list(dataset_path.glob("*.safetensors"))
            assert safetensors, f"No .safetensors files found in {folder}"
            for safetensor in safetensors:
                size = safetensor.stat().st_size
                assert (
                    size > 25 * 1024 * 1024
                ), f"Dataset file {safetensor.name} in {folder} is less than 100KB ({size:.1f} KB)"

    except AssertionError as e:
        raise type(e)(
            f"{e}\n\nRun 'hatch run test:download' to fetch test assets."
        ) from e


def create_mini_safetensor_dataset(
    original_dataset_path: Path,
    usecase_name: str = "dataset",
):
    """Create a smaller split mini dataset from train/test/val safetensors."""
    mini_dataset_root = original_dataset_path.parent / "mini_datasets"

    try:
        if not original_dataset_path.exists():
            raise FileNotFoundError(
                f"Expected source dataset directory at {original_dataset_path}"
            )

        for split in DATASET_SPLITS:
            source_split_dir = original_dataset_path / split
            if not source_split_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected {usecase_name} split directory at {source_split_dir}"
                )

            safetensor_files = sorted(source_split_dir.glob("*.safetensors"))
            if not safetensor_files:
                raise FileNotFoundError(
                    f"No .safetensors files found in {source_split_dir}"
                )

            mini_split_dir = mini_dataset_root / split
            mini_split_dir.mkdir(parents=True, exist_ok=True)

            for source_file in safetensor_files:
                target_file = mini_split_dir / f"{source_file.stem}.safetensors"
                truncate_safetensor_file(
                    source_file, target_file, desired_frames=MINI_DATASET_DESIRED_FRAMES
                )
                print(f"Created {usecase_name} mini dataset at {target_file}")

    except FileNotFoundError as e:
        raise type(e)(
            f"{e}\n\nRun 'hatch run test:download' to fetch test assets."
        ) from e


if __name__ == "__main__":
    download_pretrained_nss_weights()

    nss_datasets_path = Path("tests/usecases/nss/datasets")
    download_nss_datasets(nss_datasets_path)
    validate_nss_downloads(nss_datasets_path)
    create_mini_safetensor_dataset(nss_datasets_path, usecase_name="NSS")

    download_pretrained_nfru_weights()

    nfru_datasets_path = Path("tests/usecases/nfru/datasets")
    download_nfru_datasets(nfru_datasets_path)
    validate_nfru_downloads(nfru_datasets_path)
    create_mini_safetensor_dataset(nfru_datasets_path, usecase_name="NFRU")
