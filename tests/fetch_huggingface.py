# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import huggingface_hub as hf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from scripts.safetensors_generator.safetensor_truncate import truncate_safetensor_file


def download_pretrained_nss_weights():
    """Download pretrained NSS weights .pt files."""
    weights_dir = Path("tests/usecases/nss/weights")

    hf.snapshot_download(
        repo_id="Arm/neural-super-sampling",
        allow_patterns=["nss_*.pt", "config.json"],
        local_dir=weights_dir,
        revision="2e9b606acd9fa25071825a12f0764f1c3bef9480",
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
        expected_weights = ["nss_v0.1.0_fp32.pt", "nss_v0.1.1_int8.pt"]
        for file_name in expected_weights:
            weights_path = weights_dir / file_name
            assert weights_path.exists(), f"Missing weight file: {file_name}"
            size = weights_path.stat().st_size
            assert (
                size > 100 * 1024
            ), f"Weight file {file_name} is less than 100KB ({size:.1f} bytes)"

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
            f"{e}\n\nRun 'hatch run test-download' to fetch test assets."
        ) from e


def create_mini_safetensor_dataset(original_dataset_path: Path):
    """Create a smaller NSS dataset for quicker integration tests."""
    mini_dataset_root = original_dataset_path.parent / "mini_datasets"
    dataset_names = ("train", "test", "val")

    try:
        for subset in dataset_names:
            source_dir = original_dataset_path / subset
            if not source_dir.exists():
                raise FileNotFoundError(
                    f"Expected source dataset directory at {source_dir}"
                )

            safetensor_files = sorted(source_dir.glob("*.safetensors"))
            if not safetensor_files:
                raise FileNotFoundError(f"No .safetensors files found in {source_dir}")

            mini_dir = mini_dataset_root / subset
            mini_dir.mkdir(parents=True, exist_ok=True)

            for source_file in safetensor_files:
                target_file = mini_dir / f"{source_file.stem}.safetensors"
                truncate_safetensor_file(source_file, target_file, desired_frames=20)
                print(f"Created mini dataset at {target_file}")

    except FileNotFoundError as e:
        raise type(e)(
            f"{e}\n\nRun 'hatch run test-download' to fetch test assets."
        ) from e


if __name__ == "__main__":
    download_pretrained_nss_weights()
    nss_datasets_path = Path("tests/usecases/nss/datasets")
    download_nss_datasets(nss_datasets_path)
    validate_nss_downloads(nss_datasets_path)
    create_mini_safetensor_dataset(nss_datasets_path)
