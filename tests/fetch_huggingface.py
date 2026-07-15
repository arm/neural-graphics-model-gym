# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import os
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


def nfru_test_assets_enabled() -> bool:
    """Return whether optional NFRU test asset validation is enabled."""
    return os.getenv("NGMG_ENABLE_NFRU_TEST_ASSETS") == "1"


def download_pretrained_nss_weights():
    """Download pretrained NSS weights .pt files."""
    weights_dir = Path("tests/usecases/nss/weights")

    hf.snapshot_download(
        repo_id="Arm/neural-super-sampling",
        allow_patterns=["*.pt", "config.json"],
        local_dir=weights_dir,
        revision="3feb49cb7ee5aa295914a17c5878ffea693da8a8",
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
            Path("v0_1/nss_v0.1.0_fp32.pt"),
            Path("v0_1/nss_v0.1.1_int8.pt"),
            Path("nss_v1_high_fp32.pt"),
            Path("nss_v1_high_int8.pt"),
            Path("nss_v1_mid_low_int8.pt"),
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


def validate_nfru_datasets(dataset_dir: Path):
    """Validate local NFRU safetensor datasets provisioned via git-lfs."""
    # TODO: modify once NFRU datasets are available on HuggingFace
    # and switch to HF API for validation instead of local file checks.

    try:
        assert (
            dataset_dir.exists() and dataset_dir.is_dir()
        ), f"Missing NFRU dataset directory: {dataset_dir}"
        safetensors_files = sorted(dataset_dir.glob("*.safetensors"))
        assert safetensors_files, "No .safetensors files found for NFRU tests"

        expected_file = dataset_dir / "0000.safetensors"
        assert (
            expected_file.exists()
        ), "Missing expected NFRU sample file: 0000.safetensors"
        size = expected_file.stat().st_size
        assert (
            size > 512
        ), "NFRU sample file appears to be a git-lfs pointer. Run 'git lfs pull'"

    except AssertionError as e:
        raise type(e)(
            f"{e}\n\nNFRU test asset validation is disabled by default. "
            "Provision the NFRU test assets locally and rerun with "
            "NGMG_ENABLE_NFRU_TEST_ASSETS=1 to re-enable this validation path."
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

    if nfru_test_assets_enabled():
        nfru_datasets_path = Path("tests/usecases/nfru/datasets")
        # TODO: download NFRU datasets once available on HuggingFace
        validate_nfru_datasets(nfru_datasets_path)
        create_mini_safetensor_dataset(
            nfru_datasets_path,
            usecase_name="NFRU",
        )
