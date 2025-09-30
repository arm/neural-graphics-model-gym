# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import huggingface_hub as hf


def download_pretrained_weights():
    """Download pretrained weights .pt files."""
    weights_dir = Path("tests/usecases/nss/weights")

    hf.snapshot_download(
        repo_id="Arm/neural-super-sampling",
        allow_patterns=["nss_*.pt"],
        local_dir=weights_dir,
        revision="a90431d1ddf116ef713d39e2507f86550ed09793",
    )
    print(f"Downloaded pretrained weights to {weights_dir}")


def download_unit_test_datasets():
    """Download unit test datasets .safetensors files."""
    datasets_dir = Path("tests/usecases/nss/datasets")

    hf.snapshot_download(
        repo_id="Arm/neural-graphics-dataset",
        allow_patterns=["*.safetensors"],
        repo_type="dataset",
        local_dir=datasets_dir,
        revision="42c92f5193aead4fd4281ce8ff5258f66b03ef10",
    )
    print(f"Downloaded datasets to {datasets_dir}")


def validate_downloads():
    """Validate downloads from HF"""

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
    datasets_dir = Path("tests/usecases/nss/datasets")
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


if __name__ == "__main__":
    download_pretrained_weights()
    download_unit_test_datasets()
    validate_downloads()
