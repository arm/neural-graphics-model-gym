# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import sys
from pathlib import Path

import huggingface_hub as hf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from scripts.safetensors_generator.safetensor_truncate import truncate_safetensor_file

NSS_MINI_DATASET_DESIRED_FRAMES = 20
NFRU_MINI_DATASET_DESIRED_FRAMES = 10
DATASET_SPLITS = ("train", "test", "val")
WEIGHTS_MIN_SIZE_BYTES = 100 * 1024
DATASET_MIN_SIZE_BYTES = 25 * 1024 * 1024

HF_REVISIONS = {
    "nss_weights": "3dd6c4f054827a3d018330d5dfcd0b92e7d37974",
    "nss_dataset": "42c92f5193aead4fd4281ce8ff5258f66b03ef10",
    "nfru_weights": "57eb1caaa98338b25d23450802f4adaac63010ce",
    "nfru_dataset": "927bc07a1ad28f2a59c3d957d6b2f9e0f036fadb",
}

USECASES = {
    "nss": {
        "weights_dir": Path("tests/usecases/nss/weights"),
        "datasets_dir": Path("tests/usecases/nss/datasets"),
        "weights_repo_id": "Arm/neural-super-sampling",
        "weights_allow_patterns": ["nss_v1_0_1_*.pt", "config.json"],
        "weights_revision": HF_REVISIONS["nss_weights"],
        "dataset_repo_id": "Arm/neural-graphics-dataset",
        "dataset_prefix": None,
        "dataset_revision": HF_REVISIONS["nss_dataset"],
        "expected_weights": [
            "nss_v1_0_1_high_fp32.pt",
            "nss_v1_0_1_high_int8.pt",
            "nss_v1_0_1_mid_low_int8.pt",
        ],
        "mini_dataset_frames": NSS_MINI_DATASET_DESIRED_FRAMES,
    },
    "nfru": {
        "weights_dir": Path("tests/usecases/nfru/weights"),
        "datasets_dir": Path("tests/usecases/nfru/datasets"),
        "weights_repo_id": "Arm/neural-frame-rate-upscaling",
        "weights_allow_patterns": ["nfru_v1_*.pt", "config.json"],
        "weights_revision": HF_REVISIONS["nfru_weights"],
        "dataset_repo_id": "Arm/neural-graphics-dataset",
        "dataset_prefix": "nfru",
        "dataset_revision": HF_REVISIONS["nfru_dataset"],
        "expected_weights": [
            "nfru_v1_fp32.pt",
            "nfru_v1_int8.pt",
        ],
        "mini_dataset_frames": NFRU_MINI_DATASET_DESIRED_FRAMES,
    },
}


def _flattened_dataset_exists(datasets_dir: Path) -> bool:
    """Return True when flattened split directories already contain safetensors."""
    return all(
        (datasets_dir / split).is_dir()
        and bool(_list_split_safetensors(datasets_dir / split))
        for split in DATASET_SPLITS
    )


def _list_split_safetensors(split_dir: Path) -> list[Path]:
    """List unique safetensors under one split tree without following symlinks."""
    safetensors = []
    seen_resolved_paths = set()

    for root, dirs, files in os.walk(split_dir, followlinks=False):
        root_path = Path(root)
        dirs[:] = [name for name in dirs if not (root_path / name).is_symlink()]

        for file_name in files:
            if not file_name.endswith(".safetensors"):
                continue
            file_path = root_path / file_name
            if file_path.is_symlink():
                continue

            resolved_path = file_path.resolve()
            if resolved_path in seen_resolved_paths:
                continue

            seen_resolved_paths.add(resolved_path)
            safetensors.append(file_path)

    return sorted(safetensors)


def _flatten_dataset_subdir(datasets_dir: Path, dataset_name: str) -> None:
    """Strip one dataset prefix dir while preserving all nested structure below it."""
    source_dir = datasets_dir / dataset_name
    if not source_dir.exists():
        return

    for item in source_dir.iterdir():
        destination = datasets_dir / item.name
        if destination.exists():
            if item.is_dir() and destination.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
                shutil.rmtree(item)
            elif item.is_file() and destination.is_file():
                destination.unlink()
                item.rename(destination)
            else:
                raise RuntimeError(
                    f"Type mismatch while flattening dataset path: {item} -> {destination}"
                )
        else:
            item.rename(destination)

    source_dir.rmdir()


def download_pretrained_weights(
    usecase_name: str,
    weights_dir: Path,
    repo_id: str,
    allow_patterns: list[str],
    revision: str,
) -> None:
    """Download pretrained use-case weights .pt files."""

    hf.snapshot_download(
        repo_id=repo_id,
        allow_patterns=allow_patterns,
        ignore_patterns=["*/*"],
        local_dir=weights_dir,
        revision=revision,
    )

    print(f"Downloaded pretrained {usecase_name.upper()} weights to {weights_dir}")


def download_datasets(
    usecase_name: str,
    datasets_dir: Path,
    repo_id: str,
    dataset_prefix: str | None,
    revision: str,
) -> None:
    """Download and flatten use-case datasets from HF."""
    if _flattened_dataset_exists(datasets_dir):
        print(
            f"{usecase_name.upper()} datasets already exist at {datasets_dir}, skipping download"
        )
        return

    allow_patterns = (
        [f"{dataset_prefix}/**"]
        if dataset_prefix
        else [f"{split}/**" for split in DATASET_SPLITS]
    )

    hf.snapshot_download(
        repo_id=repo_id,
        allow_patterns=allow_patterns,
        repo_type="dataset",
        local_dir=datasets_dir,
        revision=revision,
    )

    if dataset_prefix:
        _flatten_dataset_subdir(datasets_dir, dataset_prefix)

    print(f"Downloaded {usecase_name.upper()} datasets to {datasets_dir}")


def validate_downloads(
    usecase_name: str,
    weights_dir: Path,
    datasets_dir: Path,
    expected_weights: list[str],
) -> None:
    """Validate downloaded weights and datasets for a usecase."""
    try:
        for file_name in expected_weights:
            weights_path = weights_dir / file_name
            assert weights_path.exists(), f"Missing weight file: {file_name}"
            size = weights_path.stat().st_size
            assert size > WEIGHTS_MIN_SIZE_BYTES, (
                f"Weight file {file_name} is less than {WEIGHTS_MIN_SIZE_BYTES / 1024:.0f} KB "
                f"({size / 1024:.1f} KB)"
            )

        for folder in DATASET_SPLITS:
            dataset_path = datasets_dir / folder
            assert (
                dataset_path.exists() and dataset_path.is_dir()
            ), f"Missing dataset directory for {usecase_name.upper()}: {folder}"
            safetensors = _list_split_safetensors(dataset_path)
            assert safetensors, f"No .safetensors files found under split {folder}"
            for safetensor in safetensors:
                size = safetensor.stat().st_size
                assert size > DATASET_MIN_SIZE_BYTES, (
                    f"Dataset file {safetensor.name} in {folder} "
                    f"is less than {DATASET_MIN_SIZE_BYTES / (1024 * 1024):.0f} MB "
                    f"({size / (1024 * 1024):.1f} MB)"
                )

    except AssertionError as e:
        raise type(e)(
            f"{e}\n\nRun 'hatch run test:download' to fetch test assets."
        ) from e


def create_mini_safetensor_dataset(
    original_dataset_path: Path,
    usecase_name: str,
    desired_frames: int,
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
                    source_file, target_file, desired_frames=desired_frames
                )
                print(f"Created {usecase_name} mini dataset at {target_file}")

    except FileNotFoundError as e:
        raise type(e)(
            f"{e}\n\nRun 'hatch run test:download' to fetch test assets."
        ) from e


if __name__ == "__main__":
    for usecase, config in USECASES.items():
        download_pretrained_weights(
            usecase_name=usecase,
            weights_dir=config["weights_dir"],
            repo_id=config["weights_repo_id"],
            allow_patterns=config["weights_allow_patterns"],
            revision=config["weights_revision"],
        )
        download_datasets(
            usecase_name=usecase,
            datasets_dir=config["datasets_dir"],
            repo_id=config["dataset_repo_id"],
            dataset_prefix=config["dataset_prefix"],
            revision=config["dataset_revision"],
        )
        validate_downloads(
            usecase_name=usecase,
            weights_dir=config["weights_dir"],
            datasets_dir=config["datasets_dir"],
            expected_weights=config["expected_weights"],
        )
        create_mini_safetensor_dataset(
            original_dataset_path=config["datasets_dir"],
            usecase_name=usecase.upper(),
            desired_frames=config["mini_dataset_frames"],
        )
