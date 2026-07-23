# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.fetch_huggingface import download_nfru_datasets, validate_nfru_downloads


@unittest.skip("NFRU CI/assets disabled for now")
class TestFetchHuggingFace(unittest.TestCase):
    """Test Hugging Face test asset download helpers."""

    def test_download_nfru_datasets_downloads_only_nfru_and_flattens_split_dirs(self):
        """Download NFRU safetensors into split directories without nfru prefix."""

        def fake_snapshot_download(*, local_dir, **kwargs):
            self.assertEqual(
                kwargs["allow_patterns"],
                [
                    "nfru/train/*.safetensors",
                    "nfru/test/*.safetensors",
                    "nfru/val/*.safetensors",
                ],
            )
            for split in ("train", "test", "val"):
                split_dir = Path(local_dir) / "nfru" / split
                split_dir.mkdir(parents=True)
                (split_dir / f"{split}.safetensors").write_bytes(b"test")

        with tempfile.TemporaryDirectory() as tmp_dir:
            datasets_dir = Path(tmp_dir) / "datasets"

            with patch(
                "tests.fetch_huggingface.hf.snapshot_download",
                side_effect=fake_snapshot_download,
            ) as snapshot_download:
                download_nfru_datasets(datasets_dir)

            snapshot_download.assert_called_once()
            for split in ("train", "test", "val"):
                split_file = datasets_dir / split / f"{split}.safetensors"
                self.assertTrue(split_file.is_file())

            self.assertFalse((datasets_dir / "nfru").exists())

    def test_validate_nfru_datasets_uses_flattened_split_dirs(self):
        """Validate NFRU datasets from tests/usecases/nfru/datasets/{split}."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            datasets_dir = Path(tmp_dir) / "datasets"
            for split in ("train", "test", "val"):
                split_dir = datasets_dir / split
                split_dir.mkdir(parents=True)
                with open(split_dir / f"{split}.safetensors", "wb") as sf_file:
                    sf_file.truncate(26 * 1024 * 1024)

            validate_nfru_downloads(datasets_dir)
