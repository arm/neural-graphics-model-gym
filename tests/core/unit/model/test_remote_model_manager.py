# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import random
import tempfile
import time
import unittest
from pathlib import Path

from huggingface_hub.errors import RepositoryNotFoundError

from ng_model_gym.core.repos.remote_model_manager import (
    download_pretrained_model,
    list_pretrained_models,
    parse_model_identifier,
)


class TestParseIdentifier(unittest.TestCase):
    """Test model identifier parsing"""

    def test_parse_identifier(self):
        """test parse identifier function"""
        cases = [
            ("@sample-repo/model.pt", ("sample-repo", "model.pt")),
            ("sample-repo/model.pt", ("sample-repo", "model.pt")),
            (" @sample-repo/model.pt ", ("sample-repo", "model.pt")),
            (" sample-repo/model.pt ", ("sample-repo", "model.pt")),
            (" @sample-repo/dir/model.pt ", ("sample-repo", "dir/model.pt")),
        ]

        for identifier, expected in cases:
            with self.subTest(identifier=identifier):
                repo_name, file_name = parse_model_identifier(identifier)
                self.assertEqual((repo_name, file_name), expected)

    def test_invalid_inputs(self):
        """Test raise on invalid identifiers"""
        invalid_identifiers = [
            # "!sample-repo/model.pt"
            "sample-repo",
            "@sample-repo",
            " sample-repo ",
            " @sample-repo ",
            "/model.pt",
            "sample-repo/",
            "@/model.pt",
            "@sample-repo/",
            "/",
            " @/ ",
        ]
        for identifier in invalid_identifiers:
            with self.subTest(identifier=identifier):
                with self.assertRaises(ValueError):
                    parse_model_identifier(identifier)


class TestHFModelServer(unittest.TestCase):
    """Test using the HF API listing/downloading models"""

    def setUp(self):
        """Setup"""
        # Slow down tests so we don't hit rate limits
        time.sleep(random.uniform(0, 2))

    def test_hf_model_list(self):
        """List models on HF for neural-super-sampling"""
        repo_dict = list_pretrained_models()
        self.assertIn("HuggingFace", repo_dict)
        self.assertTrue(
            any(
                hf_repos.repository.name == "neural-super-sampling"
                and len(hf_repos.models) > 1
                for hf_repos in repo_dict["HuggingFace"]
            )
        )

    def test_hf_model_download(self):
        """Test downloading model from HF to tmp dir"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            download_path = download_pretrained_model(
                "neural-super-sampling/nss_v0.1.0_fp32.pt", Path(tmp_dir)
            )
            self.assertEqual(download_path.name, "nss_v0.1.0_fp32.pt")

            # Check download path is in the temp dir specified
            self.assertTrue(
                download_path.resolve().is_relative_to(Path(tmp_dir).resolve())
            )
            self.assertTrue(download_path.exists())
            # Check not empty
            self.assertGreater(download_path.stat().st_size, 0)

    def test_download_with_config_identifier(self):
        """Test download using @ in the identifier (used in config)"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            download_path = download_pretrained_model(
                "@neural-super-sampling/nss_v0.1.0_fp32.pt", Path(tmp_dir)
            )
            self.assertEqual(download_path.name, "nss_v0.1.0_fp32.pt")

            # Check download path is in the temp dir specified
            self.assertTrue(
                download_path.resolve().is_relative_to(Path(tmp_dir).resolve())
            )
            self.assertTrue(download_path.exists())
            # Check not empty
            self.assertGreater(download_path.stat().st_size, 0)

    def test_hf_model_download_raise(self):
        """Test raise when giving random repo"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(RepositoryNotFoundError):
                _ = download_pretrained_model(
                    "random-repo/nss_v0.1.0_fp32.pt", Path(tmp_dir)
                )

    def test_download_destination_is_a_dir(self):
        """Check download destination is a dir"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            destination_file = Path(tmp_dir) / "some_file"
            destination_file.write_text("file contents")

            with self.assertRaises(ValueError):
                download_pretrained_model(
                    "neural-super-sampling/nss_v0.1.0_fp32.pt",
                    destination_file,
                )

    def test_download_raises_bad_pt_name(self):
        """Test raise when given bad pt file name"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError) as ctx:
                download_pretrained_model(
                    "neural-super-sampling/fake.pt", Path(tmp_dir)
                )

            self.assertIn("not found in repository", str(ctx.exception))

    def test_bad_identifier_raises(self):
        """Test bad identifier"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                download_pretrained_model("neural-super-sampling", Path(tmp_dir))
