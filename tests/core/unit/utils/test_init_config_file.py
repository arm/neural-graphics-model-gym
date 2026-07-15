# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ng_model_gym.core.config.config_model import CONFIG_SCHEMA_VERSION
from ng_model_gym.core.config.config_utils import (
    _discover_config_templates,
    generate_config_file,
    list_config_templates,
    TemplateInfo,
)


class TestGeneratingConfigFile(unittest.TestCase):
    """Test API function to create config file"""

    def setUp(self):
        """Setup temp dirs and config file paths"""

        # pylint: disable-next=consider-using-with
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Cleanup temp dir"""
        self.temp_dir.cleanup()

    def test_config_files_created(self):
        """Test the files created exist"""
        for template in ["NSS-v1", "NFRU-v1"]:
            with self.subTest(template=template):
                config_path, schema_path = generate_config_file(
                    template, self.output_path
                )
                self.assertTrue(config_path.exists())
                self.assertTrue(schema_path.exists())

    def test_placeholders_exist(self):
        """Test the generated config file has placeholders for the user to edit"""
        config_path, _ = generate_config_file("custom", self.output_path)

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        self.assertEqual(
            config_data["dataset"]["path"]["train"], "<PATH/TO/TRAIN_DATA_DIR>"
        )
        self.assertEqual(
            config_data["train"]["fp32"]["checkpoints"]["dir"],
            "<OUTPUT/PATH/FOR/CHECKPOINTS_DIR>",
        )

    def test_incrementing_config_name_if_already_exists(self):
        """Test if existing configs exists, it increments the file name"""
        config_path, _ = generate_config_file("nss-v1", self.output_path)
        config_path1, _ = generate_config_file("nss-v1", self.output_path)
        config_path2, _ = generate_config_file("nss-v1", self.output_path)

        config_template_name = config_path.stem
        self.assertEqual(config_path1.stem, f"{config_template_name}_1")
        self.assertEqual(config_path2.stem, f"{config_template_name}_2")

    def test_case_insensitive_generate_config(self):
        """Test case/whitespace insensitive generate_config template name"""
        config_path, _ = generate_config_file("  nSs-v1  ", self.output_path)

        self.assertEqual(config_path.name, "nss-v1_config.json")

    def test_generate_nss_config_file_uses_v1_template(self):
        """Test generating the default NSS config uses the v1 template."""
        config_path, _ = generate_config_file("NSS-v1", self.output_path)

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        self.assertEqual(config_path.name, "nss-v1_config.json")
        self.assertEqual(config_data["model"]["name"], "NSS-v1")
        self.assertNotIn("version", config_data["model"])
        self.assertEqual(config_data["model"]["quality"], "high")
        self.assertTrue(config_data["model"]["nss_v1_luma_derivative"])
        self.assertTrue(config_data["model"]["nss_v1_sharp_theta"])
        self.assertTrue(config_data["model"]["gt_history_augmentation"])
        self.assertEqual(config_data["model"]["gt_history_augmentation_chance"], 30.0)
        self.assertEqual(config_data["train"]["loss_fn"], "loss_v1")
        self.assertEqual(
            config_data["train"]["loss_args"],
            {
                "temporal_reg_weight": 0.7,
                "alpha_reg_weight": 0.0001,
                "temporal_reg_channels": 1,
                "min_weight": 0.1,
            },
        )

    def test_generate_nss_config_file_is_v1_by_default(self):
        """Test generating NSS keeps the v1 template as the default."""
        config_path, _ = generate_config_file("NSS-v1", self.output_path)

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        self.assertEqual(config_path.name, "nss-v1_config.json")
        self.assertEqual(config_data["model"]["name"], "NSS-v1")
        self.assertNotIn("version", config_data["model"])

    def test_nss_in_template_list(self):
        """Test list config templates includes nss"""
        templates = list_config_templates()

        self.assertIn("NSS-v1", templates)

    def test_nfru_in_template_list(self):
        """Test list config templates includes nfru"""
        templates = list_config_templates()

        self.assertIn("NFRU-v1", templates)

    def test_custom_in_template_list(self):
        """Test list config templates includes custom"""
        templates = list_config_templates()

        self.assertIn("custom", templates)

    def test_generate_custom_template_name(self):
        """Test custom template output uses custom_config.json"""
        config_path, _ = generate_config_file("custom", self.output_path)

        self.assertEqual(config_path.name, "custom_config.json")

    def test_generate_config_file_empty_template_raises(self):
        """Test passing an empty template raises"""
        with self.assertRaises(ValueError):
            generate_config_file("   ", self.output_path)

    def test_generate_config_file_unknown_template_raises(self):
        """Test passing an unknown template raises"""
        with self.assertRaises(FileNotFoundError):
            generate_config_file("random input", self.output_path)

    @patch("ng_model_gym.core.config.config_utils._discover_config_templates")
    def test_colliding_template_names(self, mock_discover):
        """Test multiple templates with the same name raises"""
        mock_discover.return_value = {
            "nss-v1": [
                TemplateInfo(model_name="nss-v1", json_data={}, source=Path("a.json")),
                TemplateInfo(model_name="nss-v1", json_data={}, source=Path("b.json")),
            ]
        }

        with self.assertRaises(ValueError) as raised:
            generate_config_file("nss-v1", self.output_path)

        self.assertIn("Multiple config templates found", str(raised.exception))

    def test_invalid_save_dir_raises(self):
        """Test the function raises if invalid dir is passed"""
        with self.assertRaises(FileNotFoundError):
            generate_config_file("nss-v1", "/invalid_dir")

    def test_all_templates_match_config_schema_version(self):
        """Test all discovered config templates have the current schema version."""
        templates = _discover_config_templates()

        mismatches = []
        for infos in templates.values():
            for info in infos:
                template_version = info.json_data.get("config_schema_version")
                if template_version != CONFIG_SCHEMA_VERSION:
                    mismatches.append(
                        (
                            str(info.source),
                            template_version,
                            CONFIG_SCHEMA_VERSION,
                        )
                    )

        self.assertEqual(
            len(mismatches),
            0,
            "Template config_schema_version mismatch(es): "
            + "; ".join(
                f"{path}: found {found!r}, expected {expected!r}"
                for path, found, expected in mismatches
            ),
        )
