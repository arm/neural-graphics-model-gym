# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ng_model_gym.core.config.config_utils import (
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
        config_path, schema_path = generate_config_file("nss", self.output_path)

        self.assertTrue(config_path.exists())
        self.assertTrue(schema_path.exists())

    def test_placeholders_exist(self):
        """Test the generated config file has placeholders for the user to edit"""
        config_path, _ = generate_config_file("nss", self.output_path)

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
        config_path, _ = generate_config_file("nss", self.output_path)
        config_path1, _ = generate_config_file("nss", self.output_path)
        config_path2, _ = generate_config_file("nss", self.output_path)

        config_template_name = config_path.stem
        self.assertEqual(config_path1.stem, f"{config_template_name}_1")
        self.assertEqual(config_path2.stem, f"{config_template_name}_2")

    def test_case_insensitive_generate_config(self):
        """Test case/whitespace insensitive generate_config template name"""
        config_path, _ = generate_config_file("  nSs  ", self.output_path)

        self.assertEqual(config_path.name, "nss.json")

    def test_nss_in_template_list(self):
        """Test list config templates includes nss"""
        templates = list_config_templates()

        self.assertIn("nss", templates)

    def test_custom_in_template_list(self):
        """Test list config templates includes custom"""
        templates = list_config_templates()

        self.assertIn("custom", templates)

    def test_generate_custom_template_name(self):
        """Test custom template output uses custom_template.json"""
        config_path, _ = generate_config_file("custom", self.output_path)

        self.assertEqual(config_path.name, "custom_template.json")

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
            "nss": [
                TemplateInfo(model_name="nss", json_data={}, source=Path("a.json")),
                TemplateInfo(model_name="nss", json_data={}, source=Path("b.json")),
            ]
        }

        with self.assertRaises(ValueError) as raised:
            generate_config_file("nss", self.output_path)

        self.assertIn("Multiple config templates found", str(raised.exception))

    def test_invalid_save_dir_raises(self):
        """Test the function raises if invalid dir is passed"""
        with self.assertRaises(FileNotFoundError):
            generate_config_file("nss", "/invalid_dir")
