# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path

from ng_model_gym.api import DEFAULT_PATH, generate_config_file


class TestGeneratingConfigFile(unittest.TestCase):
    """Test API function to create config file"""

    def setUp(self):
        """Setup temp dirs and config file paths"""

        # pylint: disable-next=consider-using-with
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = Path(self.temp_dir.name)
        self.default_config_path = files(DEFAULT_PATH) / "default.json"
        self.schema_config_path = files(DEFAULT_PATH) / "schema_config.json"

        with open(self.default_config_path, "r", encoding="utf-8") as f:
            self.default_config = json.load(f)

    def tearDown(self):
        """Cleanup temp dir"""
        self.temp_dir.cleanup()

    def test_config_files_created(self):
        """Test the files created exist"""
        config_path, schema_path = generate_config_file(self.output_path)

        self.assertTrue(config_path.exists())
        self.assertTrue(schema_path.exists())

    def test_placeholders_exist(self):
        """Test the generated config file has placeholders for the user to edit"""
        config_path, _ = generate_config_file(self.output_path)

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
        """Test if gen config command is invoked, it creates a new file name"""
        _, _ = generate_config_file(self.output_path)
        config_path1, _ = generate_config_file(self.output_path)
        config_path2, _ = generate_config_file(self.output_path)

        self.assertTrue("config_1.json" in str(config_path1.name))
        self.assertTrue("config_2.json" in str(config_path2.name))

    def test_invalid_save_dir_raises(self):
        """Test the function raises if invalid dir is passed"""
        with self.assertRaises(FileNotFoundError):
            generate_config_file("/invalid_dir")


if __name__ == "__main__":
    unittest.main()
