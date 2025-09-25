# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import tempfile
import unittest
from contextlib import redirect_stdout
from importlib.resources import files
from io import StringIO
from pathlib import Path

from ng_model_gym.core.utils.config_utils import load_config_file
from ng_model_gym.core.utils.general_utils import create_directory
from scripts.generate_config_schema import generate_schema
from tests.unit.utils.utils import create_simple_params

# Create a logger using the root module name 'ng_model_gym',
# to be able to get the log used throughout the project.
ROOT_MODULE_NAME = "ng_model_gym"
logger = logging.getLogger(ROOT_MODULE_NAME)


class TestConfig(unittest.TestCase):
    """Test loading configuration from disk"""

    def setUp(self):
        """Setup test case"""
        logging.disable()

    def tearDown(self):
        """Re-enable logging"""
        logging.disable(logging.NOTSET)

    def test_invalid_user_config_path(self):
        """Test invalid path passed to load_config_file() throws"""
        self.assertRaises(
            FileNotFoundError, load_config_file, Path("./invalid/path.json")
        )

    def test_loading_user_config_file(self):
        """Test loading config"""

        params = create_simple_params()
        params.train.seed = 9876
        params = params.model_dump_json()

        with tempfile.TemporaryDirectory() as tmp_dir:
            with tempfile.NamedTemporaryFile(
                dir=tmp_dir, mode="w+", suffix=".json"
            ) as temp_file:
                temp_file.write(params)
                temp_file.flush()

                # load user config
                user_config = load_config_file(Path(temp_file.name))

                # Check user config overrides default params
                self.assertEqual(user_config.train.seed, 9876)

    def test_config_validation(self):
        """Test validation logic is present"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            user_config = {
                "train": {
                    "number_of_epochs": 0,
                }
            }

            with tempfile.NamedTemporaryFile(
                dir=tmp_dir, mode="w+", suffix=".json"
            ) as temp_file:
                json.dump(user_config, temp_file)
                temp_file.flush()

                # load user config
                with redirect_stdout(StringIO()):
                    with self.assertRaises(SystemExit) as e:
                        load_config_file(Path(temp_file.name))

                self.assertEqual(e.exception.code, 1)

    def test_outdated_config_validation(self):
        """Test exception raised when parameter in user config is outdated"""

        user_config = create_simple_params().model_dump_json()
        user_config_dict = json.loads(user_config)
        user_config_dict |= {"extra": "field"}
        user_config = json.dumps(user_config_dict)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with tempfile.NamedTemporaryFile(
                dir=tmp_dir, mode="w+", suffix=".json"
            ) as temp_file:
                temp_file.write(user_config)
                temp_file.flush()

                # load user config
                with redirect_stdout(StringIO()):
                    with self.assertRaises(SystemExit) as e:
                        load_config_file(Path(temp_file.name))

                self.assertEqual(e.exception.code, 1)

    def test_invalid_json_file(self):
        """Test invalid JSON file throws exception"""
        user_config = create_simple_params()
        user_config = user_config.model_dump_json()
        with tempfile.TemporaryDirectory() as tmp_dir:
            with tempfile.NamedTemporaryFile(
                dir=tmp_dir, mode="w+", suffix=".json"
            ) as temp_file:
                temp_file.write(user_config)
                temp_file.flush()

                # Invalidate the JSON structure
                with open(Path(temp_file.name), "w+", encoding="utf-8") as f:
                    json_file = f.read()
                    invalid_json = json_file.replace('"', "'")
                    f.seek(0)
                    f.write(invalid_json)

                self.assertRaises(
                    json.JSONDecodeError, load_config_file, Path(temp_file.name)
                )

    def test_reject_incomplete_config(self):
        """Test incomplete config throws exception"""

        params = create_simple_params()
        # Delete seed param
        params = params.model_dump_json(exclude={"train": {"seed"}})

        with tempfile.TemporaryDirectory() as tmp_dir:
            with tempfile.NamedTemporaryFile(
                dir=tmp_dir, mode="w+", suffix=".json"
            ) as temp_file:
                temp_file.write(params)
                temp_file.flush()

                # load user config
                with redirect_stdout(StringIO()):
                    with self.assertRaises(SystemExit) as e:
                        load_config_file(Path(temp_file.name))

                self.assertEqual(e.exception.code, 1)


class TestConfigLogging(unittest.TestCase):
    """Test logging output for config validation errors."""

    def test_validation_errors_logged_to_file(self):
        """Test validation errors are added to the log file."""

        config_validation_logger = logging.getLogger(
            "ng_model_gym.api.config_validation"
        )
        config_validation_logger.handlers.clear()

        log_file_path = Path(__file__).parents[3] / "output" / "output.log"

        if log_file_path.exists():
            with open(log_file_path, "r", encoding="utf-8") as f:
                existing_log = f.read()

        user_config = create_simple_params()
        user_config.train.fp32.number_of_epochs = 0
        user_config.output.tensorboard_output_dir = "./tensorboard-logs"
        create_directory(user_config.output.tensorboard_output_dir)
        user_config = user_config.model_dump_json()

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as temp_file:
            temp_file.write(user_config)
            temp_file.flush()

            with self.assertRaises(SystemExit):
                load_config_file(Path(temp_file.name))

        for handler in config_validation_logger.handlers:
            handler.flush()

        with open(log_file_path, "r", encoding="utf-8") as f:
            updated_log = f.read()

        added_content = updated_log[len(existing_log) :]

        # Check added content contains expected validation errors
        self.assertIn("Config validation errors:", added_content)
        self.assertIn("greater_than_equal", added_content)
        self.assertIn("train.fp32.number_of_epochs", added_content)
        self.assertIn("input=0", added_content)
        self.assertIn("Configuration has 1 issue", added_content)


class TestConfigSchemaGenerator(unittest.TestCase):
    """Test the schema config generator"""

    def test_schema_json_outdated(self):
        """Test if the schema_config.json in repo is outdated"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Generate the most up-to-date schema_config.json by using the current Pydantic model
            regenerated_schema_path = Path(tmp_dir, "regenerated_schema_config.json")
            generate_schema(regenerated_schema_path)

            # Path to schema_config.json in the repo. Possibly outdated
            current_schema_path = (
                files("ng_model_gym.usecases.nss.configs") / "schema_config.json"
            )
            # Load files to compare contents
            with open(
                current_schema_path, "r", encoding="utf-8"
            ) as current_schema_file:
                with open(
                    regenerated_schema_path, "r", encoding="utf-8"
                ) as regenerated_schema_file:
                    current_schema_json = json.load(current_schema_file)
                    regenerated_schema_json = json.load(regenerated_schema_file)

                    # Assert schema file in repo should be the same as the newly generated schema
                    self.assertEqual(
                        current_schema_json,
                        regenerated_schema_json,
                        "The schema_config.json file is outdated,"
                        " regenerate using the script in /scripts/generate_config_schema.py",
                    )


if __name__ == "__main__":
    unittest.main()
