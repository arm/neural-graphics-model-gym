# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import inspect
import json
import logging
import tempfile
import unittest
from contextlib import redirect_stdout
from importlib.resources import files
from io import StringIO
from pathlib import Path
from typing import Annotated, get_args, get_origin, Union

from pydantic import ValidationError

from ng_model_gym.core.config import config_model
from ng_model_gym.core.config.config_utils import load_config_file
from ng_model_gym.core.utils.directory_utils import create_directory
from scripts.generate_config_schema import generate_schema
from tests.testing_utils import create_simple_params

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
                dir=tmp_dir, mode="w+", suffix=".json", delete=False
            ) as temp_file:
                temp_file.write(params)
                temp_file.flush()
                temp_path = Path(temp_file.name)

            # load user config
            user_config = load_config_file(temp_path)

            # Check user config overrides default params
            self.assertEqual(user_config.train.seed, 9876)

            temp_path.unlink(missing_ok=True)

    def test_config_validation(self):
        """Test validation logic is present"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            user_config = {
                "train": {
                    "number_of_epochs": 0,
                }
            }

            with tempfile.NamedTemporaryFile(
                dir=tmp_dir, mode="w+", suffix=".json", delete=False
            ) as temp_file:
                json.dump(user_config, temp_file)
                temp_file.flush()
                temp_path = Path(temp_file.name)

            # load user config
            with redirect_stdout(StringIO()):
                with self.assertRaises(SystemExit) as e:
                    load_config_file(temp_path)

            self.assertEqual(e.exception.code, 1)

            temp_path.unlink(missing_ok=True)

    def test_reject_vgf_output_dir_in_tmp(self):
        """Test vgf_output_dir cannot be set to a temp directory"""
        user_config = create_simple_params().model_dump_json()
        user_config_dict = json.loads(user_config)
        user_config_dict["output"]["export"]["vgf_output_dir"] = str(
            Path(tempfile.gettempdir()) / "vgf"
        )
        user_config = json.dumps(user_config_dict)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with tempfile.NamedTemporaryFile(
                dir=tmp_dir, mode="w+", suffix=".json", delete=False
            ) as temp_file:
                temp_file.write(user_config)
                temp_file.flush()
                temp_path = Path(temp_file.name)

            with redirect_stdout(StringIO()):
                with self.assertRaises(SystemExit) as e:
                    load_config_file(temp_path)

            self.assertEqual(e.exception.code, 1)

            temp_path.unlink(missing_ok=True)

    def test_outdated_config_validation(self):
        """Test exception raised when parameter in user config is outdated"""

        user_config = create_simple_params().model_dump_json()
        user_config_dict = json.loads(user_config)
        user_config_dict |= {"extra": "field"}
        user_config = json.dumps(user_config_dict)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with tempfile.NamedTemporaryFile(
                dir=tmp_dir, mode="w+", suffix=".json", delete=False
            ) as temp_file:
                temp_file.write(user_config)
                temp_file.flush()
                temp_path = Path(temp_file.name)

            # load user config
            with redirect_stdout(StringIO()):
                with self.assertRaises(SystemExit) as e:
                    load_config_file(temp_path)

            self.assertEqual(e.exception.code, 1)

            temp_path.unlink(missing_ok=True)

    def test_invalid_json_file(self):
        """Test invalid JSON file throws exception"""
        user_config = create_simple_params()
        user_config = user_config.model_dump_json()
        with tempfile.TemporaryDirectory() as tmp_dir:
            with tempfile.NamedTemporaryFile(
                dir=tmp_dir, mode="w+", suffix=".json", delete=False
            ) as temp_file:
                temp_file.write(user_config)
                temp_file.flush()
                temp_path = Path(temp_file.name)

            # Invalidate the JSON structure
            with open(temp_path, "w+", encoding="utf-8") as f:
                json_file = f.read()
                invalid_json = json_file.replace('"', "'")
                f.seek(0)
                f.write(invalid_json)

            self.assertRaises(json.JSONDecodeError, load_config_file, temp_path)

            temp_path.unlink(missing_ok=True)

    def test_reject_incomplete_config(self):
        """Test incomplete config throws exception"""

        params = create_simple_params()
        # Delete seed param
        params = params.model_dump_json(exclude={"train": {"seed"}})

        with tempfile.TemporaryDirectory() as tmp_dir:
            with tempfile.NamedTemporaryFile(
                dir=tmp_dir, mode="w+", suffix=".json", delete=False
            ) as temp_file:
                temp_file.write(params)
                temp_file.flush()
                temp_path = Path(temp_file.name)

            # load user config
            with redirect_stdout(StringIO()):
                with self.assertRaises(SystemExit) as e:
                    load_config_file(temp_path)

            self.assertEqual(e.exception.code, 1)

            temp_path.unlink(missing_ok=True)

    def test_reject_validation_enabled_without_valid_schedule(self):
        """Test config fails when validation is enabled but never scheduled to run."""

        user_config = create_simple_params().model_dump(mode="json")
        user_config["train"]["perform_validate"] = True
        user_config["train"]["validate_frequency"] = 5
        user_config["train"]["fp32"]["number_of_epochs"] = 1

        with self.assertRaises(ValidationError) as exc:
            config_model.ConfigModel.model_validate(user_config)

        self.assertIn(
            "Validation is enabled (`perform_validate=true`) but no validation pass will run",
            str(exc.exception),
        )

    def test_accept_validation_schedule_with_int_frequency(self):
        """Test config accepts a valid validation schedule when frequency is an int."""

        user_config = create_simple_params().model_dump(mode="json")
        user_config["train"]["perform_validate"] = True
        user_config["train"]["validate_frequency"] = 2
        user_config["train"]["fp32"]["number_of_epochs"] = 5

        loaded_config = config_model.ConfigModel.model_validate(user_config)

        self.assertEqual(loaded_config.train.validate_frequency, 2)

    def test_accept_validation_schedule_with_list_frequency(self):
        """Test config accepts a valid validation schedule when frequency is a list."""

        user_config = create_simple_params().model_dump(mode="json")
        user_config["train"]["perform_validate"] = True
        user_config["train"]["validate_frequency"] = [2, 5]
        user_config["train"]["fp32"]["number_of_epochs"] = 5

        loaded_config = config_model.ConfigModel.model_validate(user_config)

        self.assertEqual(loaded_config.train.validate_frequency, [2, 5])

    def test_custom_models(self):
        """Test custom model config accepts custom fields"""

        user_config = create_simple_params().model_dump(mode="json")
        user_config["model"] = {
            "name": "my_model",
            "model_source": "custom",
            "version": "1",
            "custom_field": 0.1,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with tempfile.NamedTemporaryFile(
                dir=tmp_dir, mode="w+", suffix=".json", delete=False
            ) as temp_file:
                json.dump(user_config, temp_file)
                temp_path = Path(temp_file.name)

            loaded = load_config_file(temp_path)

            self.assertEqual(loaded.model.name, "my_model")
            self.assertEqual(loaded.model.model_source, "custom")
            self.assertEqual(loaded.model.custom_field, 0.1)

    def test_model_specific_config_classes_in_union_type(self):
        """
        Ensure PrebuiltModelSettingsBase subclasses are added to the prebuilt_models_settings
        union in config_model.py
        """

        def unwrap_annotated(type_param):
            if get_origin(type_param) is Annotated:
                return get_args(type_param)[0]
            return type_param

        def get_union_members(type_param):
            type_param = unwrap_annotated(type_param)
            if get_origin(type_param) is Union:
                return set(get_args(type_param))
            return {type_param}

        union_members = get_union_members(config_model.prebuilt_models_settings)

        subclasses = {
            obj
            for obj in config_model.__dict__.values()
            if inspect.isclass(obj)
            and issubclass(obj, config_model.PrebuiltModelSettingsBase)
            and obj is not config_model.PrebuiltModelSettingsBase
        }

        missing = subclasses - union_members
        if missing:
            missing_names = sorted(cls.__name__ for cls in missing)
            message = (
                "Missing PrebuiltModelSettingsBase subclasses in "
                " the union prebuilt_models_settings:\n"
                + "\n".join(f"- {name}" for name in missing_names)
                + "\nAdd them to config_model.prebuilt_models_settings union."
            )
            self.fail(message)


class TestConfigLogging(unittest.TestCase):
    """Test logging output for config validation errors."""

    def test_validation_errors_logged_to_file(self):
        """Test validation errors are added to the log file."""

        config_validation_logger = logging.getLogger(
            "ng_model_gym.api.config_validation"
        )
        config_validation_logger.handlers.clear()

        log_file_path = Path(__file__).parents[2] / "output" / "output.log"

        if log_file_path.exists():
            with open(log_file_path, "r", encoding="utf-8") as f:
                existing_log = f.read()

        user_config = create_simple_params()
        user_config.train.fp32.number_of_epochs = 0
        user_config.output.tensorboard_output_dir = "./tensorboard-logs"
        create_directory(user_config.output.tensorboard_output_dir)
        user_config = user_config.model_dump_json()

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".json", delete=False
        ) as temp_file:
            temp_file.write(user_config)
            temp_file.flush()
            temp_path = Path(temp_file.name)

        with self.assertRaises(SystemExit):
            load_config_file(temp_path)

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

        temp_path.unlink(missing_ok=True)


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
                files("ng_model_gym.core.config") / "schema_config.json"
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
                        " cd and regenerate using the script in /scripts/generate_config_schema.py",
                    )
