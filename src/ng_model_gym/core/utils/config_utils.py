# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
import shutil
import sys
from importlib.resources import files
from pathlib import Path
from typing import List, Tuple, Union

from pydantic import ValidationError
from rich import print_json
from rich.console import Console
from rich.table import Column, Table

from ng_model_gym.core.utils.config_model import (
    CONFIG_SCHEMA_VERSION,
    ConfigModel,
    OutputDirModel,
)
from ng_model_gym.core.utils.json_reader import read_json_file
from ng_model_gym.core.utils.logging import setup_logging

DEFAULT_PATH = "ng_model_gym.usecases.nss.configs"


def generate_config_file(save_dir: Union[str, Path, None] = None) -> Tuple[Path, Path]:
    """
    Generate a JSON configuration template and its schema file. This is used to configure training.

    Args:
        save_dir (Union[str, Path, None]): Directory to save config and schema. If None,
         uses current directory.

    Returns:
        Tuple[Path, Path]: Paths to the generated configuration JSON and schema JSON files.

    Example:
        >>> config_output_path, schema_path = generate_config_file(Path('./output'))
    """

    if save_dir:
        output_dir = Path(save_dir)
        if not output_dir.exists() or not output_dir.is_dir():
            raise FileNotFoundError(f"Provided save_dir '{output_dir}' does not exist.")
    else:
        output_dir = Path(".")

    placeholders = {
        "dataset.path.train": "<PATH/TO/TRAIN_DATA_DIR>",
        "dataset.path.validation": "<PATH/TO/VALIDATION_DATA_DIR>",
        "dataset.path.test": "<PATH/TO/TEST_DATA_DIR>",
        "train.fp32.checkpoints.dir": "<OUTPUT/PATH/FOR/CHECKPOINTS_DIR>",
        "train.qat.checkpoints.dir": "<OUTPUT/PATH/FOR/CHECKPOINTS_DIR>",
    }

    # Load internal default config
    default_config_path = files(DEFAULT_PATH) / "default.json"
    default_config: dict = read_json_file(default_config_path)

    for json_loc, placeholder_text in placeholders.items():
        dict_key_path: List[str] = json_loc.split(".")
        config_section = default_config

        # Iterate over dict keys e.g dict_key_path = ["dataset", "path", "train"]
        for dict_key in dict_key_path[:-1]:
            # Check the json path provided in placeholders is valid
            if dict_key not in config_section or not isinstance(
                config_section[dict_key], dict
            ):
                raise KeyError(
                    f"Invalid config path {json_loc} to replace with a placeholder"
                )
            # Access sub dict e.g config_section["dataset"]
            config_section = config_section[dict_key]

        final_key = dict_key_path[-1]
        if final_key not in config_section:
            raise KeyError(
                f"Invalid config path {json_loc} to replace with a placeholder"
            )

        config_section[final_key] = placeholder_text

    file_name = "config"
    suffix = ".json"
    config_output_path = output_dir / f"{file_name}{suffix}"

    # If a file already exists "config.json", create config_1.json etc
    if config_output_path.exists():
        count = 1
        file_path = output_dir / f"{file_name}_{count}{suffix}"
        while file_path.exists():
            count += 1
            file_path = output_dir / f"{file_name}_{count}{suffix}"
        config_output_path = file_path

    # Write the config file
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(default_config, f, indent=4)

    # Copy schema_config.json as well
    schema_config_path = files(DEFAULT_PATH) / "schema_config.json"
    shutil.copy(src=schema_config_path, dst=output_dir)

    schema_path = output_dir / "schema_config.json"
    return config_output_path, schema_path


def load_config_file(user_config_path: Path) -> ConfigModel:
    """
    Load and validate a JSON configuration file. The created config object is used as an argument
    to other API functions.

    Note: SystemExit - if validation fails, after printing errors, exits with status code 1

    Args:
        user_config_path (Path): Path to a user-provided JSON config file

    Returns:
        ConfigModel: Parsed and validated configuration model.

    Example:
        >>> params = load_config_file(Path("config.json"))
    """

    user_config: dict = read_json_file(user_config_path)

    validate_schema_version(user_config)

    try:
        return ConfigModel.model_validate(user_config)
    except ValidationError as e:
        validation_errors = e.errors()

        # Set up logging for validating config file, logging to file only
        # Try to get the user's specified output_dir, fall back to ./output if not valid
        output_dir = user_config.get("output", {}).get("dir")

        try:
            output_dir = OutputDirModel(dir=output_dir).dir

        except ValidationError:
            output_dir = Path("./output")

        setup_logging(
            logger_name=f"{__name__}.config_validation",
            log_level=logging.ERROR,
            output_dir=output_dir,
            stdout=False,
        )

        config_logger = logging.getLogger(f"{__name__}.config_validation")
        config_logger.propagate = False

        # Format validation errors into a table
        table = Table(
            Column(
                header="Validation Issue",
                justify="center",
                style="red",
                no_wrap=False,
                overflow="fold",
            ),
            Column(header="Details", justify="center", no_wrap=False, overflow="fold"),
            Column(
                header="Your input",
                justify="center",
                style="blue",
                no_wrap=False,
                overflow="fold",
            ),
            Column(
                header="Location in JSON",
                justify="center",
                style="green",
                no_wrap=False,
                overflow="fold",
            ),
            show_lines=True,
            expand=True,
        )

        config_logger.error("Config validation errors:")

        # Create table rows sorted by type e.g. missing, int_parsing
        for error_ctx in sorted(validation_errors, key=lambda e: e["type"]):
            val_type = error_ctx["type"]
            message = error_ctx["msg"]

            user_input = "" if val_type == "missing" else str(error_ctx["input"])

            # Location is a tuple, turn into list of strings and use dot instead of space
            location_in_json = ".".join(map(str, [*error_ctx["loc"]]))

            table.add_row(val_type, message, user_input, location_in_json)

            # Log errors
            config_logger.error(
                f"[{val_type}] {message} (input={user_input}, location={location_in_json})"
            )

        # Instantiate Rich console and print table
        console = Console()
        console.print(table)

        num_errors = len(validation_errors)

        error_str = (
            f"\n[bold]Configuration has [red]{num_errors}[/red] "
            f"{'issues' if num_errors > 1 else 'issue'}[/bold]"
        )

        console.print(error_str)

        error_str = re.sub(r"\[/?[^\]]+\]", "", error_str).lstrip("\n")
        config_logger.error(error_str)

        sys.exit(1)


def validate_schema_version(user_config: dict) -> None:
    """Check the user provided config file has a valid schema version"""

    schema_version = user_config.get("config_schema_version")

    if schema_version != CONFIG_SCHEMA_VERSION:
        console = Console()
        provided_version = (
            f"'{schema_version}'" if schema_version is not None else "missing"
        )
        # pylint: disable=line-too-long

        console.print(
            f"\n[bold magenta]Configuration file version mismatch[/bold magenta]\n"
            f"[bold red]→[/bold red] Expected config_schema_version: [bold bright_green]{CONFIG_SCHEMA_VERSION}[/bold bright_green]\n"
            f"[bold red]→[/bold red] Provided config_schema_version: [bold bright_cyan]{provided_version}[/bold bright_cyan]\n"
            f"\nCreate an updated config with [bold]ng-model-gym init[/bold] or update it to the latest template.\n"
        )
        sys.exit(1)


def print_config_options() -> None:
    """
    Print the JSON configuration schema, listing each parameter with its type and description

    Example:
        >>> print_config_options()
    """
    schema_config_path = files(DEFAULT_PATH) / "schema_config.json"
    with schema_config_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    print_json(json.dumps(data))
