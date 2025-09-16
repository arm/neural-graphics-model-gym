# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import sys
from pathlib import Path

import jsonref

from ng_model_gym.core.utils.config_model import ConfigModel

sys.path.append("..")

logger = logging.getLogger(__name__)


def format_schema(schema):
    """Recurse through dictionary to remove and adjust keys"""
    if not isinstance(schema, dict):
        return schema
    cleaned_schema = {}

    # Remove keys from schema. If key type is an object, remove description and type
    skip_schema_keys = [
        "title",
        "required",
        "additionalProperties",
        "model_train_eval_mode",
    ] + [key for key in ("description", "type") if schema.get("type") == "object"]

    for key, value in schema.items():
        if key in skip_schema_keys:
            continue

        if isinstance(value, dict):
            if key == "properties":
                cleaned_schema.update(format_schema(value))
            else:
                cleaned_schema[key] = format_schema(value)
            continue
        if isinstance(value, list):
            list_items = []
            for item in value:
                if isinstance(item, dict):
                    list_items.append(format_schema(item))

                else:
                    list_items.append(item)
            value = list_items

        cleaned_schema[key] = value
    return cleaned_schema


def generate_schema(output_path: Path):
    """Generate schema of the config model"""
    with open(output_path, "w", encoding="utf-8") as schema_file:
        # Inline the "refs" keys in the generated JSON schema spec
        schema = jsonref.replace_refs(
            ConfigModel.model_json_schema(), merge_props=True
        )["properties"]

        json.dump(format_schema(schema), schema_file, indent=4)

    logger.info("Schema generation complete")


if __name__ == "__main__":
    generate_schema(
        Path(
            "..",
            "src",
            "ng_model_gym",
            "usecases",
            "nss",
            "configs",
            "schema_config.json",
        )
    )
