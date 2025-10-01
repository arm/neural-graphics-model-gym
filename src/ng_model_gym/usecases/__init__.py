# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import importlib
import os
from pathlib import Path


def import_usecase_files() -> None:
    """Function to import usecase models. This can be called to ensure models are imported."""
    import_path = Path(__name__)
    usecases_dir = f'{__file__.replace("__init__.py", "")}'
    for subdir, _, files in os.walk(usecases_dir):
        # Convert file path (path/to/module) to module import path (path.to.module)
        import_path = subdir.replace("/", ".").split("neural-graphics-model-gym.src.")[
            -1
        ]

        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                module_name = file[:-3]  # Remove the .py extension
                importlib.import_module(f"{import_path}.{module_name}")
                print(f"Imported module: {import_path}.{module_name}")
