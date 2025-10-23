# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import os
import platform
import shutil
import sys


def main():
    """Entry point script to call model-converter installed via pip

    The pip package of model-converter is exposed as model_converter.
    ExecuTorch currently expects it to still be called model-converter.
    """

    if platform.system() != "Linux":
        raise OSError(f"Unsupported OS: {platform.system()!r}. Requires Linux")

    # Find the pip-installed executable on PATH
    bin_path = shutil.which("model_converter")
    if bin_path is None:
        raise FileNotFoundError(
            "Could not find 'model_converter' on PATH. "
            "Ensure it is installed in your environment "
            "(e.g. `pip install ai-ml-sdk-model-converter`)."
        )

    # Replace this process with the binary
    os.execv(bin_path, [bin_path] + sys.argv[1:])  # nosec B606
