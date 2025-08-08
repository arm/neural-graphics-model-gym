# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import os
import platform
import sys


def main():
    """Entry point script to call model-converter"""

    if platform.system() != "Linux":
        raise OSError(f"Unsupported OS: {platform.system()!r}. Requires Linux")

    # Find model-converter executable.
    pkg_dir = os.path.dirname(__file__)
    bin_path = os.path.join(pkg_dir, "model-converter")

    # Replace this process with the binary
    os.execv(bin_path, [bin_path] + sys.argv[1:])  # nosec B606
