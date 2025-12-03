# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import subprocess  # nosec B404
import sys
from pathlib import Path

# Directories to check files in
ROOT_DIRS = ["docs", "src", "tests", "scripts"]

# Files to skip
SKIPS = [
    "src/ng_model_gym/usecases/nss/model/shaders/depth_clip.slang",
    "scripts/safetensors_generator/fsr2_methods.py",
]


def collect_files():
    """Get list of files to run blocklint on."""
    files = []

    # Recursively collect files from directories to check
    for d in ROOT_DIRS:
        path = Path(d)

        if not path.exists():
            continue

        for p in path.rglob("*"):
            if p.is_file():
                files.append(str(p))

    return files


def run_blocklint(files):
    """Run blocklint command."""
    print("Running blocklint to ensure no non-inclusive language")

    base_cmd = ["blocklint", f"--skip-files={','.join(SKIPS)}"]

    # Run as batches of 400 files
    for i in range(0, len(files), 400):
        rc = subprocess.call(base_cmd + files[i : i + 400])  # nosec B603
        if rc:
            return rc

    return 0


def main():
    """Entry point for blocklint script"""
    blocklint_files = collect_files()

    if not blocklint_files:
        sys.exit(0)

    sys.exit(run_blocklint(blocklint_files))


if __name__ == "__main__":
    main()
