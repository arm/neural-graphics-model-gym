# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path


def rm(path: Path):
    """Delete files or directories if they exist."""
    try:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    except FileNotFoundError:
        pass


def main():
    """Entry point for cleaning script"""
    print("Removing temporary directories")

    root = Path(".")

    # recursively find and delete matching files and directories
    for pattern in [
        "__pycache__",
        "*.pyc",
        ".coverage*",
        "coverage.json",
        "coverage_html",
        "report.html",
        "output",
        "checkpoints",
        "tensorboard-logs",
    ]:
        for p in root.rglob(pattern):
            rm(p)

    shader_root = (
        root / "src" / "ng_model_gym" / "usecases" / "nss" / "model" / "shaders"
    )

    for pattern in [".slangtorch_cache", "*.lock"]:
        for p in shader_root.rglob(pattern):
            rm(p)


if __name__ == "__main__":
    main()
