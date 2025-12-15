# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import argparse
import subprocess  # nosec B404
import sys
from pathlib import Path

# Directories to check files in
ROOT_DIRS = ["docs", "src", "scripts"]
TEST_DIRS = ["tests"]

# Message format for pylint output
MSG_TEMPLATE = "pylint ERROR {path}:{line}:{column}: {msg_id}: {msg} ({symbol})"


def collect_files(lint_src, lint_test):
    """Get list of files to run pylint on."""
    dirs_to_check = []
    if lint_src:
        dirs_to_check += ROOT_DIRS
    if lint_test:
        dirs_to_check += TEST_DIRS

    files = []

    # Recursively collect .py files from directories to check
    for d in dirs_to_check:
        path = Path(d)

        if not path.exists():
            continue

        for p in path.rglob("*.py"):
            if p.is_file():
                files.append(str(p))

    if lint_src:
        # Append top level .py files
        files.extend(str(f) for f in Path(".").glob("*.py") if f.is_file())

    return files


def run_pylint(files, disable_flags):
    """Run pylint command."""
    base_cmd = ["pylint", "--msg-template", MSG_TEMPLATE]

    if disable_flags:
        base_cmd += ["--disable", disable_flags]

    for i in range(0, len(files), 200):
        rc = subprocess.call(base_cmd + files[i : i + 200])  # nosec B603
        if rc:
            return rc

    return 0


def main(args):
    """Entry point for pylint script."""
    lint_src = args.src
    lint_test = args.test
    disable_flags = args.disable

    # Default to both if neither flag is provided
    if not lint_src and not lint_test:
        lint_src = True
        lint_test = True

    if lint_src:
        print("Running linting of source, scripts, docs, and top-level files")
    if lint_test:
        print("Running linting of the test files")

    files = collect_files(lint_src, lint_test)

    if not files:
        return 0

    return run_pylint(files, disable_flags)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        action="store_true",
        help="Run pylint on src files.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run pylint on test files",
    )
    parser.add_argument(
        "--disable",
        default="",
        help="Disable flags to use e.g. invalid-name",
    )

    parsed_args = parser.parse_args()

    sys.exit(main(parsed_args))
