# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import argparse
import subprocess  # nosec B404
import sys
from pathlib import Path

USECASES_ROOT = Path("tests/usecases")


def collect_targets(mode: str) -> list[str]:
    """Collect usecase test directories for the selected mode."""
    return sorted(
        str(path) for path in USECASES_ROOT.glob(f"*/{mode}") if path.is_dir()
    )


def main(args: argparse.Namespace) -> int:
    """Discover usecase test directories and run pytest."""
    targets = collect_targets(args.mode)
    if not targets:
        print(f"No usecase '{args.mode}' directories found under {USECASES_ROOT}")
        return 1

    command = [sys.executable, "-m", "pytest"]
    if args.fast_test:
        command.append("--fast-test")
    command.extend(targets)
    return subprocess.call(command)  # nosec B603


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run discovered usecase tests by mode."
    )
    parser.add_argument(
        "mode",
        choices=["unit", "integration"],
        help="Usecase test folder mode to run.",
    )
    parser.add_argument(
        "--fast-test",
        action="store_true",
        help="Pass --fast-test to pytest.",
    )
    parsed_args = parser.parse_args()
    sys.exit(main(parsed_args))
