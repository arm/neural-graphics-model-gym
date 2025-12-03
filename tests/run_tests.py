# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import argparse
import glob
import os
import platform
import subprocess
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from tests.fetch_huggingface import validate_nss_downloads

from .pkgutil_patch import apply_patch

if platform.system() == "Windows":
    # For Windows, avoid rebuilding .dlls that might already be loaded in another process
    os.environ.setdefault("SKIP_NINJA_CHECK", "1")


def run_tests(start_test_dir, run_coverage=False, fast_test=False):
    """Run tests recursively from start_test_dir"""
    if run_coverage:
        return not execute_code_coverage(start_test_dir)
    loader = unittest.TestLoader()

    # If fast_test is enabled, expose to tests as an env variable
    with patch.dict(
        os.environ, {"FAST_TEST": "1"}, clear=False
    ) if fast_test else nullcontext():
        tests = loader.discover(start_dir=start_test_dir, pattern="test_*.py")
        result = unittest.TextTestRunner(verbosity=2).run(tests)
    return result.wasSuccessful()


def execute_code_coverage(start_test_dir):
    """Get code coverage for each test file. Returns list of failed tests."""
    failed_tests = []
    failed_outputs = {}

    pattern = os.path.join(start_test_dir, "**", "test_*.py")
    test_files = sorted(glob.glob(pattern, recursive=True))

    for test_file in test_files:
        print(f"Running Test: {test_file}")

        cmd = ["coverage", "run", "--parallel-mode", test_file]

        root = str(Path.cwd())
        src = str(Path.cwd() / "src")

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [root, src, env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)

        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )

        if result.returncode != 0:
            failed_tests.append(test_file)
            failed_outputs[test_file] = result.stdout

    if failed_tests:
        print("\n=== Failed Test Outputs ===")
        for test_file in failed_tests:
            print(f"\n--- {test_file} ---")
            print(failed_outputs[test_file])
    return failed_tests


if __name__ == "__main__":
    # Apply patch for Python 3.10 before running tests
    if sys.version_info[:2] == (3, 10):
        apply_patch()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-dirs",
        nargs="+",
        required=True,
        help="Test directories to look for tests in.",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Enable coverage in subprocesses",
    )
    parser.add_argument(
        "--fast-test",
        action="store_true",
        help="Use smaller dataset and slang patch",
    )
    args = parser.parse_args()

    # Check required pretrained weights/datasets have been downloaded
    dataset_path = Path("tests/usecases/nss/datasets")
    validate_nss_downloads(dataset_path)

    results = [
        run_tests(test_dir, run_coverage=args.coverage, fast_test=args.fast_test)
        for test_dir in args.test_dirs
    ]
    success = all(results)

    sys.exit(0 if success else 1)
