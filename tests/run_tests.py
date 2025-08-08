# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import argparse
import glob
import os
import subprocess
import sys
import unittest

from tests.fetch_huggingface import validate_downloads

from .pkgutil_patch import apply_patch


def run_tests(start_test_dir, sequential=False, run_coverage=False):
    """Run tests recursively from start_test_dir"""
    if sequential:
        failed_tests = run_in_own_process(start_test_dir, run_coverage)
        return not failed_tests
    loader = unittest.TestLoader()

    tests = loader.discover(start_dir=start_test_dir, pattern="test_*.py")

    # Verbosity=2 prints each test name - otherwise no idea which test is running
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    return result.wasSuccessful()


def run_in_own_process(start_test_dir, run_coverage):
    """Run each test file in its own process sequentially to isolate GPU memory usage."""
    failed_tests = []
    failed_outputs = {}

    pattern = os.path.join(start_test_dir, "**", "test_*.py")
    test_files = sorted(glob.glob(pattern, recursive=True))

    for test_file in test_files:
        print(f"Running Test: {test_file}")

        if run_coverage:
            cmd = ["coverage", "run", "--parallel-mode", test_file]
        else:
            cmd = [sys.executable, "-m", "unittest", test_file]

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"

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
        "--sequential",
        action="store_true",
        help="Run each test file in its own process",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Enable coverage in subprocesses",
    )
    args = parser.parse_args()

    # Check required pretrained weights/datasets have been downloaded
    validate_downloads()

    results = [
        run_tests(test_dir, sequential=args.sequential, run_coverage=args.coverage)
        for test_dir in args.test_dirs
    ]
    success = all(results)

    sys.exit(0 if success else 1)
