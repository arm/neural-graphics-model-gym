# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import os
import platform
import sys
from pathlib import Path

from ng_model_gym.core.utils.logging_utils import filter_warnings, WARNING_FILTERS
from tests.fetch_huggingface import (
    nfru_test_assets_enabled,
    validate_nfru_datasets,
    validate_nss_downloads,
)
from tests.pkgutil_patch import apply_patch

filter_warnings()


def warning_filter_to_pytest(warning_filter: dict) -> str:
    """Convert a Python warnings filter dictionary to pytest's string format."""
    category = warning_filter["category"].__name__
    module = warning_filter.get("module", "")
    return f"{warning_filter['action']}:{warning_filter['message']}:{category}:{module}"


class _FastTestState:
    enabled = False
    previous = None


if platform.system() == "Windows":
    # For Windows, avoid rebuilding .dlls that might already be loaded in another process.
    os.environ.setdefault("SKIP_NINJA_CHECK", "1")


def pytest_addoption(parser) -> None:
    """Add command line options for PyTest to use."""
    group = parser.getgroup("ng-model-gym")
    group.addoption(
        "--fast-test",
        action="store_true",
        help="Use the reduced-size datasets and enable the slang FAST_TEST optimisations.",
    )


def pytest_configure(config) -> None:
    """Configuration hook, called after command-line options get parsed."""
    filter_warnings()
    for warning_filter in WARNING_FILTERS:
        config.addinivalue_line(
            "filterwarnings", warning_filter_to_pytest(warning_filter)
        )

    # Apply patch for Python 3.10 before running tests.
    if sys.version_info[:2] == (3, 10):
        apply_patch()

    if config.getoption("--fast-test"):
        _FastTestState.enabled = True
        _FastTestState.previous = os.environ.get("FAST_TEST")
        os.environ["FAST_TEST"] = "1"


def pytest_sessionstart(session) -> None:  # pylint: disable=unused-argument
    """Hook called after PyTest Session object is created."""
    dataset_path = Path("tests/usecases/nss/datasets")
    validate_nss_downloads(dataset_path)
    if nfru_test_assets_enabled():
        nfru_dataset_path = Path("tests/usecases/nfru/data/nfru_sample")
        validate_nfru_datasets(nfru_dataset_path)


def pytest_unconfigure(config) -> None:  # pylint: disable=unused-argument
    """Clean up any configuration changes made."""
    if _FastTestState.enabled:
        if _FastTestState.previous is None:
            os.environ.pop("FAST_TEST", None)
        else:
            os.environ["FAST_TEST"] = _FastTestState.previous
        _FastTestState.enabled = False
        _FastTestState.previous = None
