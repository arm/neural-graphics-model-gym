# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    """Resolve package version from installed metadata."""
    try:
        return version("ng-model-gym")
    except PackageNotFoundError:
        return "0.0.0+unknown"
