# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import os
import pkgutil


def apply_patch():
    """
    get_importer() from pkgutil.py needs to be patched for Python 3.10,
    otherwise tests will fail
    """
    get_importer = pkgutil.get_importer

    def patched_get_importer(path_item):
        path_item = os.fsdecode(path_item)
        return get_importer(path_item)

    pkgutil.get_importer = patched_get_importer
