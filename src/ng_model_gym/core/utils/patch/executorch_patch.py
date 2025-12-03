# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSES/LicenseRef-BSD-ExecuTorch.txt file in the top-level directory.
# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0 AND LicenseRef-BSD-ExecuTorch
# pylint: skip-file

import importlib
import logging
import os
import platform
import shutil
import subprocess  # nosec B404
import tempfile
from typing import List

logger = logging.getLogger(__name__)


def vgf_compile(
    tosa_flatbuffer: bytes,
    compile_flags: List[str],
    artifact_path: str | None = None,
    tag_name: str = "",
):
    with tempfile.TemporaryDirectory() as tmpdir:
        # We currently write out a flatbuffer as input to the converter
        tosaname = f"output_{tag_name}.tosa"
        tosa_path = os.path.join(tmpdir, tosaname)
        with open(tosa_path, "wb") as f:
            f.write(tosa_flatbuffer)

        vgf_path = tosa_path + ".vgf"

        conversion_command = [
            "model-converter",
            *compile_flags,
            "-i",
            tosa_path,
            "-o",
            vgf_path,
        ]

        try:
            subprocess.run(  # nosec B603
                conversion_command, shell=False, check=True, capture_output=True
            )

        except subprocess.CalledProcessError as process_error:
            raise RuntimeError(
                f"Vgf compiler ('{conversion_command}') failed with error:\n \
                {process_error.stderr.decode()}\n \
                Stdout:\n{process_error.stdout.decode()}"
            )

        if artifact_path is not None:
            logger.debug(f"Emitting debug output to: {vgf_path=}")
            os.makedirs(artifact_path, exist_ok=True)
            shutil.copy2(vgf_path, artifact_path)

        vgf_bytes = open(vgf_path, "rb").read()
        return vgf_bytes


def _apply_patch():
    target_module = "executorch.backends.arm.vgf.backend"
    target_function = "vgf_compile"
    module = importlib.import_module(target_module)

    if hasattr(module, target_function):
        setattr(module, target_function, vgf_compile)
        logger.debug(f"Patch applied: {target_module}.{target_function}")
    else:
        logger.debug(f"Patch not applied: target function {target_function} not found.")


if platform.system() == "Windows":
    _apply_patch()
