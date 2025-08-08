# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
# pylint: skip-file

import sys
import types

import serializer.tosa_serializer as real_serializer
from tosa import ResizeMode as real_ResizeMode  # Adjust if needed

# Create fake packages
tosa_tools = types.ModuleType("tosa_tools")
tosa_tools.__path__ = []

v0_80 = types.ModuleType("tosa_tools.v0_80")
v0_80.__path__ = []

serializer = types.ModuleType("tosa_tools.v0_80.serializer")
serializer.__path__ = []

tosa_pkg = types.ModuleType("tosa_tools.v0_80.tosa")
tosa_pkg.__path__ = []

# Create a fake module named ResizeMode, which exports the ResizeMode symbol
resize_module = types.ModuleType("tosa_tools.v0_80.tosa.ResizeMode")
resize_module.ResizeMode = real_ResizeMode.ResizeMode

# Register everything
sys.modules["tosa_tools"] = tosa_tools
sys.modules["tosa_tools.v0_80"] = v0_80
sys.modules["tosa_tools.v0_80.serializer"] = serializer
sys.modules["tosa_tools.v0_80.serializer.tosa_serializer"] = real_serializer

sys.modules["tosa_tools.v0_80.tosa"] = tosa_pkg
sys.modules["tosa_tools.v0_80.tosa.ResizeMode"] = resize_module
