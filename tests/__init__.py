# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

from ng_model_gym.core.utils.logging_utils import filter_warnings
from ng_model_gym.usecases import import_usecase_files

filter_warnings()

# Import all use case models and datasets to trigger registration
import_usecase_files()
