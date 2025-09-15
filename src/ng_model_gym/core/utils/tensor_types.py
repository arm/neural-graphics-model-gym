# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, List, Union

import torch

TensorData = Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]
