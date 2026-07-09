# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch

from ng_model_gym.core.utils.enum_definitions import TrainEvalMode
from tests.testing_utils import create_simple_params

_GOLDEN_VALUES_DIR = Path("tests/usecases/nss/unit/data/nss_v1_golden_values")
NSS_V1_COMMON_OUTPUT_KEYS = (
    "output",
    "output_linear",
    "out_filtered",
    "temporal_params",
    "disocclusion_mask",
    "derivative",
    "ground_truth",
    "input_color",
)
NSS_V1_CORE_OUTPUT_KEYS = NSS_V1_COMMON_OUTPUT_KEYS
NSS_V1_RECURRENT_OUTPUT_KEYS = NSS_V1_COMMON_OUTPUT_KEYS + ("reset_event", "motion")


def load_nss_v1_golden(filename: str, device: torch.device) -> dict:
    """Load NSS v1 golden test data on the requested device."""

    return torch.load(
        _GOLDEN_VALUES_DIR / filename,
        map_location=device,
        weights_only=True,
    )


def create_nss_v1_test_params(quality: str):
    """Create shared NSS v1 params used by unit tests."""

    params = create_simple_params(usecase="nss_v1")
    params.model_train_eval_mode = TrainEvalMode.FP32
    params.model.quality = quality
    params.model.recurrent_samples = 2
    params.train.batch_size = 2
    return params


def tensor_values(values: dict) -> dict[str, torch.Tensor]:
    """Return tensor entries from a loaded golden-value dictionary."""

    return {
        key: value for key, value in values.items() if isinstance(value, torch.Tensor)
    }


def assert_tensors_close(
    actual: dict,
    expected: dict,
    keys: tuple[str, ...],
    *,
    rtol: float,
    atol: float,
) -> None:
    """Assert that selected tensors in two dictionaries are close."""

    for key in keys:
        torch.testing.assert_close(
            actual[key],
            expected[key],
            rtol=rtol,
            atol=atol,
        )
