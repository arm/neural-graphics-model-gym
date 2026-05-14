# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path

import torch

from ng_model_gym.core.model import create_model
from ng_model_gym.core.utils.enum_definitions import TrainEvalMode
from tests.base_gpu_test import BaseGPUMemoryTest
from tests.testing_utils import create_simple_params

_GOLDEN_ROOT = Path(__file__).resolve().parent / "data" / "nss_v1_golden_values"
_CHECK_KEYS = (
    "output_linear",
    "output",
    "out_filtered",
    "temporal_params",
    "derivative",
)


class TestNSSV1Golden(BaseGPUMemoryTest):
    """Parity test for the NSS v1 port against reference goldens."""

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required for the NSS v1 golden parity test.",
    )
    def test_forward_matches_reference_golden_values(self) -> None:
        """NSS v1 forward matches the stored reference tensors."""

        device = torch.device("cuda")
        params = create_simple_params(usecase="nss_v1")
        params.model_train_eval_mode = TrainEvalMode.FP32
        params.train.batch_size = 2
        params.model.recurrent_samples = 4

        golden_input = torch.load(
            _GOLDEN_ROOT / "nss_v1_input_golden.pt",
            map_location=device,
            weights_only=True,
        )
        golden_output = torch.load(
            _GOLDEN_ROOT / "nss_v1_output_golden.pt",
            map_location=device,
            weights_only=True,
        )["outputs"]

        model = create_model(params, device)
        model.get_neural_network().load_state_dict(golden_input["autoencoder_state"])
        model.eval()

        with torch.no_grad():
            model_output = model(golden_input["feedback_input"])

        for key in _CHECK_KEYS:
            with self.subTest(key=key):
                torch.testing.assert_close(
                    model_output[key],
                    golden_output[key],
                    rtol=1e-5,
                    atol=5e-6,
                )
