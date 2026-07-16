# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import mock

import torch
from torchao.quantization.pt2e import MovingAverageMinMaxObserver
from torchao.quantization.pt2e.fake_quantize import FixedQParamsFakeQuantize
from torchao.quantization.pt2e.observer import FixedQParamsObserver

from ng_model_gym.core.quantization import (
    FixedQParamsFakeQuantizeFix,
    FusedMovingAvgObsFakeQuantizeFix,
    replace_fixed_qparams_fake_quant,
)

# pylint: disable=missing-function-docstring

_ZERO_POINT_ERROR = "`zero_point` must be between `quant_min` and `quant_max`"


def _fused_fake_quant():
    return FusedMovingAvgObsFakeQuantizeFix(
        observer=MovingAverageMinMaxObserver.with_args(
            dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
        ),
        quant_min=-128,
        quant_max=127,
    )


def _fixed_fake_quant(fake_quant_type=FixedQParamsFakeQuantizeFix):
    return fake_quant_type(
        observer=FixedQParamsObserver.with_args(
            scale=1.0 / 254.0,
            zero_point=-127,
            dtype=torch.int8,
            qscheme=torch.per_tensor_affine,
            quant_min=-127,
            quant_max=127,
        )
    )


class TestFakeQuantObservers(unittest.TestCase):
    """Test hardened fake-quant behavior."""

    def assert_qparams_are_safe(self, fake_quant):
        observer = fake_quant.activation_post_process
        self.assertTrue(torch.all(torch.isfinite(fake_quant.scale)))
        self.assertTrue(torch.all(fake_quant.scale > 0))
        self.assertGreaterEqual(int(fake_quant.zero_point.min()), observer.quant_min)
        self.assertLessEqual(int(fake_quant.zero_point.max()), observer.quant_max)

    def test_fused_fake_quant_sanitizes_invalid_qparams(self):
        fake_quant = _fused_fake_quant()
        x = torch.ones(4)

        def fused_kernel(*_args):
            fake_quant.scale.fill_(float("nan"))
            fake_quant.zero_point.fill_(torch.iinfo(fake_quant.zero_point.dtype).min)
            return x

        with mock.patch(
            "torch.fused_moving_avg_obs_fake_quant", side_effect=fused_kernel
        ):
            fake_quant(x)

        self.assert_qparams_are_safe(fake_quant)

    def test_fused_fake_quant_recovers_from_zero_point_error(self):
        fake_quant = _fused_fake_quant()
        fake_quant.zero_point.fill_(torch.iinfo(fake_quant.zero_point.dtype).min)
        x = torch.ones(4)

        with (
            mock.patch(
                "torch.fused_moving_avg_obs_fake_quant",
                side_effect=RuntimeError(_ZERO_POINT_ERROR),
            ),
            self.assertLogs(
                "ng_model_gym.core.quantization.observers", level="WARNING"
            ),
        ):
            output = fake_quant(x)

        torch.testing.assert_close(output, x)
        self.assertEqual(fake_quant.observer_enabled.item(), 0)
        self.assert_qparams_are_safe(fake_quant)

    def test_fused_fake_quant_does_not_hide_unrelated_errors(self):
        fake_quant = _fused_fake_quant()

        with mock.patch(
            "torch.fused_moving_avg_obs_fake_quant",
            side_effect=RuntimeError("unrelated failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "unrelated failure"):
                fake_quant(torch.ones(4))

    def test_fixed_fake_quant_compiles_and_respects_enabled_flag(self):
        fake_quant = _fixed_fake_quant()
        compiled = torch.compile(fake_quant, backend="eager", fullgraph=True)
        x = torch.linspace(0.0, 1.0, 16)

        torch.testing.assert_close(compiled(x), fake_quant(x))

        fake_quant.disable_fake_quant()
        torch.testing.assert_close(compiled(x), x)

    def test_replacement_preserves_state(self):
        module = torch.nn.Sequential(
            torch.nn.Sequential(_fixed_fake_quant(FixedQParamsFakeQuantize))
        )
        original_state = {
            name: value.clone() for name, value in module.state_dict().items()
        }

        self.assertEqual(replace_fixed_qparams_fake_quant(module), 1)

        replacement = module[0][0]
        self.assertIs(type(replacement), FixedQParamsFakeQuantizeFix)
        for name, value in original_state.items():
            torch.testing.assert_close(module.state_dict()[name], value)
        self.assertEqual(replace_fixed_qparams_fake_quant(module), 0)
