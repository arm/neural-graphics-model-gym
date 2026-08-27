# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest
from collections import OrderedDict

import torch
from torch import nn

from ng_model_gym.core.model.qat_checkpoint_compat import prepare_legacy_qat_state_dict

# pylint: disable=abstract-method,missing-function-docstring


class _Observer(nn.Module):
    """Minimal stateful observer module for compatibility tests."""

    def __init__(self, value: float):
        super().__init__()
        self.register_buffer("scale", torch.tensor(value))
        self.register_buffer("zero_point", torch.tensor(int(value)))


def _add_network_root(model: nn.Module, root: str, include_buffer: bool) -> nn.Module:
    """Build a small module with the state categories used by QAT graphs."""
    parent = model
    components = root.split(".")
    for component in components:
        child = nn.Module()
        parent.add_module(component, child)
        parent = child

    parent.add_module("first", nn.Linear(2, 2))
    if include_buffer:
        parent.register_buffer("running_mean", torch.tensor([1.0, 2.0]))
    parent.add_module("activation_post_process_0", _Observer(1.0))
    parent.add_module("activation_post_process_1", _Observer(2.0))
    return model


def _legacy_state_dict(model: nn.Module, root: str) -> OrderedDict[str, torch.Tensor]:
    """Create an old exported-program state dict for the synthetic target."""
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for index, (_, value) in enumerate(
        (item for item in model.named_parameters() if item[0].startswith(f"{root}."))
    ):
        state_dict[f"{root}._param_constant{index}"] = value.detach().clone()

    non_observer_buffers = [
        value
        for name, value in model.named_buffers()
        if name.startswith(f"{root}.") and ".activation_post_process_" not in name
    ]
    for index, value in enumerate(non_observer_buffers):
        state_dict[f"{root}._tensor_constant{index}"] = value.detach().clone()

    target_state = model.state_dict()
    for legacy_index, target_index in ((3, 0), (7, 1)):
        target_prefix = f"{root}.activation_post_process_{target_index}"
        for key, value in target_state.items():
            if key.startswith(f"{target_prefix}."):
                suffix = key.removeprefix(target_prefix)
                state_dict[
                    f"{root}.activation_post_process_{legacy_index}{suffix}"
                ] = value.detach().clone()
    return state_dict


class TestQATCheckpointCompatibility(unittest.TestCase):
    """Test translation of bundled legacy QAT checkpoint names."""

    def test_modern_state_dict_is_returned_unchanged(self):
        model = _add_network_root(nn.Module(), "autoencoder", include_buffer=False)
        state_dict = model.state_dict()

        prepared = prepare_legacy_qat_state_dict(model, state_dict)

        self.assertIs(prepared, state_dict)

    def test_translates_nss_parameter_constants_and_observers(self):
        model = _add_network_root(nn.Module(), "autoencoder", include_buffer=False)
        legacy_state = _legacy_state_dict(model, "autoencoder")

        prepared = prepare_legacy_qat_state_dict(model, legacy_state)

        self.assertEqual(list(prepared), list(model.state_dict()))
        model.load_state_dict(prepared, strict=True)

    def test_translates_nfru_parameter_buffer_constants_and_observers(self):
        root = "network.auto_encoder"
        model = _add_network_root(nn.Module(), root, include_buffer=True)
        legacy_state = _legacy_state_dict(model, root)

        prepared = prepare_legacy_qat_state_dict(model, legacy_state)

        self.assertEqual(list(prepared), list(model.state_dict()))
        model.load_state_dict(prepared, strict=True)

    def test_rejects_unsupported_legacy_root(self):
        model = _add_network_root(nn.Module(), "other", include_buffer=False)
        legacy_state = _legacy_state_dict(model, "other")

        with self.assertRaisesRegex(
            ValueError, "Unsupported legacy QAT checkpoint root"
        ):
            prepare_legacy_qat_state_dict(model, legacy_state)

    def test_rejects_generated_constant_count_mismatch(self):
        model = _add_network_root(nn.Module(), "autoencoder", include_buffer=False)
        legacy_state = _legacy_state_dict(model, "autoencoder")
        del legacy_state["autoencoder._param_constant1"]

        with self.assertRaisesRegex(ValueError, "parameter count mismatch"):
            prepare_legacy_qat_state_dict(model, legacy_state)
