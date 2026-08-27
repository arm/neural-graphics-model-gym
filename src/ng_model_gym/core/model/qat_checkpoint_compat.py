# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
"""Compatibility helpers for loading older QAT checkpoints."""

import re
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
from torch import nn

_SUPPORTED_ROOTS = frozenset(("autoencoder", "network.auto_encoder"))
_PARAM_CONSTANT_PATTERN = re.compile(r"^(?P<root>.+)\._param_constant(?P<index>\d+)$")
_TENSOR_CONSTANT_PATTERN = re.compile(r"^(?P<root>.+)\._tensor_constant(?P<index>\d+)$")
_OBSERVER_PATTERN = re.compile(
    r"^(?P<root>.+)\.activation_post_process_(?P<index>\d+)(?P<suffix>\..+)$"
)


def _get_legacy_root(state_dict: Dict[str, torch.Tensor]) -> Optional[str]:
    """Return the single supported legacy root name, or None for a modern state dict."""
    roots = {
        match.group("root")
        for key in state_dict
        for pattern in (_PARAM_CONSTANT_PATTERN, _TENSOR_CONSTANT_PATTERN)
        if (match := pattern.match(key)) is not None
    }
    if not roots:
        return None
    if len(roots) != 1 or not roots.issubset(_SUPPORTED_ROOTS):
        raise ValueError(
            "Unsupported legacy QAT checkpoint root(s): " + ", ".join(sorted(roots))
        )
    return next(iter(roots))


def _generated_keys(
    state_dict: Dict[str, torch.Tensor],
    pattern: re.Pattern[str],
    root: str,
    category: str,
) -> List[str]:
    """Return generated constant keys in numeric traversal order."""
    indexed_keys = [
        (int(match.group("index")), key)
        for key in state_dict
        if (match := pattern.match(key)) is not None and match.group("root") == root
    ]
    indexed_keys.sort()
    indices = [index for index, _ in indexed_keys]
    if indices != list(range(len(indices))):
        raise ValueError(
            f"Legacy QAT checkpoint {category} indices are not contiguous: {indices}"
        )
    return [key for _, key in indexed_keys]


def _observer_prefixes(state_dict: Dict[str, torch.Tensor], root: str) -> List[str]:
    """Return stateful observer module prefixes in numeric order."""
    prefixes_by_index: Dict[int, str] = {}
    for key in state_dict:
        match = _OBSERVER_PATTERN.match(key)
        if match is None or match.group("root") != root:
            continue
        index = int(match.group("index"))
        prefix = f"{root}.activation_post_process_{index}"
        existing_prefix = prefixes_by_index.setdefault(index, prefix)
        if existing_prefix != prefix:
            raise ValueError(f"Ambiguous observer index {index} in QAT checkpoint")
    return [prefixes_by_index[index] for index in sorted(prefixes_by_index)]


def _pair_renames(
    category: str, legacy: List[str], target: List[str]
) -> Dict[str, str]:
    """Pair traversal-ordered legacy and target keys after validating counts."""
    if len(legacy) != len(target):
        raise ValueError(
            f"Legacy QAT checkpoint {category} count mismatch: "
            f"checkpoint has {len(legacy)}, prepared model expects {len(target)}"
        )
    return dict(zip(legacy, target, strict=True))


def _constant_renames(
    model: nn.Module, state_dict: Dict[str, torch.Tensor], root: str
) -> Dict[str, str]:
    """Build generated parameter and buffer renames."""
    root_prefix = f"{root}."
    legacy_parameters = _generated_keys(
        state_dict, _PARAM_CONSTANT_PATTERN, root, "parameter"
    )
    target_parameters = [
        name for name, _ in model.named_parameters() if name.startswith(root_prefix)
    ]
    legacy_buffers = _generated_keys(
        state_dict, _TENSOR_CONSTANT_PATTERN, root, "buffer"
    )
    target_buffers = [
        name
        for name, _ in model.named_buffers()
        if name.startswith(root_prefix)
        and not name.startswith(f"{root}.activation_post_process_")
    ]

    renames = _pair_renames("parameter", legacy_parameters, target_parameters)
    renames.update(_pair_renames("buffer", legacy_buffers, target_buffers))
    return renames


def _observer_renames(
    state_dict: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
    root: str,
) -> Dict[str, str]:
    """Build full-key renames from ordered legacy and target observer modules."""
    prefix_renames = _pair_renames(
        "observer",
        _observer_prefixes(state_dict, root),
        _observer_prefixes(target_state, root),
    )
    renames: Dict[str, str] = {}
    for key in state_dict:
        match = _OBSERVER_PATTERN.match(key)
        if match is None or match.group("root") != root:
            continue
        suffix = match.group("suffix")
        legacy_prefix = key.removesuffix(suffix)
        renames[key] = prefix_renames[legacy_prefix] + suffix
    return renames


def _translate_and_validate(
    state_dict: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
    renames: Dict[str, str],
) -> Dict[str, torch.Tensor]:
    """Apply renames, reject collisions, and return the target state ordering."""
    translated_by_key: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        translated_key = renames.get(key, key)
        if translated_key in translated_by_key:
            raise ValueError(
                f"Legacy QAT checkpoint translation produced duplicate key: "
                f"{translated_key}"
            )
        translated_by_key[translated_key] = value

    target_keys = set(target_state)
    translated_keys = set(translated_by_key)
    if target_keys != translated_keys:
        missing = sorted(target_keys - translated_keys)
        unexpected = sorted(translated_keys - target_keys)
        raise ValueError(
            "Legacy QAT checkpoint translation does not match the prepared model. "
            f"Missing keys: {missing}; unexpected keys: {unexpected}"
        )

    return OrderedDict((key, translated_by_key[key]) for key in target_state)


def prepare_legacy_qat_state_dict(
    model: nn.Module, state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Rename an old NSS/NFRU exported-program QAT state dict for strict loading.

    Torch export used generated constant names for parameters and buffers in the
    bundled checkpoints. Current non-composable QAT preparation exposes semantic
    module names instead. Both layouts retain traversal order, which provides the
    stable correspondence used here.
    """
    root = _get_legacy_root(state_dict)
    if root is None:
        # In this case we have FP32 or modern QAT checkpoint
        return state_dict

    target_state = model.state_dict()
    renames = _constant_renames(model, state_dict, root)
    renames.update(_observer_renames(state_dict, target_state, root))
    return _translate_and_validate(state_dict, target_state, renames)
