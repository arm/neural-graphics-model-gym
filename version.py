# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import re


def normalise_tag(tag: str, params: dict) -> str:  # pylint: disable=unused-argument
    """
    Normalise the tag to be PEP-440 compliant for building the wheel
    Works with both 'upstream/v' and 'v' tags, plus optional end suffix e.g. '-rc1'
        v0.1.0          -> 0.1.0
        upstream/v0.1.0 -> 0.1.0
        v0.1.0-rc1      -> 0.1.0rc1
    """
    tag_matched = re.match(
        r"^(?:upstream/)?v?(?P<base>[0-9]+(?:\.[0-9]+)*)(?:-(?P<suffix>[A-Za-z0-9\.]+))?$",
        tag,
    )

    if not tag_matched:
        raise ValueError(f"Tag {tag} is not in a recognised format.")

    base = tag_matched.group("base")
    suffix = tag_matched.group("suffix") or ""

    return base + suffix
