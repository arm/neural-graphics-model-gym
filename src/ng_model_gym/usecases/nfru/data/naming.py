# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


class DataVariable:
    """Utility for parsing and reformatting variable names stored in safetensors."""

    def __init__(self, var_name: str) -> None:
        self.is_mv = False
        self.concrete = False
        version_split = var_name.split("@")
        if len(version_split) > 1:
            self.version = version_split[1]
            var_name = version_split[0]
        else:
            self.version = ""
        var_split = var_name.split("_")
        if var_split[0] in ("sy", "flow", "mv"):
            self.is_mv = True
            if len(var_split) != 4:
                raise ValueError(
                    "Expected motion variable format '<src>_<from>_<fps>_<to>' "
                    f"but received '{var_name}'."
                )
            self.src, self.vec_from, self.fps, self.vec_to = var_split
            try:
                if self.vec_from != "{}":
                    self.concrete = True
                    self.ivec_from = convert_str_offset_to_int(self.vec_from)
                else:
                    self.ivec_from = 0
                self.ivec_to = convert_str_offset_to_int(self.vec_to)
                self.ifps = int(self.fps[1:])
            except ValueError as exc:
                raise ValueError(
                    f"Variable {var_name} was in an incorrect naming convention format"
                ) from exc
        self.var_name = var_name

    def generate_concrete_variable(self, frame: int, timeline_fps: int = 60):
        """Generate an absolute naming token for the given frame index."""
        new_vec_from = convert_int_to_str_offset(frame)
        if not self.is_mv:
            return f"{self.var_name}_{new_vec_from}"
        version_suffix = "" if not self.version else f"@{self.version}"
        new_vec_to = convert_int_to_str_offset(
            frame + int(self.ivec_to * (timeline_fps / self.ifps))
        )
        return f"{self.src}_{new_vec_from}_{self.fps}_{new_vec_to}{version_suffix}"

    def generate_non_concrete_variable(self, timeline_fps: int = 60):
        """Return a templated variable name with relative offsets encoded."""
        if not self.is_mv:
            return remove_suffix_if_needed(self.var_name)
        version_suffix = "" if not self.version else f"@{self.version}"
        new_vec_to = convert_int_to_str_offset(
            int((self.ivec_to - self.ivec_from) * (self.ifps / timeline_fps))
        )
        return f"{self.src}_{{}}_{self.fps}_{new_vec_to}{version_suffix}"


def remove_suffix_if_needed(s: str):
    """Drop trailing offset if present (e.g., mv_t becomes mv)."""
    if has_offset_suffix(s):
        s_split = s.split("_")
        return "_".join(s_split[:-1])
    return s


def has_offset_suffix(s: str):
    """Check if the string ends with an offset token (m/p/t)."""
    s_split = s.split("_")
    if len(s_split) <= 1:
        return False
    suffix = s_split[-1]
    sign_char = suffix[0].lower()
    if sign_char not in ["m", "p", "t"]:
        return False
    return True


def convert_str_offset_to_int(s: str) -> int:
    """
    Converts strings like 'm1' -> -1, 'p1' -> +1, 'm10' -> -10, 'p25' -> +25.
    """
    if not s:
        raise ValueError("Input string cannot be empty")

    sign_char = s[0].lower()
    if sign_char not in ["m", "p", "t"]:
        raise ValueError(f"Invalid format: {s}. Must start with 'm', 'p' or 't'.")

    if sign_char == "t":
        return 0

    try:
        number = int(s[1:])
    except ValueError as exc:
        raise ValueError(f"Invalid number format: {s[1:]}") from exc

    return -number if sign_char == "m" else number


def convert_int_to_str_offset(n: int) -> str:
    """
    Converts integers into string offsets:
    -1 -> 'm1'
     1 -> 'p1'
    -10 -> 'm10'
     25 -> 'p25'
     0 -> 't'
    """
    if n == 0:
        return "t"
    if n < 0:
        return f"m{abs(n)}"
    return f"p{n}"
