# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import ast
import unittest
from pathlib import Path


class TestFolderStructure(unittest.TestCase):
    """Unit tests for folder structure"""

    def test_core_does_not_import_usecases(self):
        """Test modules in core do not import from usecase"""
        violations = []
        for py in Path("src/ng_model_gym/core").rglob("*.py"):
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except Exception as e:  # pylint: disable=broad-exception-caught
                self.fail(f"Could not parse {py}: {e}")

            matches = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        n = alias.name
                        if "usecases" in n:
                            matches.add(n)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    n = node.module
                    if "usecases" in n:
                        matches.add(n)

            if matches:
                violations.append(f"{py}: {', '.join(sorted(matches))}")

        self.assertFalse(
            violations,
            "Core module files importing usecase imports found:\n"
            + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
