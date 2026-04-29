# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

from ng_model_gym.usecases.nfru.data.naming import (
    convert_int_to_str_offset,
    convert_str_offset_to_int,
    DataVariable,
    has_offset_suffix,
    remove_suffix_if_needed,
)


class TestNaming(unittest.TestCase):
    """Test the DataVariable utility for parsing and reformatting variable names."""

    def test_init_mv_variable(self):
        """Test initialization of DataVariable with a motion vector variable."""
        var_name = "mv_m1_f60_p1"
        data_var = DataVariable(var_name)
        self.assertTrue(data_var.is_mv)
        self.assertTrue(data_var.concrete)

        self.assertEqual(data_var.version, "")
        self.assertEqual(data_var.src, "mv")
        self.assertEqual(data_var.vec_from, "m1")
        self.assertEqual(data_var.fps, "f60")
        self.assertEqual(data_var.vec_to, "p1")

        self.assertEqual(data_var.ivec_from, -1)
        self.assertEqual(data_var.ivec_to, 1)
        self.assertEqual(data_var.ifps, 60)

    def test_init_mv_variable_with_version(self):
        """
        Test initialization of DataVariable with a motion vector variable
        with a version suffix.
        """
        var_name = "mv_m1_60_p1@v1"
        data_var = DataVariable(var_name)
        self.assertTrue(data_var.is_mv)
        self.assertTrue(data_var.concrete)

        self.assertEqual(data_var.version, "v1")
        self.assertEqual(data_var.src, "mv")
        self.assertEqual(data_var.vec_from, "m1")
        self.assertEqual(data_var.fps, "60")
        self.assertEqual(data_var.vec_to, "p1")

        self.assertEqual(data_var.ivec_from, -1)
        self.assertEqual(data_var.ivec_to, 1)
        self.assertEqual(data_var.ifps, 0)

    def test_init_non_mv_variable(self):
        """Test initialization of DataVariable with a non-motion vector variable."""
        var_name = "non_mv_var"
        data_var = DataVariable(var_name)
        self.assertFalse(data_var.is_mv)
        self.assertFalse(data_var.concrete)

        self.assertEqual(data_var.version, "")
        # The following sttributes are only set for motion vector variables
        self.assertFalse(hasattr(data_var, "src"))
        self.assertFalse(hasattr(data_var, "vec_from"))
        self.assertFalse(hasattr(data_var, "fps"))
        self.assertFalse(hasattr(data_var, "vec_to"))
        self.assertFalse(hasattr(data_var, "ivec_from"))
        self.assertFalse(hasattr(data_var, "ivec_to"))
        self.assertFalse(hasattr(data_var, "ifps"))

    def test_generate_concrete_variable_no_version(self):
        """Test generation of concrete variable names without a version."""
        var_name = "mv_m1_f60_p1"
        frame = 3
        data_var = DataVariable(var_name)
        expected_concrete_name = "mv_p3_f60_p4"
        self.assertEqual(
            data_var.generate_concrete_variable(frame), expected_concrete_name
        )

    def test_generate_concrete_variable_w_version(self):
        """Test generation of concrete variable names with a version."""
        var_name = "mv_m1_f60_p1@v1"
        frame = 3
        data_var = DataVariable(var_name)
        expected_concrete_name = "mv_p3_f60_p4@v1"
        self.assertEqual(
            data_var.generate_concrete_variable(frame), expected_concrete_name
        )

    def test_generate_concrete_variable_non_mv(self):
        """Test generation of concrete variable names for non-motion vector variables."""
        var_name = "non_mv_var"
        frame = 5
        data_var = DataVariable(var_name)
        expected_concrete_name = "non_mv_var_p5"
        self.assertEqual(
            data_var.generate_concrete_variable(frame), expected_concrete_name
        )

    def test_generate_non_concrete_variable_no_version(self):
        """Test generation of non-concrete variable names without a version."""
        var_name = "mv_m1_f60_p1"
        data_var = DataVariable(var_name)
        expected_non_concrete_name = "mv_{}_f60_p2"
        self.assertEqual(
            data_var.generate_non_concrete_variable(), expected_non_concrete_name
        )

    def test_generate_non_concrete_variable_w_version(self):
        """Test generation of non-concrete variable names with a version."""
        var_name = "mv_m1_f60_p1@v1"
        data_var = DataVariable(var_name)
        expected_non_concrete_name = "mv_{}_f60_p2@v1"
        self.assertEqual(
            data_var.generate_non_concrete_variable(), expected_non_concrete_name
        )

    def test_generate_non_concrete_variable_non_mv(self):
        """Test generation of non-concrete variable names for non-motion vector variables."""
        var_name = "non_mv_var_p5"
        data_var = DataVariable(var_name)
        expected_non_concrete_name = "non_mv_var"
        self.assertEqual(
            data_var.generate_non_concrete_variable(), expected_non_concrete_name
        )

    def test_remove_suffix_if_needed(self):
        """Test removal of offset suffixes from variable names."""
        self.assertEqual(remove_suffix_if_needed("var_m1"), "var")
        self.assertEqual(remove_suffix_if_needed("var_p10"), "var")
        self.assertEqual(remove_suffix_if_needed("var_t"), "var")

        self.assertEqual(remove_suffix_if_needed("var"), "var")
        self.assertEqual(remove_suffix_if_needed("var_x5"), "var_x5")

    def test_has_offset_suffix(self):
        """Test detection of offset suffixes in variable names."""
        self.assertTrue(has_offset_suffix("var_m1"))
        self.assertTrue(has_offset_suffix("var_p10"))
        self.assertTrue(has_offset_suffix("var_t"))

        self.assertFalse(has_offset_suffix("var"))
        self.assertFalse(has_offset_suffix("var_x5"))

    def test_convert_str_to_int_offset(self):
        """
        Test conversion of string frame offsets to integer format.

        'm1' -> -1
        'p1' -> +1
        'm10' -> -10,
        'p25' -> +25
        't' -> 0
        """
        self.assertEqual(convert_str_offset_to_int("m1"), -1)
        self.assertEqual(convert_str_offset_to_int("p1"), 1)
        self.assertEqual(convert_str_offset_to_int("m10"), -10)
        self.assertEqual(convert_str_offset_to_int("p25"), 25)
        self.assertEqual(convert_str_offset_to_int("t"), 0)

    def test_convert_str_to_int_offset_empty_input(self):
        """Test that empty string input raises an error."""
        with self.assertRaises(ValueError) as e:
            convert_str_offset_to_int("")
        self.assertEqual(str(e.exception), "Input string cannot be empty")

    def test_convert_str_to_int_offset_invalid_format(self):
        """Test that invalid string formats raise an error."""
        test_str = "x5"
        with self.assertRaises(ValueError) as e:
            convert_str_offset_to_int(test_str)
        self.assertEqual(
            str(e.exception),
            f"Invalid format: {test_str}. Must start with 'm', 'p' or 't'.",
        )

    def test_convert_str_to_int_offset_non_integer(self):
        """Test that non-integer offsets raise an error."""
        test_str = "m1.5"
        with self.assertRaises(ValueError) as e:
            convert_str_offset_to_int(test_str)
        self.assertEqual(str(e.exception), f"Invalid number format: {test_str[1:]}")

    def test_convert_int_to_str_offset(self):
        """
        Test conversion of integer frame offsets to string format.

        -1 -> 'm1'
         1 -> 'p1'
        -10 -> 'm10'
         25 -> 'p25'
         0 -> 't'
        """
        self.assertEqual(convert_int_to_str_offset(-1), "m1")
        self.assertEqual(convert_int_to_str_offset(1), "p1")
        self.assertEqual(convert_int_to_str_offset(-10), "m10")
        self.assertEqual(convert_int_to_str_offset(25), "p25")
        self.assertEqual(convert_int_to_str_offset(0), "t")
