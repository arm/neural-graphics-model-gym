# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
import shutil
import tempfile
import unittest
from pathlib import Path

from ng_model_gym.utils.logging import add_file_handler, set_log_level
from tests.unit.utils.utils import clear_loggers


class TestLogging(unittest.TestCase):
    """Unit tests for functions in logging.py file."""

    def tearDown(self):
        """Close and remove handlers from loggers."""
        clear_loggers()

    def test_set_log_level(self):
        """Test set_log_level function."""
        # Create logger.
        logger_name = "test_logger"
        logger = logging.getLogger(logger_name)

        # Only second message should be printed, as logging is set to ERROR.
        with self.assertLogs(logger) as al:
            set_log_level(logger_name, logging.ERROR)
            logger.info("First message")
            logger.error("Second message")
            logger.info("Third message")
        self.assertEqual(al.output, ["ERROR:test_logger:Second message"])


class TestFileLogging(unittest.TestCase):
    """Unit tests for file based functions in logging.py file."""

    def setUp(self):
        """Create a temporary directory for testing"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up the temporary directory and remove handlers from loggers."""
        shutil.rmtree(self.test_dir)
        clear_loggers()

    def test_add_file_handler(self):
        """Test add_file_handler function."""
        logger_name = "test_logger"
        logger = logging.getLogger(logger_name)

        set_log_level(logger_name, logging.INFO)
        add_file_handler(logger_name, self.test_dir, "output.log")
        logger.info("Test message.")

        # Verify that the log file was created.
        file_path = Path(self.test_dir, "output.log")
        self.assertTrue(Path(file_path).exists())

        # Verify that the log file contents matches the log.
        with open(file_path, "r", encoding="utf-8") as file:
            contents = file.read()
        self.assertIn("test_logger - INFO - Test message.\n", contents)


if __name__ == "__main__":
    unittest.main()
