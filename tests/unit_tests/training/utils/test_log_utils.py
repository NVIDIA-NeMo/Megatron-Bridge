# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from unittest.mock import patch

from megatron.bridge.training.utils.log_utils import setup_logging


class TestSetupLogging:
    """Test cases for the setup_logging function."""

    def setup_method(self):
        """Setup before each test method."""
        # Store original logging state
        self.original_root_level = logging.getLogger().level
        self.original_env = os.environ.get("MEGATRON_BRIDGE_LOGGING_LEVEL")

        # Store original filters for all loggers
        self.original_filters = {}
        root_logger = logging.getLogger()
        self.original_filters["root"] = root_logger.filters[:]

        for logger_name in logging.root.manager.loggerDict:
            logger = logging.getLogger(logger_name)
            self.original_filters[logger_name] = logger.filters[:]

        # Clean up any existing test loggers
        for logger_name in list(logging.root.manager.loggerDict.keys()):
            if logger_name.startswith("test_") or logger_name.startswith("megatron.bridge.test"):
                del logging.root.manager.loggerDict[logger_name]

    def teardown_method(self):
        """Cleanup after each test method."""
        # Restore original logging state
        logging.getLogger().setLevel(self.original_root_level)

        # Restore environment variable
        if self.original_env is not None:
            os.environ["MEGATRON_BRIDGE_LOGGING_LEVEL"] = self.original_env
        elif "MEGATRON_BRIDGE_LOGGING_LEVEL" in os.environ:
            del os.environ["MEGATRON_BRIDGE_LOGGING_LEVEL"]

        # Restore original filters for all loggers
        for logger_name, original_filters in self.original_filters.items():
            if logger_name == "root":
                logger = logging.getLogger()
            else:
                logger = logging.getLogger(logger_name)

            # Clear all current filters
            logger.filters.clear()

            # Restore original filters
            for filter_obj in original_filters:
                logger.addFilter(filter_obj)

        # Clean up test loggers
        for logger_name in list(logging.root.manager.loggerDict.keys()):
            if logger_name.startswith("test_") or logger_name.startswith("megatron.bridge.test"):
                del logging.root.manager.loggerDict[logger_name]

    def test_setup_logging_sets_root_level(self):
        """Test that setup_logging sets the root logger level."""
        setup_logging(logging_level=logging.DEBUG)
        assert logging.getLogger().level == logging.DEBUG

    def test_setup_logging_respects_env_var(self):
        """Test that MEGATRON_BRIDGE_LOGGING_LEVEL environment variable overrides the argument."""
        os.environ["MEGATRON_BRIDGE_LOGGING_LEVEL"] = str(logging.WARNING)

        setup_logging(logging_level=logging.DEBUG)

        # Should use env var value (WARNING), not argument value (DEBUG)
        assert logging.getLogger().level == logging.WARNING

    def test_setup_logging_sets_megatron_bridge_loggers(self):
        """Test that setup_logging sets level for megatron.bridge loggers by default."""
        # Create some test loggers
        megatron_logger = logging.getLogger("megatron.bridge.test_module")
        other_logger = logging.getLogger("some.other.module")
        nemo_logger = logging.getLogger("nemo.test_module")

        # Set different initial levels
        megatron_logger.setLevel(logging.WARNING)
        other_logger.setLevel(logging.WARNING)
        nemo_logger.setLevel(logging.WARNING)

        setup_logging(logging_level=logging.DEBUG, set_level_for_all_loggers=False)

        # megatron.bridge logger should be updated
        assert megatron_logger.level == logging.DEBUG

        # Other loggers should not be updated
        assert other_logger.level == logging.WARNING
        assert nemo_logger.level == logging.WARNING

    def test_setup_logging_sets_all_loggers_when_flag_true(self):
        """Test that setup_logging sets level for all loggers when set_level_for_all_loggers=True."""
        # Create some test loggers
        megatron_logger = logging.getLogger("megatron.bridge.test_module")
        other_logger = logging.getLogger("some.other.module")
        nemo_logger = logging.getLogger("nemo.test_module")

        # Set different initial levels
        megatron_logger.setLevel(logging.WARNING)
        other_logger.setLevel(logging.WARNING)
        nemo_logger.setLevel(logging.WARNING)

        setup_logging(logging_level=logging.DEBUG, set_level_for_all_loggers=True)

        # All loggers should be updated
        assert megatron_logger.level == logging.DEBUG
        assert other_logger.level == logging.DEBUG
        assert nemo_logger.level == logging.DEBUG

    def test_setup_logging_with_filter_warnings_true(self):
        """Test that setup_logging adds warning filter when filter_warning=True."""
        with patch("megatron.bridge.training.utils.log_utils.add_filter_to_all_loggers") as mock_add_filter:
            setup_logging(filter_warning=True)

            # Should call add_filter_to_all_loggers once for the warning filter
            assert mock_add_filter.call_count >= 1

            # Check that warning_filter was passed
            call_args = [call[0][0] for call in mock_add_filter.call_args_list]
            from megatron.bridge.training.utils.log_utils import warning_filter

            assert warning_filter in call_args

    def test_setup_logging_with_filter_warnings_false(self):
        """Test that setup_logging doesn't add warning filter when filter_warning=False."""
        with patch("megatron.bridge.training.utils.log_utils.add_filter_to_all_loggers") as mock_add_filter:
            setup_logging(filter_warning=False, modules_to_filter=None)

            # Should not call add_filter_to_all_loggers for warning filter
            mock_add_filter.assert_not_called()

    def test_setup_logging_with_modules_to_filter(self):
        """Test that setup_logging adds module filter when modules_to_filter is provided."""
        modules = ["test_module1", "test_module2"]

        with patch("megatron.bridge.training.utils.log_utils.add_filter_to_all_loggers") as mock_add_filter:
            setup_logging(filter_warning=False, modules_to_filter=modules)

            # Should call add_filter_to_all_loggers once for the module filter
            mock_add_filter.assert_called_once()

            # The filter should be a partial function
            filter_func = mock_add_filter.call_args[0][0]
            assert callable(filter_func)

    def test_setup_logging_default_values(self):
        """Test that setup_logging works with default parameter values."""
        # Should not raise any exceptions
        setup_logging()

        # Root logger should be set to INFO (default)
        assert logging.getLogger().level == logging.INFO

    def test_env_var_precedence(self):
        """Test that environment variable takes precedence over function argument."""
        # Set env var to ERROR level
        os.environ["MEGATRON_BRIDGE_LOGGING_LEVEL"] = str(logging.ERROR)

        # Call with different level
        setup_logging(logging_level=logging.DEBUG)

        # Should use env var value
        assert logging.getLogger().level == logging.ERROR

    def test_megatron_bridge_prefix_matching(self):
        """Test that only loggers with correct prefix are updated."""
        # Create loggers with various prefixes
        test_cases = [
            ("megatron.bridge.core", True),
            ("megatron.bridge.training.utils", True),
            ("megatron.bridge", True),
            ("megatron.core", False),
            ("megatron", False),
            ("nemo.core", False),
            ("other.module", False),
        ]

        loggers = {}
        for logger_name, _ in test_cases:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
            loggers[logger_name] = logger

        setup_logging(logging_level=logging.DEBUG, set_level_for_all_loggers=False)

        for logger_name, should_be_updated in test_cases:
            expected_level = logging.DEBUG if should_be_updated else logging.WARNING
            actual_level = loggers[logger_name].level
            assert actual_level == expected_level, (
                f"Logger '{logger_name}' should {'be' if should_be_updated else 'not be'} updated"
            )
