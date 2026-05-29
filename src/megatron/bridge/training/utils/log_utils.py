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

import os

from megatron.training.utils.log_utils import (
    add_filter_to_all_loggers,  # noqa: F401
    append_to_progress_log,  # noqa: F401
    barrier_and_log,  # noqa: F401
    module_filter,  # noqa: F401
    warning_filter,  # noqa: F401
)
from megatron.training.utils.log_utils import setup_logging as _mlm_setup_logging


def setup_logging(
    logging_level: int | None = None,
    filter_warning: bool = True,
    modules_to_filter: list[str] | None = None,
    set_level_for_all_loggers: bool = False,
) -> None:
    """Set up logging level and filters for the application.

    Thin wrapper around :func:`megatron.training.utils.log_utils.setup_logging`
    that also honors the legacy Bridge env var ``MEGATRON_BRIDGE_LOGGING_LEVEL``
    by promoting it to ``MEGATRON_LOGGING_LEVEL`` when the latter is unset.

    Logging Level Precedence (matches MLM):
    1. ``logging_level`` argument
    2. Env var ``MEGATRON_LOGGING_LEVEL`` (or legacy ``MEGATRON_BRIDGE_LOGGING_LEVEL``)
    3. Default: ``logging.INFO``

    Args:
        logging_level: The desired logging level (e.g., logging.INFO, logging.DEBUG).
        filter_warning: If True, adds a filter to suppress WARNING level messages.
        modules_to_filter: An optional list of module name prefixes to filter out.
        set_level_for_all_loggers: If True, sets the logging level for all existing
                                   loggers. If False (default), only sets the level
                                   for the root logger and loggers starting with 'megatron.bridge'.
    """
    bridge_env = os.getenv("MEGATRON_BRIDGE_LOGGING_LEVEL")
    if bridge_env is not None:
        os.environ.setdefault("MEGATRON_LOGGING_LEVEL", bridge_env)
    _mlm_setup_logging(
        logging_level=logging_level,
        filter_warning=filter_warning,
        modules_to_filter=modules_to_filter,
        set_level_for_all_loggers=set_level_for_all_loggers,
    )


def safe_serialize(obj) -> str:
    """Safely convert any object to a JSON-serializable type.

    Handles objects with broken __str__ or __repr__ methods that return
    non-string types (e.g., PipelineParallelLayerLayout returns list).
    """
    try:
        # Try str() first
        result = str(obj)
        # Verify it actually returns a string
        if not isinstance(result, str):
            # __str__ returned non-string type, use type name instead
            return f"<{type(obj).__name__}>"
        return result
    except Exception:
        # __str__ raised an exception, use type name as fallback
        return f"<{type(obj).__name__}>"
