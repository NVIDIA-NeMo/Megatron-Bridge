# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
from datetime import datetime
from functools import partial
from logging import Filter, LogRecord
from typing import Callable

import torch

from megatron.bridge.utils.common_utils import get_rank_safe, get_world_size_safe, print_rank_0


def warning_filter(record: LogRecord) -> bool:
    """Filter out warning-level log records."""
    return record.levelno != logging.WARNING


def module_filter(record: LogRecord, modules_to_filter: list[str]) -> bool:
    """Filter out log records whose logger name starts with configured modules."""
    for module in modules_to_filter:
        if record.name.startswith(module):
            return False
    return True


def add_filter_to_all_loggers(log_filter: Filter | Callable[[LogRecord], bool]) -> None:
    """Add a filter to the root logger and all existing loggers.

    Args:
        log_filter: Logging filter instance or callable.
    """
    logging.getLogger().addFilter(log_filter)
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).addFilter(log_filter)


def setup_logging(
    logging_level: int | None = None,
    filter_warning: bool = True,
    modules_to_filter: list[str] | None = None,
    set_level_for_all_loggers: bool = False,
) -> None:
    """Set up logging level and filters for the application.

    This mirrors the former Megatron-LM logging helper and also honors the legacy
    Bridge env var ``MEGATRON_BRIDGE_LOGGING_LEVEL``.

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
    env_logging_level = os.getenv("MEGATRON_LOGGING_LEVEL")
    if bridge_env is not None and env_logging_level is None:
        os.environ["MEGATRON_LOGGING_LEVEL"] = bridge_env
        env_logging_level = bridge_env

    selected_level = logging.INFO
    if env_logging_level is not None:
        selected_level = int(env_logging_level)
    elif bridge_env is not None:
        selected_level = int(bridge_env)
    if logging_level is not None:
        selected_level = logging_level

    logging.getLogger().setLevel(selected_level)
    for logger_name in logging.root.manager.loggerDict:
        if set_level_for_all_loggers or logger_name.startswith("megatron.bridge"):
            logging.getLogger(logger_name).setLevel(selected_level)

    if filter_warning:
        add_filter_to_all_loggers(warning_filter)
    if modules_to_filter:
        add_filter_to_all_loggers(partial(module_filter, modules_to_filter=modules_to_filter))


def append_to_progress_log(save_dir: str, string: str, barrier: bool = True) -> None:
    """Append a formatted rank-0 message to ``progress.txt`` under ``save_dir``."""
    if save_dir is None:
        return

    progress_log_filename = os.path.join(save_dir, "progress.txt")
    if barrier and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if get_rank_safe() == 0:
        os.makedirs(os.path.dirname(progress_log_filename), exist_ok=True)
        with open(progress_log_filename, "a+") as progress_log:
            job_id = os.getenv("SLURM_JOB_ID", "")
            num_gpus = get_world_size_safe()
            progress_log.write(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\tJob ID: {job_id}\t# GPUs: {num_gpus}\t{string}\n"
            )


def barrier_and_log(string: str) -> None:
    """Synchronize initialized distributed workers and log a rank-0 timestamp."""
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_rank_0(f"[{string}] datetime: {time_str} ")


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
