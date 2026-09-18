# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Opt-in, per-rank Python stack dumps for stalled performance runs."""

import faulthandler
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


logger = logging.getLogger(__name__)


@contextmanager
def periodic_debug_tracebacks() -> Iterator[None]:
    """Dump all Python threads periodically when explicitly enabled.

    Set MBRIDGE_DEBUG_TRACEBACK_SECONDS to a positive integer and
    MBRIDGE_DEBUG_TRACEBACK_DIR to an artifact directory. Each global rank
    writes its own file. This uses faulthandler's watchdog rather than a
    Python thread so a stalled extension can still produce Python stacks.
    It does not synchronize CUDA or change the model configuration.

    Raises:
        ValueError: If enabled without a valid interval, directory, or rank.
    """
    interval = os.environ.get("MBRIDGE_DEBUG_TRACEBACK_SECONDS")
    if not interval:
        yield
        return

    seconds = int(interval)
    if seconds <= 0:
        raise ValueError("MBRIDGE_DEBUG_TRACEBACK_SECONDS must be a positive integer")
    directory = os.environ.get("MBRIDGE_DEBUG_TRACEBACK_DIR")
    if not directory:
        raise ValueError("MBRIDGE_DEBUG_TRACEBACK_DIR is required when stack dumps are enabled")
    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID"))
    if rank is None or not rank.isdecimal():
        raise ValueError("Stack dumps require a nonnegative RANK or SLURM_PROCID")

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"python_tracebacks_rank-{int(rank)}.log"
    # Keep the descriptor open until after the watchdog is canceled. Append
    # preserves earlier dumps if a launcher restarts the same rank.
    with path.open("a", encoding="utf-8") as output:
        logger.warning("Writing Python stacks every %s seconds to %s", seconds, path)
        faulthandler.dump_traceback_later(seconds, repeat=True, file=output)
        try:
            yield
        finally:
            faulthandler.cancel_dump_traceback_later()
