# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, rank-local diagnostics for distributed training teardown."""

import faulthandler
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


_log_file: TextIO | None = None


def _output() -> TextIO | None:
    global _log_file
    directory = os.environ.get("MB_TEARDOWN_DIAG_DIR")
    if not directory:
        return None
    if _log_file is None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "unknown"))
        _log_file = (path / f"rank-{rank}-pid-{os.getpid()}.log").open("a", buffering=1)
        faulthandler.dump_traceback_later(45, repeat=True, file=_log_file)
    return _log_file


def mark_teardown(phase: str) -> None:
    """Record a timestamped phase boundary; start stack snapshots on first use."""
    output = _output()
    if output is not None:
        now = datetime.now(timezone.utc).isoformat()
        output.write(f"{now} monotonic={time.monotonic():.3f} phase={phase}\n")


@contextmanager
def teardown_phase(name: str):
    """Record both sides of a cleanup call without changing its behavior."""
    mark_teardown(f"{name}.begin")
    try:
        yield
    except BaseException as error:
        mark_teardown(f"{name}.error.{type(error).__name__}")
        raise
    else:
        mark_teardown(f"{name}.end")


def finish_teardown_diagnostics() -> None:
    """Stop rank-local snapshots once process-group destruction succeeds."""
    global _log_file
    if _log_file is not None:
        mark_teardown("teardown.complete")
        faulthandler.cancel_dump_traceback_later()
        _log_file.close()
        _log_file = None
