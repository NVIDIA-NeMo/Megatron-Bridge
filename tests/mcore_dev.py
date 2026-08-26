# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Identify the unreleased MCore dev validation lane from immutable repository pins."""

import subprocess
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_pin(path: str) -> str:
    return (_REPO_ROOT / path).read_text().strip()


def _mcore_head() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(_REPO_ROOT / "3rdparty/Megatron-LM"), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


_MAIN_COMMIT = _read_pin(".main.commit")
_DEV_COMMIT = _read_pin(".dev.commit")
HAS_MCORE_DEV_BRANCH = _DEV_COMMIT != _MAIN_COMMIT and _mcore_head() == _DEV_COMMIT
