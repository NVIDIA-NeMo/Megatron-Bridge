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

import os
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_convert_sh_uses_nemo_run_from_active_environment(tmp_path, monkeypatch):
    uv = tmp_path / "uv"
    uv.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("VIRTUAL_ENV", "/opt/venv")
    monkeypatch.setenv("UV_NO_SYNC", "1")

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/conversion/convert.sh"), "roundtrip", "--executor", "local"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        "run",
        "--no-project",
        "--active",
        "python",
        str(REPO_ROOT / "scripts/conversion/setup_conversion.py"),
        "roundtrip",
        "--executor",
        "local",
    ]
