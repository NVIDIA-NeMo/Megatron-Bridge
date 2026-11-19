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

"""CPU-only checks for the overrides examples/models/qwen/qwen2_audio/sft.sh passes to run_recipe.py."""

import os
import re
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "examples/models/qwen/qwen2_audio/sft.sh"


@pytest.fixture
def overrides(tmp_path):
    """Run sft.sh with a stub uv and an existing checkpoint; return its key=value overrides."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$UV_LOG"\n')
    uv.chmod(0o755)
    (tmp_path / "megatron_ckpts" / "qwen2_audio_7b" / "iter_0000000").mkdir(parents=True)

    env = {k: v for k, v in os.environ.items() if k not in ("NPROC", "HF_MODEL", "PRETRAINED_CHECKPOINT")}
    env.update({"PATH": f"{fake_bin}:{env['PATH']}", "UV_LOG": str(tmp_path / "uv.log"), "WORKSPACE": str(tmp_path)})
    result = subprocess.run(["bash", str(SCRIPT)], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

    args = (tmp_path / "uv.log").read_text().splitlines()
    return dict(arg.split("=", 1) for arg in args if re.match(r"^[\w.]+=", arg))


def _script_default(name: str) -> int:
    return int(re.search(rf"^{name}=(?:\${{{name}:-)?(\d+)\}}?$", SCRIPT.read_text(), re.M).group(1))


class TestQwen2AudioSftOverrides:
    """sft.sh overrides that decide the effective training shape."""

    def test_sequence_length_reaches_dataset(self, overrides):
        """The runner takes the sequence length from the dataset, so SEQ_LENGTH must be passed there."""
        assert overrides.get("dataset.seq_length") == str(_script_default("SEQ_LENGTH"))
        assert overrides.get("model.seq_length", overrides["dataset.seq_length"]) == overrides["dataset.seq_length"]

    def test_batch_sizes_are_valid_for_in_batch_packing(self, overrides):
        """In-batch packing needs at least two samples per micro-batch and GBS divisible by MBS x DP."""
        mbs = int(overrides["train.micro_batch_size"])
        gbs = int(overrides["train.global_batch_size"])
        assert overrides["dataset.enable_in_batch_packing"] == "true"
        assert mbs >= 2
        assert gbs % (mbs * _script_default("NPROC")) == 0
