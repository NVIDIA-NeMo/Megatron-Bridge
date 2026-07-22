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

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[3] / "tools" / "qwen3_omni_fixed_score_corr.py"
SPEC = importlib.util.spec_from_file_location("qwen3_omni_fixed_score_corr", SCRIPT_PATH)
fixed_score_corr = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fixed_score_corr
SPEC.loader.exec_module(fixed_score_corr)


def _write_payload(path, key, values, ok=True):
    rows = [{"pos": index, key: value} for index, value in enumerate(values)]
    path.write_text(json.dumps({"ok": ok, "rows": rows}), encoding="utf-8")


def test_corr_gate_passes_aligned_scores(tmp_path):
    actor = tmp_path / "actor.json"
    hf = tmp_path / "hf.json"
    _write_payload(actor, "actor_log_prob", [-1.0, -2.0, -3.0, -4.0])
    _write_payload(hf, "hf_log_prob", [-1.01, -2.01, -3.01, -4.01])

    summary = fixed_score_corr.summarize(str(actor), str(hf))

    assert summary.num_common == 4
    assert summary.pearson > 0.999
    assert fixed_score_corr.main(["--actor", str(actor), "--hf", str(hf)]) == 0


def test_corr_gate_fails_low_pearson(tmp_path):
    actor = tmp_path / "actor.json"
    hf = tmp_path / "hf.json"
    _write_payload(actor, "actor_log_prob", [-1.0, -2.0, -3.0, -4.0])
    _write_payload(hf, "hf_log_prob", [-4.0, -3.0, -2.0, -1.0])

    assert fixed_score_corr.main(["--actor", str(actor), "--hf", str(hf), "--min-pearson", "0.99"]) == 1


def test_corr_gate_errors_on_failed_payload(tmp_path):
    actor = tmp_path / "actor.json"
    hf = tmp_path / "hf.json"
    _write_payload(actor, "actor_log_prob", [-1.0, -2.0], ok=False)
    _write_payload(hf, "hf_log_prob", [-1.0, -2.0])

    assert fixed_score_corr.main(["--actor", str(actor), "--hf", str(hf)]) == 2


def test_corr_gate_glob_uses_latest_match(tmp_path):
    older = tmp_path / "actor_older.json"
    newer = tmp_path / "actor_newer.json"
    hf = tmp_path / "hf.json"
    _write_payload(older, "actor_log_prob", [-4.0, -3.0, -2.0, -1.0])
    _write_payload(newer, "actor_log_prob", [-1.0, -2.0, -3.0, -4.0])
    _write_payload(hf, "hf_log_prob", [-1.0, -2.0, -3.0, -4.0])

    old_time = time.time() - 10.0
    os.utime(older, (old_time, old_time))
    os.utime(newer, None)

    summary = fixed_score_corr.summarize(str(tmp_path / "actor_*.json"), str(hf))

    assert summary.actor_path == str(newer)
    assert summary.pearson == pytest.approx(1.0)
