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

"""Log parsing and throughput arithmetic of the static-versus-dynamic comparison tool."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

_SOURCE = Path(__file__).resolve().parents[3] / "examples/agentic_sft/compare_runs.py"
_SPEC = importlib.util.spec_from_file_location("agentic_sft_compare_runs_under_test", _SOURCE)
compare = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = compare
_SPEC.loader.exec_module(compare)


def _line(iteration: int, milliseconds: float, loss: str = "1.250000E+01") -> str:
    return (
        f"[2026-10-27 13:00:00] iteration {iteration:8d}/{100:8d} | consumed samples: {iteration * 32:12d} | "
        f"elapsed time per iteration (ms): {milliseconds:.1f} | learning rate: 1.0E-05 | global batch size: 32 | "
        f"lm loss: {loss} | mtp_1 loss: 1.2E+01 |"
    )


def _log(times_ms: list[float]) -> str:
    return "noise before\n" + "\n".join(_line(i + 1, ms) for i, ms in enumerate(times_ms)) + "\nnoise after\n"


def test_parse_log_reads_every_iteration_and_ignores_other_lines():
    steps = compare.parse_log(_log([300000.0, 60000.0, 61000.5]))

    assert sorted(steps) == [1, 2, 3]
    assert steps[3] == {"seconds": 61.0005, "loss": 12.5}


def test_parse_log_rejects_logs_without_iterations():
    with pytest.raises(ValueError, match="No iteration lines"):
        compare.parse_log("nothing to see\n")


def test_summary_excludes_the_warmup_compile_step():
    steps = compare.parse_log(_log([300000.0, 100000.0, 60000.0, 70000.0]))

    summary = compare.summarize(steps, warmup=2)

    assert (summary["first_step"], summary["last_step"], summary["steps"]) == (3, 4, 2)
    assert summary["mean_s"] == pytest.approx(65.0)
    assert summary["total_s"] == pytest.approx(130.0)
    with pytest.raises(ValueError, match="No iterations after warmup"):
        compare.summarize(steps, warmup=4)


def test_tokens_per_step_take_rows_in_file_order():
    lengths = list(range(1, 9))

    assert compare.tokens_per_step(lengths, 4, [1, 2]) == {1: 1 + 2 + 3 + 4, 2: 5 + 6 + 7 + 8}


def test_relative_gain_is_throughput_on_identical_work():
    baseline = {"total_s": 120.0}
    assert compare.relative_gain(baseline, {"total_s": 100.0}) == pytest.approx(0.20)
    assert compare.relative_gain(baseline, {"total_s": 150.0}) == pytest.approx(-0.20)


def test_main_reports_seconds_tokens_and_the_relative_change(tmp_path, caplog):
    baseline, candidate = tmp_path / "static.log", tmp_path / "dynamic.log"
    baseline.write_text(_log([500000.0, 100000.0, 100000.0]))
    candidate.write_text(_log([480000.0, 80000.0, 80000.0]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"splits": {"training": {"runtime_lengths": [1000] * 96}}}))

    with caplog.at_level("INFO"):
        compare.main(
            [
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--warmup",
                "1",
                "--manifest",
                str(manifest),
                "--global-batch-size",
                "32",
            ]
        )

    text = caplog.text
    assert "100.00 s/step mean" in text and "80.00 s/step mean" in text
    # Measured steps 2-3 process 64 rows of 1000 tokens: 64,000 tokens over 200 s versus 160 s.
    assert "320 runtime tokens/s" in text and "400 runtime tokens/s" in text
    assert "+25.00% throughput" in text
