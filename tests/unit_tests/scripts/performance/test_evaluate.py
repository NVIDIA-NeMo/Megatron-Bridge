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

"""Tests for scripts/performance/utils/evaluate.py golden-value handling."""

import json
import sys
from pathlib import Path

import pytest


_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

from utils import evaluate  # noqa: E402
from utils.evaluate import _unwrap_golden_values, downsample_golden_values  # noqa: E402


def _make_values(n_steps: int) -> dict:
    """Build a golden-values mapping with n_steps per-step entries plus scalar keys."""
    values = {str(i): {"lm loss": float(i), "GPU utilization": 1.0} for i in range(n_steps)}
    values.update({"alloc": 5.95, "max_alloc": 19.3, "max_reserved": 19.5, "job_id": 123})
    return values


def _step_keys(values: dict) -> list:
    return sorted((int(k) for k in values if k.lstrip("-").isdigit()))


def test_unwrap_golden_values_supports_flat_and_snapshot_formats():
    step_values = {"0": {"lm loss": 1.0}, "alloc": 2.0}

    assert _unwrap_golden_values(step_values) is step_values
    assert _unwrap_golden_values({"baseline": step_values}) is step_values
    assert _unwrap_golden_values({"current": step_values}) is step_values


def test_unwrap_golden_values_prefers_baseline_snapshot():
    baseline = {"0": {"lm loss": 1.0}}
    current = {"0": {"lm loss": 2.0}}

    assert _unwrap_golden_values({"current": current, "baseline": baseline}) is baseline


def test_calc_convergence_supports_current_snapshot_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(
        json.dumps(
            {
                "current": {
                    "0": {
                        "lm loss": 1.0,
                        "elapsed time per iteration (ms)": 2.0,
                        "GPU utilization": 3.0,
                    },
                    "alloc": 4.0,
                    "max_alloc": 5.0,
                    "max_reserved": 6.0,
                }
            }
        )
    )
    metrics = {
        "lm loss": {"0": 1.0},
        "elapsed time per iteration (ms)": {"0": 2.0},
        "GPU utilization": {"0": 3.0},
        "grad norm": {"0": 0.5},
        "alloc": 4.0,
        "max_alloc": 5.0,
        "max_reserved": 6.0,
    }
    monkeypatch.setattr(evaluate, "get_metrics_from_logfiles", lambda _paths, metric: metrics[metric])
    monkeypatch.setattr(evaluate, "write_golden_values_to_disk", lambda **_kwargs: None)
    monkeypatch.setattr(
        evaluate,
        "validate_convergence",
        lambda **_kwargs: {"passed": True, "failed_metrics": [], "summary": ""},
    )
    monkeypatch.setattr(
        evaluate,
        "validate_performance",
        lambda **_kwargs: {"passed": True, "failed_metrics": [], "summary": "", "metrics": {}},
    )
    monkeypatch.setattr(
        evaluate,
        "validate_memory",
        lambda **_kwargs: {"passed": True, "failed_metrics": [], "summary": "", "metrics": {}},
    )

    passed, error_message, _current_values = evaluate.calc_convergence_and_performance(
        model_family_name="llama",
        model_recipe_name="llama3_8b",
        assets_dir=str(tmp_path / "assets"),
        log_paths=[],
        loss_metric="lm loss",
        timing_metric="elapsed time per iteration (ms)",
        alloc_metric="alloc",
        max_alloc_metric="max_alloc",
        max_reserved_metric="max_reserved",
        golden_values_path=str(golden_path),
        convergence_config={},
        performance_config={"eval_time_start_step": 0, "eval_time_end_step": 1},
        memory_config={"memory_threshold": 0.1},
    )

    assert passed is True
    assert error_message == ""


def test_downsample_noop_when_under_cap():
    values = _make_values(100)
    result = downsample_golden_values(values, max_steps=10000)
    assert result == values
    # A fresh mapping is returned (defensive copy), not the same object.
    assert result is not values


def test_downsample_caps_step_count():
    values = _make_values(150000)
    result = downsample_golden_values(values, max_steps=10000)
    step_keys = _step_keys(result)
    # Evenly strided subset plus a couple of pinned/final steps; comfortably bounded.
    assert 10000 <= len(step_keys) <= 10010


def test_downsample_preserves_scalars_and_endpoints():
    values = _make_values(30000)
    result = downsample_golden_values(values, max_steps=10000)
    # Scalar metadata is untouched.
    for key in ("alloc", "max_alloc", "max_reserved", "job_id"):
        assert result[key] == values[key]
    # First step, pinned threshold step (49), and the final step survive.
    assert "0" in result
    assert "49" in result
    assert "29999" in result
    # Every retained step is a real step from the input (no fabricated keys).
    assert set(_step_keys(result)).issubset(set(_step_keys(values)))


def test_downsample_does_not_mutate_input():
    values = _make_values(30000)
    before = len(values)
    downsample_golden_values(values, max_steps=10000)
    assert len(values) == before
