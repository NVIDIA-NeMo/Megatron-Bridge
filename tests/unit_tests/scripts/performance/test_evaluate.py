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

"""Tests for scripts/performance/utils/evaluate.py golden-value downsampling."""

import sys
from pathlib import Path


_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

from utils.evaluate import downsample_golden_values  # noqa: E402


def _make_values(n_steps: int) -> dict:
    """Build a golden-values mapping with n_steps per-step entries plus scalar keys."""
    values = {str(i): {"lm loss": float(i), "GPU utilization": 1.0} for i in range(n_steps)}
    values.update({"alloc": 5.95, "max_alloc": 19.3, "max_reserved": 19.5, "job_id": 123})
    return values


def _step_keys(values: dict) -> list:
    return sorted((int(k) for k in values if k.lstrip("-").isdigit()))


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
