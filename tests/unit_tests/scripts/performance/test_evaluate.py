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

import json
import logging

import pytest
from scripts.performance.utils import evaluate


@pytest.mark.unit
def test_get_metrics_from_logfiles_tracks_max_reserved_across_all_occurrences(tmp_path):
    log_file = tmp_path / "log.out"
    log_file.write_text(
        """
p4_v2_perf/0 [Rank 1] (after 2 iterations) memory (GB) | mem-allocated-gigabytes: 63.66 | mem-active-gigabytes: 63.66 | mem-inactive-gigabytes: 20.563 | mem-reserved-gigabytes: 182.2 | mem-max-allocated-gigabytes: 153.4 | mem-max-active-gigabytes: 169.96 | mem-max-inactive-gigabytes: 32.473 | mem-max-reserved-gigabytes: 182.2 | mem-alloc-retires: 0 | mem-allocated-count: 507
p4_v2_perf/0 [Rank 2] (after 2 iterations) memory (GB) | mem-allocated-gigabytes: 61.881 | mem-active-gigabytes: 61.881 | mem-inactive-gigabytes: 20.369 | mem-reserved-gigabytes: 182.24 | mem-max-allocated-gigabytes: 153.47 | mem-max-active-gigabytes: 169.75 | mem-max-inactive-gigabytes: 31.879 | mem-max-reserved-gigabytes: 182.24 | mem-alloc-retires: 0 | mem-allocated-count: 433[Rank 3] (after 2 iterations) memory (GB) | mem-allocated-gigabytes: 63.018 | mem-active-gigabytes: 63.018 | mem-inactive-gigabytes: 19.358 | mem-reserved-gigabytes: 181.59 | mem-max-allocated-gigabytes: 153.39 | mem-max-active-gigabytes: 169.94 | mem-max-inactive-gigabytes: 31.955 | mem-max-reserved-gigabytes: 181.59 | mem-alloc-retires: 0 | mem-allocated-count: 481
p4_v2_perf/0 [Rank 0] (after 2 iterations) memory (GB) | mem-allocated-gigabytes: 63.4 | mem-active-gigabytes: 63.4 | mem-inactive-gigabytes: 20.444 | mem-reserved-gigabytes: 182.16 | mem-max-allocated-gigabytes: 153.48 | mem-max-active-gigabytes: 169.89 | mem-max-inactive-gigabytes: 32.481 | mem-max-reserved-gigabytes: 182.16 | mem-alloc-retires: 0 | mem-allocated-count: 497
""",
    )

    assert evaluate.get_metrics_from_logfiles([str(log_file)], "max_reserved") == pytest.approx(182.24)


@pytest.mark.unit
def test_validate_memory_compares_max_reserved():
    result = evaluate.validate_memory(
        golden_alloc=10.0,
        current_alloc=10.0,
        golden_max_alloc=20.0,
        current_max_alloc=20.0,
        logger=logging.getLogger(__name__),
        golden_max_reserved=100.0,
        current_max_reserved=120.0,
        config={"memory_threshold": 0.05},
    )

    assert result["passed"] is False
    assert result["failed_metrics"] == ["max_reserved"]
    assert result["metrics"]["max_reserved_diff"] == pytest.approx(0.2)


@pytest.mark.unit
def test_calc_writes_max_reserved_and_reports_missing_golden_metric(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    monkeypatch.setattr(evaluate, "HAVE_NUMPY", True)
    monkeypatch.setattr(evaluate, "HAVE_WANDB", True)

    log_file = tmp_path / "log.out"
    log_file.write_text(
        """
[2026-05-06 00:00:01] iteration        1/       2 | consumed samples: 1 | elapsed time per iteration (ms): 100.0 | lm loss: 1.000000E+00 | grad norm: 1.0 | GPU utilization: 50.0MODEL_TFLOP/s/GPU[Rank 0] (after 1 iterations) memory (GB) | mem-allocated-gigabytes: 10.0 | mem-max-allocated-gigabytes: 20.0 | mem-max-reserved-gigabytes: 30.0 |
[2026-05-06 00:00:02] iteration        2/       2 | consumed samples: 2 | elapsed time per iteration (ms): 90.0 | lm loss: 9.000000E-01 | grad norm: 1.1 | GPU utilization: 55.0MODEL_TFLOP/s/GPU[Rank 0] (after 2 iterations) memory (GB) | mem-allocated-gigabytes: 11.0 | mem-max-allocated-gigabytes: 21.0 | mem-max-reserved-gigabytes: 31.5 |
[Rank 1] (after 2 iterations) memory (GB) | mem-allocated-gigabytes: 12.0 | mem-max-allocated-gigabytes: 22.0 | mem-max-reserved-gigabytes: 31.0 |
""",
    )

    golden_values_path = tmp_path / "golden_values" / "values.json"
    golden_values_path.parent.mkdir()
    golden_values_path.write_text(
        json.dumps(
            {
                "0": {
                    "lm loss": 1.0,
                    "elapsed time per iteration (ms)": 100.0,
                    "GPU utilization": 50.0,
                },
                "1": {
                    "lm loss": 0.9,
                    "elapsed time per iteration (ms)": 90.0,
                    "GPU utilization": 55.0,
                },
                "alloc": 10.0,
                "max_alloc": 20.0,
            }
        )
    )

    passed, error_msg = evaluate.calc_convergence_and_performance(
        model_family_name="llama",
        model_recipe_name="llama3_8b",
        assets_dir=str(tmp_path / "assets"),
        log_paths=[str(log_file)],
        loss_metric="lm loss",
        timing_metric="elapsed time per iteration (ms)",
        alloc_metric="alloc",
        max_alloc_metric="max_alloc",
        golden_values_path=str(golden_values_path),
        convergence_config={
            "correlation_threshold": 0.95,
            "high_loss_tolerance": 0.1,
            "medium_loss_tolerance": 0.1,
            "low_loss_tolerance": 0.1,
            "final_loss_tolerance": 0.1,
            "max_outlier_ratio": 1.0,
        },
        performance_config={"timing_threshold": 0.05, "skip_first_percent_time": 0.0},
        memory_config={"memory_threshold": 0.05},
    )

    actual_values_path = tmp_path / "assets" / "golden_values" / "values.json"
    actual_values = json.loads(actual_values_path.read_text())

    assert passed is True
    assert '"max_reserved": 31.5' in error_msg
    assert actual_values["max_reserved"] == pytest.approx(31.5)


@pytest.mark.unit
def test_calc_fails_when_current_max_reserved_metric_is_missing(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    monkeypatch.setattr(evaluate, "HAVE_NUMPY", True)
    monkeypatch.setattr(evaluate, "HAVE_WANDB", True)

    log_file = tmp_path / "log.out"
    log_file.write_text(
        """
[2026-05-06 00:00:01] iteration        1/       2 | elapsed time per iteration (ms): 100.0 | lm loss: 1.000000E+00 | grad norm: 1.0 | GPU utilization: 50.0MODEL_TFLOP/s/GPU[Rank 0] memory | mem-allocated-gigabytes: 10.0 | mem-max-allocated-gigabytes: 20.0 |
[2026-05-06 00:00:02] iteration        2/       2 | elapsed time per iteration (ms): 90.0 | lm loss: 9.000000E-01 | grad norm: 1.1 | GPU utilization: 55.0MODEL_TFLOP/s/GPU[Rank 0] memory | mem-allocated-gigabytes: 10.0 | mem-max-allocated-gigabytes: 20.0 |
""",
    )

    golden_values_path = tmp_path / "golden_values" / "values.json"
    golden_values_path.parent.mkdir()
    golden_values_path.write_text(
        json.dumps(
            {
                "0": {
                    "lm loss": 1.0,
                    "elapsed time per iteration (ms)": 100.0,
                    "GPU utilization": 50.0,
                },
                "1": {
                    "lm loss": 0.9,
                    "elapsed time per iteration (ms)": 90.0,
                    "GPU utilization": 55.0,
                },
                "alloc": 10.0,
                "max_alloc": 20.0,
                "max_reserved": 30.0,
            }
        )
    )

    passed, error_msg = evaluate.calc_convergence_and_performance(
        model_family_name="llama",
        model_recipe_name="llama3_8b",
        assets_dir=str(tmp_path / "assets"),
        log_paths=[str(log_file)],
        loss_metric="lm loss",
        timing_metric="elapsed time per iteration (ms)",
        alloc_metric="alloc",
        max_alloc_metric="max_alloc",
        golden_values_path=str(golden_values_path),
        convergence_config={
            "correlation_threshold": 0.95,
            "high_loss_tolerance": 0.1,
            "medium_loss_tolerance": 0.1,
            "low_loss_tolerance": 0.1,
            "final_loss_tolerance": 0.1,
            "max_outlier_ratio": 1.0,
        },
        performance_config={"timing_threshold": 0.05, "skip_first_percent_time": 0.0},
        memory_config={"memory_threshold": 0.05},
    )

    assert passed is False
    assert "Memory check failed." in error_msg
    assert "Max reserved metric missing from current logs." in error_msg


@pytest.mark.unit
def test_calc_still_validates_existing_memory_when_golden_max_reserved_is_missing(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    monkeypatch.setattr(evaluate, "HAVE_NUMPY", True)
    monkeypatch.setattr(evaluate, "HAVE_WANDB", True)

    log_file = tmp_path / "log.out"
    log_file.write_text(
        """
[2026-05-06 00:00:01] iteration        1/       2 | elapsed time per iteration (ms): 100.0 | lm loss: 1.000000E+00 | grad norm: 1.0 | GPU utilization: 50.0MODEL_TFLOP/s/GPU[Rank 0] memory | mem-allocated-gigabytes: 12.0 | mem-max-allocated-gigabytes: 20.0 | mem-max-reserved-gigabytes: 30.0 |
[2026-05-06 00:00:02] iteration        2/       2 | elapsed time per iteration (ms): 90.0 | lm loss: 9.000000E-01 | grad norm: 1.1 | GPU utilization: 55.0MODEL_TFLOP/s/GPU[Rank 0] memory | mem-allocated-gigabytes: 12.0 | mem-max-allocated-gigabytes: 20.0 | mem-max-reserved-gigabytes: 31.0 |
""",
    )

    golden_values_path = tmp_path / "golden_values" / "values.json"
    golden_values_path.parent.mkdir()
    golden_values_path.write_text(
        json.dumps(
            {
                "0": {
                    "lm loss": 1.0,
                    "elapsed time per iteration (ms)": 100.0,
                    "GPU utilization": 50.0,
                },
                "1": {
                    "lm loss": 0.9,
                    "elapsed time per iteration (ms)": 90.0,
                    "GPU utilization": 55.0,
                },
                "alloc": 10.0,
                "max_alloc": 20.0,
            }
        )
    )

    passed, error_msg = evaluate.calc_convergence_and_performance(
        model_family_name="llama",
        model_recipe_name="llama3_8b",
        assets_dir=str(tmp_path / "assets"),
        log_paths=[str(log_file)],
        loss_metric="lm loss",
        timing_metric="elapsed time per iteration (ms)",
        alloc_metric="alloc",
        max_alloc_metric="max_alloc",
        golden_values_path=str(golden_values_path),
        convergence_config={
            "correlation_threshold": 0.95,
            "high_loss_tolerance": 0.1,
            "medium_loss_tolerance": 0.1,
            "low_loss_tolerance": 0.1,
            "final_loss_tolerance": 0.1,
            "max_outlier_ratio": 1.0,
        },
        performance_config={"timing_threshold": 0.05, "skip_first_percent_time": 0.0},
        memory_config={"memory_threshold": 0.05},
    )

    actual_values_path = tmp_path / "assets" / "golden_values" / "values.json"
    actual_values = json.loads(actual_values_path.read_text())

    assert passed is False
    assert "Memory check failed." in error_msg
    assert "Alloc difference: 20.00%" in error_msg
    assert "Memory metric (max_reserved) is also missing from golden values" in error_msg
    assert actual_values["max_reserved"] == pytest.approx(31.0)
