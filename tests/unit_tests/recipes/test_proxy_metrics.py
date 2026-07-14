# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tests.functional_tests.test_groups.recipes.proxy_metrics import summarize_metrics, validate_metrics


def test_summarize_metrics_skips_warmup_and_uses_final_loss(tmp_path):
    log_path = tmp_path / "proxy.log"
    log_path.write_text(
        "\n".join(
            [
                "iteration 1/ 4 | elapsed time per iteration (ms): 900.0 | lm loss: 9.0",
                "iteration 2/ 4 | elapsed time per iteration (ms): 120.0 | lm loss: 8.0",
                "iteration 3/ 4 | elapsed time per iteration (ms): 100.0 | lm loss: 7.0",
                "iteration 4/ 4 | elapsed time per iteration (ms): 110.0 | lm loss: 6.0",
            ]
        ),
        encoding="utf-8",
    )

    actual = summarize_metrics([str(log_path)], warmup_steps=1)

    assert actual == {"warmup_steps": 1, "mean_step_time_ms": 110.0, "final_loss": 6.0}


def test_validate_metrics_allows_speedup_and_loss_within_tolerance():
    golden = {
        "mean_step_time_ms": 100.0,
        "step_time_regression_tolerance": 0.2,
        "final_loss": 5.0,
        "final_loss_relative_tolerance": 0.05,
    }

    assert validate_metrics({"mean_step_time_ms": 80.0, "final_loss": 5.2}, golden) == []


def test_validate_metrics_reports_timing_and_loss_regressions():
    golden = {
        "mean_step_time_ms": 100.0,
        "step_time_regression_tolerance": 0.1,
        "final_loss": 5.0,
        "final_loss_relative_tolerance": 0.05,
    }

    failures = validate_metrics({"mean_step_time_ms": 120.0, "final_loss": 5.5}, golden)

    assert len(failures) == 2
    assert failures[0].startswith("mean step time regressed")
    assert failures[1].startswith("final loss changed")


def test_validate_metrics_supports_timing_only_golden():
    golden = {
        "mean_step_time_ms": 100.0,
        "step_time_regression_tolerance": 0.1,
        "final_loss": None,
    }

    assert validate_metrics({"mean_step_time_ms": 105.0, "final_loss": 999.0}, golden) == []
