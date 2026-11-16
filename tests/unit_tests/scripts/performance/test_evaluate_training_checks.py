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

"""Tests for the parameter-update, expert-balance and resume checks in scripts/performance/utils/evaluate.py."""

import json
import sys
from pathlib import Path

import pytest


_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

from utils import evaluate  # noqa: E402


pytestmark = pytest.mark.unit


def _record(trainable=True, optimizer=True, gradient=True, changed=True):
    return {"trainable": trainable, "numel": 4, "optimizer": optimizer, "gradient": gradient, "changed": changed}


def _parameter_report(tmp_path, tensors):
    path = tmp_path / "parameter_updates.json"
    path.write_text(json.dumps({"version": 1, "tensors": tensors}))
    return str(path)


def test_parameter_updates_pass_when_every_trainable_tensor_updated_and_frozen_ones_are_declared(tmp_path):
    report = _parameter_report(
        tmp_path,
        {
            "decoder.layers.0.mlp.linear_fc1.weight": _record(),
            "vision_model.patch_embed.weight": _record(trainable=False, optimizer=False, gradient=None, changed=False),
        },
    )
    result = evaluate.validate_parameter_updates(report, frozen_pattern=r"^vision_model\.")
    assert result["passed"], result["details"]
    assert result["metrics"]["frozen_tensors"] == 1


@pytest.mark.parametrize(
    "record,problem",
    [
        (_record(optimizer=False, gradient=None, changed=None), "not held by the optimizer"),
        (_record(gradient=False), "never received a gradient"),
        (_record(changed=False), "master weights unchanged"),
    ],
)
def test_parameter_updates_fail_for_a_trainable_tensor_that_did_not_update(tmp_path, record, problem):
    report = _parameter_report(
        tmp_path, {"decoder.layers.3.self_attention.k_layernorm.weight": record, "output_layer.weight": _record()}
    )
    result = evaluate.validate_parameter_updates(report)
    assert not result["passed"]
    assert f"decoder.layers.3.self_attention.k_layernorm.weight: {problem}" in result["details"]
    assert evaluate.validate_parameter_updates(report, without_updates_pattern=r"k_layernorm")["passed"]


def test_parameter_updates_fail_when_the_frozen_set_differs_from_the_declared_one(tmp_path):
    undeclared = _parameter_report(
        tmp_path,
        {"w": _record(), "vision_model.weight": _record(trainable=False, optimizer=False, changed=False)},
    )
    assert "does not declare it frozen" in evaluate.validate_parameter_updates(undeclared)["details"]

    trainable = _parameter_report(tmp_path, {"w": _record(), "vision_model.weight": _record()})
    result = evaluate.validate_parameter_updates(trainable, frozen_pattern=r"^vision_model\.")
    assert "declared frozen, but trainable" in result["details"]

    changed = _parameter_report(
        tmp_path, {"w": _record(), "vision_model.weight": _record(trainable=False, optimizer=False, changed=True)}
    )
    result = evaluate.validate_parameter_updates(changed, frozen_pattern=r"^vision_model\.")
    assert "frozen, but its values changed" in result["details"]


def _expert_report(tmp_path, steps, total_tokens=((4, 4, 4, 4),)):
    path = tmp_path / "expert_load.json"
    layers = [f"decoder.layers.{index}.mlp" for index in range(len(steps[0]["max_load"]))]
    path.write_text(
        json.dumps({"version": 1, "num_experts": 4, "layers": layers, "steps": steps, "total_tokens": total_tokens})
    )
    return str(path)


def _step(iteration, max_load, idle=None):
    return {"iteration": iteration, "max_load": max_load, "min_load": [0.5] * len(max_load), "idle": idle or [0]}


def test_expert_balance_reports_without_limits_and_enforces_the_declared_ones(tmp_path):
    report = _expert_report(tmp_path, [_step(1, [2.0]), _step(2, [3.5]), _step(3, [2.5])])

    observed = evaluate.validate_expert_balance(report)
    assert observed["passed"]
    assert observed["metrics"]["worst_max_load_ratio"] == 3.5
    assert observed["metrics"]["worst_max_load_iteration"] == 2
    assert observed["metrics"]["mean_max_load_ratio"] == pytest.approx(8.0 / 3)

    assert evaluate.validate_expert_balance(report, max_load_ratio=4.0, min_mean_max_load_ratio=1.05)["passed"]
    too_high = evaluate.validate_expert_balance(report, max_load_ratio=3.0)
    assert "iteration 2: max/mean load 3.500" in too_high["details"]


def test_expert_balance_fails_forced_balancing_idle_and_dead_experts(tmp_path):
    forced = _expert_report(tmp_path, [_step(1, [1.01]), _step(2, [1.02])])
    assert (
        "run mean max/mean load 1.015 < limit 1.05"
        in (evaluate.validate_expert_balance(forced, min_mean_max_load_ratio=1.05)["details"])
    )

    idle = _expert_report(tmp_path, [_step(1, [2.0], idle=[3])], total_tokens=[[0, 4, 4, 4]])
    result = evaluate.validate_expert_balance(idle, max_idle_experts_per_step=0, max_experts_without_tokens_in_run=0)
    assert "3 experts received no tokens > limit 0" in result["details"]
    assert "1 experts received no tokens in the whole run" in result["details"]

    empty_layer = _expert_report(tmp_path, [_step(1, [None, 2.0])])
    assert "no tokens reached decoder.layers.0.mlp" in evaluate.validate_expert_balance(empty_layer)["details"]


def _training_log(path, iterations, *, loss=lambda i: 2.0 - i / 1000, lr=lambda i: 1e-4, grad=lambda i: 1.0):
    lines = [
        f" [2026-11-13 10:00:00] iteration {i:8d}/     100 | consumed samples: {i * 256:12d} | "
        f"elapsed time per iteration (ms): 3400.0 | learning rate: {lr(i):.6E} | global batch size:   256 | "
        f"lm loss: {loss(i):.6E} | loss scale: 1.0 | grad norm: {grad(i):.3f} | number of skipped iterations:   0 |"
        for i in iterations
    ]
    path.write_text("\n".join(lines) + "\n")
    return [str(path)]


def test_metrics_parser_reads_learning_rate_and_consumed_samples(tmp_path):
    log = _training_log(tmp_path / "log-allranks.out", [1, 2])
    assert evaluate.get_metrics_from_logfiles(log, "learning rate") == {"0": 1e-4, "1": 1e-4}
    assert evaluate.get_metrics_from_logfiles(log, "consumed samples") == {"0": 256.0, "1": 512.0}


def test_resume_continuity_requires_equal_steps_when_exact(tmp_path):
    reference = _training_log(tmp_path / "log-allranks.out", range(1, 101))
    resumed = _training_log(tmp_path / "log-resume-allranks.out", range(51, 56))

    result = evaluate.validate_resume_continuity(reference, resumed, expected_iterations=list(range(51, 56)))
    assert result["passed"], result["details"]
    assert result["metrics"]["resumed_iterations"] == [51, 52, 53, 54, 55]

    restarted_schedule = _training_log(tmp_path / "log-resume-allranks.out", range(51, 56), lr=lambda i: 1e-6)
    result = evaluate.validate_resume_continuity(reference, restarted_schedule)
    assert "iteration 51: learning rate 1e-06 after resuming, 0.0001 uninterrupted" in result["details"]

    drifted = _training_log(tmp_path / "log-resume-allranks.out", range(51, 56), loss=lambda i: 2.0 - i / 1000 + 1e-5)
    assert not evaluate.validate_resume_continuity(reference, drifted)["passed"]


def test_resume_continuity_with_a_loss_tolerance_and_missing_steps(tmp_path):
    reference = _training_log(tmp_path / "log-allranks.out", range(1, 101))
    drifted = _training_log(
        tmp_path / "log-resume-allranks.out",
        range(51, 54),
        loss=lambda i: (2.0 - i / 1000) * 1.0005,
        grad=lambda i: 1.01,
    )

    within = evaluate.validate_resume_continuity(reference, drifted, loss_tolerance=1e-3)
    assert within["passed"], within["details"]
    assert within["metrics"]["worst_relative_lm_loss_difference"] == pytest.approx(5e-4, rel=1e-3)

    outside = evaluate.validate_resume_continuity(reference, drifted, loss_tolerance=1e-4)
    assert "iteration 51: lm loss" in outside["details"]

    short = evaluate.validate_resume_continuity(
        reference, drifted, expected_iterations=list(range(51, 56)), loss_tolerance=1e-3
    )
    assert "did not log iterations [54, 55]" in short["details"]


def _log_with_validation(path, losses):
    lines = []
    for iteration in range(1, 101):
        lines.append(
            f" [2026-11-16 10:00:00] iteration {iteration:8d}/     100 | consumed samples: {iteration * 256:12d} | "
            f"elapsed time per iteration (ms): 3400.0 | learning rate: 1.000000E-04 | global batch size:   256 | "
            f"lm loss: {2.0 - iteration / 1000:.6E} | loss scale: 1.0 | grad norm: 1.000 |"
        )
        if iteration in losses:
            lines.append(
                f" validation loss at iteration {iteration} | lm loss value: {losses[iteration]:.6E} | "
                f"lm loss PPL: 7.000000E+00 | "
            )
    lines.append(" validation loss at iteration 100 on validation set | lm loss value: 9.000000E+00 | ")
    path.write_text("\n".join(lines) + "\n")
    return [str(path)]


def test_validation_losses_are_parsed_per_in_training_evaluation(tmp_path):
    log = _log_with_validation(tmp_path / "log-allranks.out", {50: 1.95, 100: 1.9})
    # The evaluation after training repeats step 100 on other validation batches and is not used.
    assert evaluate.get_validation_losses_from_logfiles(log) == {"49": 1.95, "99": 1.9}
    # Validation lines do not leak into the training-loss parse.
    assert evaluate.get_metrics_from_logfiles(log, "lm loss")["49"] == pytest.approx(1.95)
    assert len(evaluate.get_metrics_from_logfiles(log, "lm loss")) == 100


def test_validation_loss_check_within_and_outside_the_tolerance():
    golden = {"49": 2.0, "99": 1.9}
    assert evaluate.validate_validation_loss({"49": 2.001, "99": 1.9}, golden, loss_tolerance=1e-3)["passed"]
    outside = evaluate.validate_validation_loss({"49": 2.01, "99": 1.9}, golden, loss_tolerance=1e-3)
    assert "iteration 50: validation loss 2.010000 vs golden 2.000000" in outside["details"]
    missing = evaluate.validate_validation_loss({"49": 2.0, "99": float("nan")}, golden, loss_tolerance=1e-3)
    assert "iteration 100: validation loss is nan" in missing["details"]
    assert (
        "iteration 50: no validation loss logged"
        in (evaluate.validate_validation_loss({"99": 1.9}, golden, loss_tolerance=1e-3)["details"])
    )


def _golden(path, validation):
    steps = {
        str(step): {
            "lm loss": 2.0 - (step + 1) / 1000,
            "elapsed time per iteration (ms)": 3400.0,
            "GPU utilization": 1,
        }
        for step in range(100)
    }
    for step, loss in validation.items():
        steps[step]["validation lm loss"] = loss
    steps.update({"alloc": 1.0, "max_alloc": 1.0})
    path.write_text(json.dumps(steps))
    return str(path)


def _calc(tmp_path, log, golden, validation_config, monkeypatch):
    monkeypatch.setattr(
        evaluate, "get_metrics_from_logfiles", _metrics_with_memory(evaluate.get_metrics_from_logfiles)
    )
    return evaluate.calc_convergence_and_performance(
        model_family_name="test",
        model_recipe_name="test_recipe",
        assets_dir=str(tmp_path / "assets"),
        log_paths=log,
        loss_metric="lm loss",
        timing_metric="elapsed time per iteration (ms)",
        alloc_metric="alloc",
        max_alloc_metric="max_alloc",
        golden_values_path=golden,
        convergence_config={},
        performance_config={"timing_threshold": 0.1},
        memory_config={"memory_threshold": 0.05},
        validation_config=validation_config,
    )


def _metrics_with_memory(parse):
    def metrics(log_paths, metric):
        if metric in ("alloc", "max_alloc"):
            return 1.0
        if metric == "max_reserved":
            return None
        if metric == "GPU utilization":
            return {str(step): 1.0 for step in range(100)}
        return parse(log_paths, metric)

    return metrics


def test_calc_records_validation_losses_and_checks_them_against_golden_values(tmp_path, monkeypatch):
    log = _log_with_validation(tmp_path / "log-allranks.out", {50: 1.95, 100: 1.9})

    passed, message, current = _calc(tmp_path, log, _golden(tmp_path / "golden.json", {}), None, monkeypatch)
    assert passed, message
    assert current["49"]["validation lm loss"] == 1.95  # recorded for the next golden refresh

    golden = _golden(tmp_path / "golden.json", {"49": 1.95, "99": 1.8})
    passed, message, _ = _calc(tmp_path, log, golden, {"loss_tolerance": 0.01}, monkeypatch)
    assert not passed
    assert "Validation loss check failed" in message and "iteration 100" in message
    # Without a declared tolerance the golden validation losses are reported, not checked.
    passed, message, _ = _calc(tmp_path, log, golden, None, monkeypatch)
    assert passed, message
