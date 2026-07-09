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

"""Check compact step-time and final-loss goldens for recipe proxy tests."""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from scripts.performance.utils.evaluate import get_metrics_from_logfiles


def summarize_metrics(log_paths: list[str], *, warmup_steps: int) -> dict[str, float | int]:
    """Summarize steady-state iteration time and the final logged loss."""
    iteration_times = get_metrics_from_logfiles(log_paths, "elapsed time per iteration (ms)")
    losses = get_metrics_from_logfiles(log_paths, "lm loss")

    ordered_times = [value for _, value in sorted(iteration_times.items(), key=lambda item: int(item[0]))]
    steady_state_times = ordered_times[warmup_steps:]
    if not steady_state_times:
        raise ValueError(
            f"No steady-state iteration times remain after skipping {warmup_steps} warmup steps "
            f"from {len(ordered_times)} logged steps."
        )
    if not losses:
        raise ValueError("No 'lm loss' values were found in the proxy training log.")

    final_step = max(losses, key=int)
    return {
        "warmup_steps": warmup_steps,
        "mean_step_time_ms": statistics.mean(steady_state_times),
        "final_loss": losses[final_step],
    }


def validate_metrics(actual: dict[str, float | int], golden: dict[str, Any]) -> list[str]:
    """Return human-readable validation failures for a compact golden."""
    failures = []
    golden_step_time = float(golden["mean_step_time_ms"])
    step_time_tolerance = float(golden.get("step_time_regression_tolerance", 0.2))
    actual_step_time = float(actual["mean_step_time_ms"])
    max_step_time = golden_step_time * (1.0 + step_time_tolerance)
    if actual_step_time > max_step_time:
        failures.append(
            f"mean step time regressed: {actual_step_time:.3f} ms > {max_step_time:.3f} ms "
            f"(golden {golden_step_time:.3f} ms, tolerance {step_time_tolerance:.1%})"
        )

    golden_final_loss = golden.get("final_loss")
    if golden_final_loss is not None:
        golden_final_loss = float(golden_final_loss)
        actual_final_loss = float(actual["final_loss"])
        final_loss_tolerance = float(golden.get("final_loss_relative_tolerance", 0.05))
        relative_difference = abs(actual_final_loss - golden_final_loss) / max(abs(golden_final_loss), 1e-12)
        if relative_difference > final_loss_tolerance:
            failures.append(
                f"final loss changed: {actual_final_loss:.6f} vs golden {golden_final_loss:.6f} "
                f"(relative difference {relative_difference:.2%}, tolerance {final_loss_tolerance:.1%})"
            )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", action="append", required=True, help="Training log to parse; repeat as needed.")
    parser.add_argument("--golden-values", type=Path, help="Compact golden JSON. Omit to print bootstrap values.")
    parser.add_argument("--warmup-steps", type=int, default=3, help="Steps to exclude before averaging timing.")
    args = parser.parse_args()

    golden = None
    warmup_steps = args.warmup_steps
    if args.golden_values is not None:
        with args.golden_values.open(encoding="utf-8") as fp:
            golden = json.load(fp)
        warmup_steps = int(golden.get("warmup_steps", warmup_steps))

    actual = summarize_metrics(args.log_path, warmup_steps=warmup_steps)
    print(json.dumps(actual, indent=2, sort_keys=True))

    if golden is None:
        return 0

    failures = validate_metrics(actual, golden)
    if failures:
        for failure in failures:
            print(f"FAILED: {failure}")
        return 1

    print("Proxy metric golden validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
