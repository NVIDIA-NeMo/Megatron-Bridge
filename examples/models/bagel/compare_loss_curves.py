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

"""Compare official BAGEL and Bridge loss curves."""

import argparse
import csv
import json
import logging
import math
from pathlib import Path


logger = logging.getLogger(__name__)
LOSS_ATOL = 1e-2
GRAD_RTOL = 1e-2


def parse_args() -> argparse.Namespace:
    """Parse loss-curve comparison arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--curve-output", type=Path, required=True)
    return parser.parse_args()


def load_trace(path: Path) -> list[dict]:
    """Load one JSONL training trace."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def summarize(values: list[float]) -> dict[str, float]:
    """Summarize one sequence of nonnegative differences."""
    return {
        "max": max(values),
        "mean": sum(values) / len(values),
        "rmse": math.sqrt(sum(value * value for value in values) / len(values)),
        "final": values[-1],
    }


def correlation(reference: list[float], actual: list[float]) -> float:
    """Calculate Pearson correlation between two loss curves."""
    reference_mean = sum(reference) / len(reference)
    actual_mean = sum(actual) / len(actual)
    numerator = sum((left - reference_mean) * (right - actual_mean) for left, right in zip(reference, actual))
    denominator = math.sqrt(
        sum((value - reference_mean) ** 2 for value in reference) * sum((value - actual_mean) ** 2 for value in actual)
    )
    return numerator / denominator


def main() -> None:
    """Write per-step curve values and aggregate parity measurements."""
    args = parse_args()
    if args.output.exists() or args.curve_output.exists():
        raise ValueError("Loss-curve output already exists")
    official = load_trace(args.official)
    bridge = load_trace(args.bridge)
    if len(official) != len(bridge) or not official:
        raise ValueError(f"Training trace lengths differ: official={len(official)}, bridge={len(bridge)}")
    rows = []
    official_losses = []
    bridge_losses = []
    for reference, actual in zip(official, bridge, strict=True):
        if reference["step"] != actual["step"]:
            raise ValueError(f"Training steps differ: official={reference['step']}, bridge={actual['step']}")
        official_loss = reference["ce"] + reference["mse"]
        bridge_loss = actual["losses"]["loss"]
        row = {
            "step": reference["step"],
            "official_ce": reference["ce"],
            "bridge_ce": actual["losses"]["ce"],
            "official_mse": reference["mse"],
            "bridge_mse": actual["losses"]["mse"],
            "official_loss": official_loss,
            "bridge_loss": bridge_loss,
            "official_grad_norm": reference["total_norm"],
            "bridge_grad_norm": actual["grad_norm"],
            "official_lr": reference["lr"],
            "bridge_lr": actual["lr"],
        }
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError(f"Non-finite training metric at step {reference['step']}")
        rows.append(row)
        official_losses.append(official_loss)
        bridge_losses.append(bridge_loss)
    differences = {
        "ce": [abs(row["official_ce"] - row["bridge_ce"]) for row in rows],
        "mse": [abs(row["official_mse"] - row["bridge_mse"]) for row in rows],
        "loss": [abs(row["official_loss"] - row["bridge_loss"]) for row in rows],
        "grad_norm_relative": [
            abs(row["official_grad_norm"] - row["bridge_grad_norm"]) / row["official_grad_norm"] for row in rows
        ],
        "lr": [abs(row["official_lr"] - row["bridge_lr"]) for row in rows],
    }
    report = {
        "num_steps": len(rows),
        "loss_atol": LOSS_ATOL,
        "grad_rtol": GRAD_RTOL,
        "loss_correlation": correlation(official_losses, bridge_losses),
        "differences": {name: summarize(values) for name, values in differences.items()},
        "violations": {
            "ce": sum(value >= LOSS_ATOL for value in differences["ce"]),
            "mse": sum(value >= LOSS_ATOL for value in differences["mse"]),
            "loss": sum(value >= LOSS_ATOL for value in differences["loss"]),
            "grad_norm_relative": sum(value >= GRAD_RTOL for value in differences["grad_norm_relative"]),
            "lr": sum(value > 1e-15 for value in differences["lr"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with args.curve_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d-step BAGEL loss-curve comparison to %s", len(rows), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
