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

"""Compare short official BAGEL and Bridge/MCore training traces."""

import argparse
import json
import logging
import math
from pathlib import Path


logger = logging.getLogger(__name__)
LOSS_ATOL = 1e-2
GRAD_RTOL = 1e-2


def parse_args() -> argparse.Namespace:
    """Parse training-parity arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_trace(path: Path) -> list[dict]:
    """Load one JSONL training trace."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    """Compare every recorded step and write a fixed-tolerance report."""
    args = parse_args()
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}")
    official = load_trace(args.official)
    bridge = load_trace(args.bridge)
    if len(official) != len(bridge) or not official:
        raise ValueError(f"Training trace lengths differ: official={len(official)}, bridge={len(bridge)}")
    steps = []
    passed = True
    for reference, actual in zip(official, bridge, strict=True):
        if reference["step"] != actual["step"]:
            raise ValueError(f"Training steps differ: official={reference['step']}, bridge={actual['step']}")
        differences = {
            "ce": abs(reference["ce"] - actual["losses"]["ce"]),
            "mse": abs(reference["mse"] - actual["losses"]["mse"]),
            "loss": abs(reference["ce"] + reference["mse"] - actual["losses"]["loss"]),
            "grad_norm_relative": abs(reference["total_norm"] - actual["grad_norm"]) / reference["total_norm"],
            "lr": abs(reference["lr"] - actual["lr"]),
        }
        step_passed = all(differences[name] < LOSS_ATOL for name in ("ce", "mse", "loss"))
        step_passed &= differences["grad_norm_relative"] < GRAD_RTOL
        step_passed &= math.isclose(reference["lr"], actual["lr"], rel_tol=0.0, abs_tol=1e-15)
        steps.append({"step": reference["step"], "differences": differences, "passed": step_passed})
        passed &= step_passed
    report = {"loss_atol": LOSS_ATOL, "grad_rtol": GRAD_RTOL, "steps": steps, "passed": passed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote BAGEL training parity report to %s", args.output)
    if not passed:
        raise RuntimeError("BAGEL training parity missed the fixed tolerance")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
