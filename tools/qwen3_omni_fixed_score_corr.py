#!/usr/bin/env python3
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

"""Compare Qwen3-Omni Megatron actor fixed-score JSON against HF."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CorrSummary:
    """Summary metrics for a fixed-score actor/HF comparison."""

    actor_path: str
    hf_path: str
    num_common: int
    pearson: float
    diff_mean: float
    abs_diff_mean: float
    abs_diff_max: float


def _resolve_path(path_or_glob: str) -> str:
    matches = glob.glob(path_or_glob)
    if matches:
        return max(matches, key=lambda path: (os.path.getmtime(path), path))
    if glob.has_magic(path_or_glob):
        raise FileNotFoundError(f"no files matched: {path_or_glob}")
    if not os.path.exists(path_or_glob):
        raise FileNotFoundError(path_or_glob)
    return path_or_glob


def _load_scores(path: str, key: str) -> dict[int, float]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("ok") is False:
        error = payload.get("error") or "fixed-score payload is marked ok=false"
        raise ValueError(f"{path}: {error}")

    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON object with a rows list")

    scores: dict[int, float] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {index} is not an object")
        if row.get(key) is None:
            continue
        position = row.get("pos", index)
        try:
            scores[int(position)] = float(row[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: row {index} has invalid {key!r} or pos") from exc
    return scores


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("pearson inputs must have the same length")
    if len(xs) < 2:
        return math.nan

    x_mean = _mean(xs)
    y_mean = _mean(ys)
    x_centered = [value - x_mean for value in xs]
    y_centered = [value - y_mean for value in ys]
    numerator = sum(x * y for x, y in zip(x_centered, y_centered))
    x_norm = math.sqrt(sum(x * x for x in x_centered))
    y_norm = math.sqrt(sum(y * y for y in y_centered))
    if x_norm == 0.0 or y_norm == 0.0:
        return math.nan
    return numerator / (x_norm * y_norm)


def summarize(actor_path: str, hf_path: str) -> CorrSummary:
    """Load actor and HF fixed-score files and compute correlation metrics."""

    actor_path = _resolve_path(actor_path)
    hf_path = _resolve_path(hf_path)
    actor_scores = _load_scores(actor_path, "actor_log_prob")
    hf_scores = _load_scores(hf_path, "hf_log_prob")

    common_positions = sorted(actor_scores.keys() & hf_scores.keys())
    if not common_positions:
        raise ValueError("no common scored token positions")

    actor_values = [actor_scores[position] for position in common_positions]
    hf_values = [hf_scores[position] for position in common_positions]
    diffs = [actor - hf for actor, hf in zip(actor_values, hf_values)]
    abs_diffs = [abs(diff) for diff in diffs]

    return CorrSummary(
        actor_path=actor_path,
        hf_path=hf_path,
        num_common=len(common_positions),
        pearson=_pearson(actor_values, hf_values),
        diff_mean=_mean(diffs),
        abs_diff_mean=_mean(abs_diffs),
        abs_diff_max=max(abs_diffs),
    )


def _format_summary(summary: CorrSummary) -> str:
    return "\n".join(
        [
            f"actor_path={summary.actor_path}",
            f"hf_path={summary.hf_path}",
            f"num_common={summary.num_common}",
            f"pearson={summary.pearson:.9f}",
            f"diff_mean={summary.diff_mean:.9f}",
            f"abs_diff_mean={summary.abs_diff_mean:.9f}",
            f"abs_diff_max={summary.abs_diff_max:.9f}",
        ]
    )


def _passes(summary: CorrSummary, args: argparse.Namespace) -> bool:
    checks = [
        summary.num_common >= args.min_common,
        math.isfinite(summary.pearson) and summary.pearson >= args.min_pearson,
    ]
    if args.max_abs_mean is not None:
        checks.append(summary.abs_diff_mean <= args.max_abs_mean)
    if args.max_abs_max is not None:
        checks.append(summary.abs_diff_max <= args.max_abs_max)
    return all(checks)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate Megatron actor vs HF fixed-score JSON correlation without Ray.",
    )
    parser.add_argument("--actor", required=True, help="Actor fixed-score JSON path or glob.")
    parser.add_argument("--hf", required=True, help="HF fixed-score JSON path or glob.")
    parser.add_argument("--min-common", type=int, default=2, help="Minimum common token positions.")
    parser.add_argument("--min-pearson", type=float, default=0.99, help="Minimum Pearson correlation.")
    parser.add_argument("--max-abs-mean", type=float, default=None, help="Optional max mean absolute diff.")
    parser.add_argument("--max-abs-max", type=float, default=None, help="Optional max absolute diff.")
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Run the fixed-score correlation command line interface."""

    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        summary = summarize(args.actor, args.hf)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(asdict(summary), sort_keys=True))
    else:
        print(_format_summary(summary))

    if not _passes(summary, args):
        print(
            f"FAIL: correlation gate did not pass (min_common={args.min_common}, min_pearson={args.min_pearson})",
            file=sys.stderr,
        )
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
