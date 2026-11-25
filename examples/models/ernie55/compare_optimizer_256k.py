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

"""Compare two complete eight-document 256K optimizer captures using only NumPy."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SHAPES = {"lm_head": (201216, 4096), "embedding": (201216, 4096), "last_shared_fc2": (4096, 2048)}
FIELDS = ("pre", "grad", "post")


def check(condition, reason):
    """Reject incomplete, mixed or incompatible captures before computing numerical gates."""
    if not condition:
        raise ValueError(reason)


def coordinates(name):
    """Return the complete declared set of sampled logical coordinates."""
    base = np.arange(0, int(np.prod(SHAPES[name])), 1021, dtype=np.int64)
    return np.union1d(base, np.arange(4096, 8192)) if name == "embedding" else base


def read_arrays(folder, records, name):
    """Join the selected FP32 master shards by global parameter index."""
    chunks = []
    for r in records:
        owner = r["observed"].get(name)
        if not owner or not owner["primary"]:
            continue
        with np.load(folder / f"{r['rank']:02}-{name}.npz") as z:
            ix = z["index"]
            check(np.all((owner["start"] <= ix) & (ix < owner["end"])), f"{name}: index outside its owner")
            values = [z[field] for field in FIELDS]
            check(
                all(v.dtype == np.float32 and v.shape == ix.shape and np.isfinite(v).all() for v in values),
                f"{name}: invalid FP32 values",
            )
            chunks.append((ix, *values))
    check(bool(chunks), f"No parameter owner for {name}")
    ix, *arrays = [np.concatenate(parts) for parts in zip(*chunks)]
    order = np.argsort(ix)
    check(np.array_equal(ix[order], coordinates(name)), f"{name}: overlapping or missing coordinates")
    data = {key: values[order] for key, values in zip(FIELDS, arrays)}
    data["delta"] = data["post"].astype(np.float64) - data["pre"].astype(np.float64)
    return data


def input_hashes(folder, records):
    """Verify every physical position and hash actual token IDs and targets per GA/DP pair."""
    hashes = []
    for dp in range(4):
        ranks = [r for r in records if r["dp"] == dp]
        check(sorted(r["cp"] for r in ranks) == list(range(16)), f"DP {dp}: expected all 16 CP owners")
        buckets = [[] for _ in range(8)]
        for record in ranks:
            with np.load(folder / f"{record['rank']:02}-batches.npz") as z:
                for mb, parts in enumerate(buckets):
                    parts.append(tuple(z[f"{field}{mb}"] for field in ("pos", "token", "label")))
        for parts in buckets:
            pos, ids, labels = [np.concatenate(col) for col in zip(*parts)]
            order = np.argsort(pos)
            check(np.array_equal(pos[order], np.arange(262144)), "Incomplete physical CP coverage")
            check(np.count_nonzero(labels >= 0) == 262080, "Different valid-label count for eight documents")
            h = hashlib.sha256()
            for values in (ids, labels):
                h.update(values[order].astype("<i4", copy=False).tobytes())
            hashes.append(h.hexdigest())
    return hashes


def read_capture(path):
    """Check both full native updates and the optimizer state carried across the warmup."""
    steps = []
    initial = []
    for step in (1, 2):
        folder = path / f"step{step}"
        records = [json.loads((folder / f"complete-{r:02}.json").read_text()) for r in range(64)]
        for rank, r in enumerate(records):
            check(
                r["rank"] == rank and r["mb"] == 8 and r["optimizer_step_count"] == step and r["update_success"],
                "Incorrect rank, GA or optimizer step",
            )
            check(r["loss_scale"] == 1, "Expected the BF16 fixture at loss scale 1")
            check(
                {name: tuple(value["shape"]) for name, value in r["logical"].items()} == SHAPES,
                "Different model parameter shapes",
            )
        arrays = {k: read_arrays(folder, records, k) for k in SHAPES}
        rates = {}
        for name in SHAPES:
            observed = {float(r["param_group"][name]["lr"]) for r in records if name in r["param_group"]}
            expected = 0.0 if step == 1 else 2e-6
            check(
                len(observed) == 1 and np.isclose(next(iter(observed)), expected, atol=0, rtol=1e-12),
                f"Unexpected step {step} learning rate(s) for {name}: {observed}",
            )
            rates[name] = observed.pop()
        if step == 1:
            check(all(not np.any(v["delta"]) for v in arrays.values()), "Masters changed with zero learning rate")
            initial = [r["post_state"] for r in records]
        else:
            for previous, r in zip(initial, records):
                check(previous == r["pre_state"], "Optimizer state was reset between updates")
            for name in SHAPES:
                check(any(r["pre_state"].get(name) for r in records), f"Missing carried optimizer moments for {name}")
                check(
                    np.array_equal(steps[0]["arrays"][name]["post"], arrays[name]["pre"]),
                    f"Lost master continuity for {name}",
                )
                check(np.any(arrays[name]["delta"]), f"Missing nonzero optimizer delta for {name}")
        norms = sorted({float(r["grad_norm"]) for r in records})
        check(bool(norms) and np.isfinite(norms).all(), "Invalid optimizer norm")
        steps.append(dict(arrays=arrays, inputs=input_hashes(folder, records), rates=rates, grad_norms=norms))
    return steps


def errors(reference, candidate):
    """Use the original trainer as the denominator without fitting a scale."""
    r = reference.astype(np.float64, copy=False)
    diff = candidate.astype(np.float64, copy=False) - r
    norm = np.linalg.norm(r)
    peak = np.abs(r).max()
    check(norm > 0 and peak > 0, "Undefined relative error for a zero reference vector")
    l2 = float(np.linalg.norm(diff) / norm)
    relative_peak = float(np.max(np.abs(diff)) / peak)
    return dict(
        elements=len(r), relative_l2=l2, peak_relative=relative_peak, passed=l2 <= 0.005 and relative_peak <= 0.01
    )


def compare(ref_path, candidate_path):
    """Score the first nonzero optimizer update after validating both complete inputs."""
    ref, cur = read_capture(ref_path), read_capture(candidate_path)
    for step in (0, 1):
        check(ref[step]["inputs"] == cur[step]["inputs"], f"Input IDs or labels differ on step {step + 1}")
        for name in SHAPES:
            check(
                np.array_equal(ref[step]["arrays"][name]["pre"], cur[step]["arrays"][name]["pre"]),
                f"Different starting masters for step {step + 1} {name}",
            )
    report = {}
    for name in SHAPES:
        x, y = ref[1]["arrays"][name], cur[1]["arrays"][name]
        selectors = {name: slice(None)}
        if name == "embedding":
            ix = coordinates(name)
            bos = (ix >= 4096) & (ix < 8192)
            selectors.update(embedding_bos=bos, embedding_nonbos=~bos)
        for key, mask in selectors.items():
            report[key] = {field: errors(x[field][mask], y[field][mask]) for field in ("grad", "delta")}
    return dict(
        matrices=report,
        gate_relative_l2=0.005,
        gate_peak_relative=0.01,
        step2_lr=ref[1]["rates"],
        reference_grad_norms=ref[1]["grad_norms"],
        candidate_grad_norms=cur[1]["grad_norms"],
    )


def main():
    """Print relative errors as fractions, optionally write JSON and exit nonzero on failed gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail on sampled preclip gradient or true update errors above either limit.",
    )
    args = parser.parse_args()
    report = compare(args.reference, args.candidate)
    out = json.dumps(report, indent=2)
    print(out)
    if args.json:
        args.json.write_text(out + "\n")
    if args.check and not all(
        metric["passed"] for values in report["matrices"].values() for metric in values.values()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
