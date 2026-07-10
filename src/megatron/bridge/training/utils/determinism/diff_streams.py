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

"""Offline diff of two collective-trace fingerprint streams (prototype).

Given two job directories of per-logical-rank ``.fp`` streams (from
``collective_trace.enable``), find — for each matching logical rank — the **first**
collective whose output signature diverges. That record is the root-cause candidate.

Run::

    uv run python -m megatron.bridge.training.utils.determinism.diff_streams \\
        /lustre/.../det_streams/job_A  /lustre/.../det_streams/job_B
"""

import argparse
import glob
import json
import os
from typing import Optional


def _load(path: str) -> tuple[dict, list[dict]]:
    """Load a stream file → (header, records)."""
    header: dict = {}
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_header"):
                header = obj
            else:
                records.append(obj)
    return header, records


def _index(records: list[dict]) -> dict:
    """Key records by (window, group, op, align_idx) for cross-process alignment."""
    idx = {}
    for r in records:
        idx[(r.get("window"), r["group"], r["op"], r["align_idx"])] = r
    return idx


def _digest(s: dict):
    """Signature content hash, tolerating the pre-rename schema (crc32 -> digest)."""
    return s.get("digest", s.get("crc32"))


def _sigs_match(a: Optional[list], b: Optional[list]) -> bool:
    """Bitwise match of two lists of signature dicts (crc + shape + dtype)."""
    if a is None or b is None:
        return a == b
    if len(a) != len(b):
        return False
    for sa, sb in zip(a, b):
        if sa is None or sb is None:
            if sa != sb:
                return False
            continue
        if (sa["shape"], sa["dtype"], _digest(sa)) != (sb["shape"], sb["dtype"], _digest(sb)):
            return False
    return True


def _sum_gap(a: Optional[list], b: Optional[list]) -> float:
    """Max |sum_a - sum_b| over the signature list — divergence magnitude proxy.

    Returns NaN when the streams carry no ``sum`` moment (the digest-only fast path):
    the digest already localizes the divergence; the magnitude is just not available.
    """
    if not a or not b:
        return float("nan")
    gap = float("nan")
    for sa, sb in zip(a, b):
        if sa and sb and "sum" in sa and "sum" in sb:
            gap = max(0.0 if gap != gap else gap, abs(sa["sum"] - sb["sum"]))
    return gap


def diff_rank(path_a: str, path_b: str) -> Optional[dict]:
    """Diff one logical rank's two streams; return the first divergence or None."""
    _, rec_a = _load(path_a)
    _, rec_b = _load(path_b)
    idx_b = _index(rec_b)

    # Walk job A in execution order; the first key whose output diverges wins.
    for r in rec_a:
        key = (r.get("window"), r["group"], r["op"], r["align_idx"])
        rb = idx_b.get(key)
        if rb is None:
            continue
        input_match = _sigs_match(r.get("input"), rb.get("input"))
        output_match = _sigs_match(r.get("output"), rb.get("output"))
        if not output_match:
            return {
                "key": key,
                "op": r["op"],
                "group": r["group"],
                "window": r.get("window"),
                "align_idx": r["align_idx"],
                "seq_a": r["seq_id"],
                "scope": r.get("scope"),
                "caller_a": r.get("caller"),
                "caller_b": rb.get("caller"),
                "input_match": input_match,
                "output_sum_gap": _sum_gap(r.get("output"), rb.get("output")),
            }
    return None


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Diff two collective-trace stream dirs.")
    ap.add_argument("dir_a", help="job A stream directory")
    ap.add_argument("dir_b", help="job B stream directory")
    args = ap.parse_args()

    files_a = {os.path.basename(p): p for p in glob.glob(os.path.join(args.dir_a, "*.fp"))}
    files_b = {os.path.basename(p): p for p in glob.glob(os.path.join(args.dir_b, "*.fp"))}
    common = sorted(set(files_a) & set(files_b))

    if not common:
        print(f"No matching logical-rank stream files between {args.dir_a} and {args.dir_b}")
        return

    # Precondition: comparison is only valid when both jobs share the parallel
    # config. Differing sizes (e.g. 24n vs 48n) make same-coord comparison
    # meaningless (dp ranges differ → different data shards). Refuse.
    hdr_a, _ = _load(files_a[common[0]])
    hdr_b, _ = _load(files_b[common[0]])
    cfg_a, cfg_b = hdr_a.get("config", {}), hdr_b.get("config", {})
    if cfg_a and cfg_b and cfg_a != cfg_b:
        print("REFUSING TO COMPARE: parallel config differs between the two jobs.")
        print(f"  job A config: {cfg_a}")
        print(f"  job B config: {cfg_b}")
        print(
            "  Same-coord comparison is only valid at identical parallel config "
            "(same rank decomposition + same per-dp data shard)."
        )
        return

    print(f"Comparing {len(common)} logical ranks ({len(files_a)} vs {len(files_b)} files present)\n")
    any_div = False
    for fname in common:
        div = diff_rank(files_a[fname], files_b[fname])
        coord = fname.replace("stream_", "").replace(".fp", "")
        if div is None:
            print(f"  [{coord}] OK — all outputs match")
        else:
            any_div = True
            if div["group"] == "aten":
                # Op-level record: first divergent op in execution order is the root
                # cause by construction (everything earlier, incl. its inputs, matched).
                cause = "first divergent op → ROOT CAUSE (ordered-stream: all earlier ops matched)"
            elif div["input_match"]:
                cause = "INPUTS MATCHED → reduction-order/topology root cause"
            else:
                cause = "input also differs (upstream — look earlier in the stream)"
            print(
                f"  [{coord}] FIRST DIVERGENCE: {div['op']} on group '{div['group']}' "
                f"(window={div['window']}, align_idx={div['align_idx']}, seq_a={div['seq_a']})"
            )
            print(f"            layer={div.get('scope')}  caller_a={div['caller_a']}  caller_b={div['caller_b']}")
            gap = div["output_sum_gap"]
            mag = "|Δsum| n/a (digest-only)" if gap != gap else f"output |Δsum|={gap:.3e}"
            print(f"            {mag}  — {cause}")

    print()
    print(
        "=> Non-determinism localized to the first divergent record(s) above."
        if any_div
        else "=> No divergence found. If only collectives were traced, enable op-level (DET_TRACE_OPS) to look inside compute."
    )


if __name__ == "__main__":
    main()
