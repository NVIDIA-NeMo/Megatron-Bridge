#!/usr/bin/env python3
"""Build a full NVTX namespace comparison between det/non-det sqlite profiles."""
import csv
import re
import sqlite3
import sys
from collections import defaultdict

OP_ID_RE = re.compile(r",\s*(op_id|seq)\s*=\s*\d+")


def load(path):
    """Return {name: (total_ms, count)} aggregated over all NVTX text events."""
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT text, end-start FROM NVTX_EVENTS WHERE text IS NOT NULL AND end IS NOT NULL")
    bucket = defaultdict(lambda: [0, 0])
    for text, dur in cur:
        name = OP_ID_RE.sub("", text)
        bucket[name][0] += dur or 0
        bucket[name][1] += 1
    conn.close()
    return {k: (v[0] / 1e6, v[1]) for k, v in bucket.items()}


def namespace_of(name):
    """Reduce a range name to a coarser namespace bucket for grouping."""
    if name.startswith("autograd::engine::evaluate_function: "):
        return "autograd::evaluate_function"
    if name.startswith("autograd::engine"):
        return "autograd::engine"
    if name.startswith("aten::"):
        return "aten::*"
    if name.startswith("nccl"):
        return "nccl::*"
    if name.startswith("nvte_"):
        return "nvte::*"
    if name.startswith("Optimizer.step"):
        return "Optimizer.step"
    if name.startswith("megatron.core.transformer.mlp."):
        return "mcore.mlp"
    if name.startswith("megatron.core.transformer.attention."):
        return "mcore.attention"
    if name.startswith("megatron.core.transformer.transformer_layer."):
        return "mcore.transformer_layer"
    if name.startswith("megatron.core.pipeline_parallel."):
        return "mcore.pipeline_parallel"
    if name.startswith("megatron.core.fusions."):
        return "mcore.fusions"
    if name.startswith("megatron.core.models."):
        return "mcore.models"
    if name.startswith("megatron.core.ssm.") or name.startswith("megatron.core.transformer.spec_utils"):
        return "mcore.ssm"
    if name.startswith("megatron.core."):
        return "mcore.other"
    if "Backward" in name and "::" not in name:
        return "backward_node"
    return "other"


def main():
    det_path = sys.argv[1]
    non_path = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "/tmp/nvtx-compare"
    det = load(det_path)
    non = load(non_path)

    # Write per-name CSV with both columns
    all_names = sorted(set(det) | set(non), key=lambda k: -(abs(det.get(k, (0, 0))[0] - non.get(k, (0, 0))[0])))
    with open(f"{out_dir}/nvtx_per_range.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Range", "Namespace", "Det ms", "Non-det ms", "Δ ms", "Δ %", "Det count", "Non-det count"])
        for name in all_names:
            d_ms, d_n = det.get(name, (0, 0))
            n_ms, n_n = non.get(name, (0, 0))
            delta = d_ms - n_ms
            pct = f"{(d_ms - n_ms) / n_ms * 100:+.1f}" if n_ms > 0 else ("" if d_ms == 0 else "new")
            w.writerow([name, namespace_of(name), f"{d_ms:.2f}", f"{n_ms:.2f}", f"{delta:+.2f}", pct, d_n, n_n])

    # Write namespace-aggregated CSV
    ns_det = defaultdict(lambda: [0, 0])
    ns_non = defaultdict(lambda: [0, 0])
    for n, (ms, cnt) in det.items():
        ns = namespace_of(n)
        ns_det[ns][0] += ms
        ns_det[ns][1] += cnt
    for n, (ms, cnt) in non.items():
        ns = namespace_of(n)
        ns_non[ns][0] += ms
        ns_non[ns][1] += cnt
    namespaces = sorted(set(ns_det) | set(ns_non), key=lambda k: -(abs(ns_det.get(k, [0, 0])[0] - ns_non.get(k, [0, 0])[0])))
    with open(f"{out_dir}/nvtx_per_namespace.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Namespace", "Det ms", "Non-det ms", "Δ ms", "Δ %", "Det count", "Non-det count"])
        for ns in namespaces:
            d_ms, d_n = ns_det.get(ns, [0, 0])
            n_ms, n_n = ns_non.get(ns, [0, 0])
            delta = d_ms - n_ms
            pct = f"{(d_ms - n_ms) / n_ms * 100:+.1f}" if n_ms > 0 else ""
            w.writerow([ns, f"{d_ms:.2f}", f"{n_ms:.2f}", f"{delta:+.2f}", pct, d_n, n_n])

    # Print top 30 most-different ranges, and namespace summary
    print(f"\n=== NAMESPACE TOTALS (sorted by |Δ ms|) ===")
    print(f"{'namespace':<30}  {'det ms':>10}  {'nondet ms':>10}  {'Δ ms':>10}  {'Δ %':>8}  {'d cnt':>8}  {'n cnt':>8}")
    print("-" * 100)
    for ns in namespaces:
        d_ms, d_n = ns_det.get(ns, [0, 0])
        n_ms, n_n = ns_non.get(ns, [0, 0])
        pct = (d_ms - n_ms) / n_ms * 100 if n_ms > 0 else float("nan")
        print(f"{ns:<30}  {d_ms:>10.2f}  {n_ms:>10.2f}  {d_ms - n_ms:>+10.2f}  {pct:>+7.1f}%  {d_n:>8}  {n_n:>8}")

    print(f"\n=== TOP 30 RANGES BY |Δ ms| (full names; op_id/seq stripped) ===")
    print(f"{'range':<80}  {'det ms':>10}  {'nondet ms':>10}  {'Δ ms':>10}")
    print("-" * 130)
    for name in all_names[:30]:
        d_ms, _ = det.get(name, (0, 0))
        n_ms, _ = non.get(name, (0, 0))
        short = name if len(name) <= 80 else name[:79] + "…"
        print(f"{short:<80}  {d_ms:>10.2f}  {n_ms:>10.2f}  {d_ms - n_ms:>+10.2f}")


if __name__ == "__main__":
    main()


# additional: per-namespace top-10 detail
