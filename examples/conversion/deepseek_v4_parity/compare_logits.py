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

"""Offline comparator for HF vs Megatron-Bridge logits artifacts.

Loads two ``.pt`` files produced by ``run_hf_reference.py`` and ``run_mbridge.py``, matches
prompts by ``id``, and reports per-prompt cosine similarity, max-abs-diff, and top-K agreement
at the last real position (primary metric). When both artifacts also carry
``last_padded_logits`` (the position compare.py samples), padded-position cosine is reported
alongside for diagnostic comparison — BF16 cosine is typically tighter at the padded position.

Megatron may pad its vocab dimension for kernel efficiency; we truncate to the HF vocab size
before comparing logits.

Run example:

    python examples/conversion/deepseek_v4_parity/compare_logits.py \
        --hf /chcui/parity/dsv4/logits_hf.pt \
        --mb /chcui/parity/dsv4/logits_mb.pt \
        --threshold 0.9999
"""

import argparse
import logging
from pathlib import Path

import torch


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare HF vs Megatron-Bridge logits artifacts")
    parser.add_argument("--hf", type=str, required=True, help="Path to HF artifact .pt")
    parser.add_argument("--mb", type=str, required=True, help="Path to Megatron-Bridge artifact .pt")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9999,
        help="Cosine similarity pass threshold (BF16 expectation per the parity-testing skill)",
    )
    parser.add_argument("--report", type=str, default=None, help="Optional path to write a markdown report")
    return parser.parse_args()


def _index_by_id(results: list[dict]) -> dict[str, dict]:
    """Index a results list by prompt id (falling back to position-based ids)."""
    out: dict[str, dict] = {}
    for i, r in enumerate(results):
        rid = r.get("id", str(i))
        out[rid] = r
    return out


def _align_vocab(hf_logits: torch.Tensor, mb_logits: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Truncate Megatron's vocab-padded logits to the HF vocab size before comparing."""
    hf_vocab = hf_logits.shape[0]
    mb_vocab = mb_logits.shape[0]
    if mb_vocab > hf_vocab:
        return mb_logits[:hf_vocab], hf_vocab, mb_vocab
    if mb_vocab < hf_vocab:
        pad = torch.full((hf_vocab - mb_vocab,), float("-inf"))
        return torch.cat([mb_logits, pad], dim=0), hf_vocab, mb_vocab
    return mb_logits, hf_vocab, mb_vocab


def _compare_at(hf_entry: dict, mb_entry: dict, field: str) -> dict | None:
    """Compute cosine / max-diff / top-K at one logits position. Returns None when missing."""
    if field not in hf_entry or field not in mb_entry:
        return None
    hf_logits = hf_entry[field].float()
    mb_logits = mb_entry[field].float()
    mb_logits_cmp, hf_vocab, mb_vocab = _align_vocab(hf_logits, mb_logits)
    diff = (hf_logits - mb_logits_cmp).abs()
    cos = torch.nn.functional.cosine_similarity(hf_logits.unsqueeze(0), mb_logits_cmp.unsqueeze(0)).item()
    hf_top1 = int(torch.argmax(hf_logits).item())
    mb_top1 = int(torch.argmax(mb_logits_cmp).item())
    k = min(int(hf_entry.get("top_k", 5)), int(mb_entry.get("top_k", 5)), 10)
    hf_topk = set(int(t.item()) for t in torch.topk(hf_logits, k).indices)
    mb_topk = set(int(t.item()) for t in torch.topk(mb_logits_cmp, k).indices)
    return {
        "hf_vocab": hf_vocab,
        "mb_vocab": mb_vocab,
        "cosine": cos,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "top1_match": hf_top1 == mb_top1,
        "hf_top1": hf_top1,
        "mb_top1": mb_top1,
        "topk_k": k,
        "topk_overlap": len(hf_topk & mb_topk),
    }


def _compare_one(hf_entry: dict, mb_entry: dict) -> dict:
    """Compute parity stats at the primary (last-real) and, when present, padded position."""
    primary = _compare_at(hf_entry, mb_entry, "last_pos_logits")
    assert primary is not None, "Both artifacts must contain last_pos_logits."
    padded = _compare_at(hf_entry, mb_entry, "last_padded_logits")
    out = dict(primary)
    if padded is not None:
        out["cosine_padded"] = padded["cosine"]
        out["max_abs_diff_padded"] = padded["max_abs_diff"]
        out["top1_match_padded"] = padded["top1_match"]
        out["topk_overlap_padded"] = padded["topk_overlap"]
    return out


def _format_md(rows: list[tuple[str, dict]], threshold: float, hf_meta: dict, mb_meta: dict) -> str:
    """Format the per-prompt rows as a markdown report."""
    lines = []
    lines.append("# DeepSeek Parity Report\n")
    lines.append(f"- HF source: `{hf_meta.get('source', '?')}` model=`{hf_meta.get('model_path', '?')}`")
    lines.append(f"- MB source: `{mb_meta.get('source', '?')}` model=`{mb_meta.get('model_path', '?')}`")
    if "tp" in mb_meta:
        lines.append(
            f"- MB parallelism: tp={mb_meta['tp']} pp={mb_meta['pp']} ep={mb_meta['ep']} etp={mb_meta['etp']}"
        )
    lines.append(f"- Threshold: cosine >= {threshold} at last-real position\n")
    has_padded = any("cosine_padded" in r for _, r in rows)
    if has_padded:
        lines.append("| id | cos@real | cos@padded | max_diff@real | top1@real | top1@padded | topK@real |")
        lines.append("|---|---|---|---|---|---|---|")
    else:
        lines.append("| id | cosine | max_abs_diff | top1 match | top-K overlap | HF vocab | MB vocab |")
        lines.append("|---|---|---|---|---|---|---|")
    for rid, r in rows:
        status = "PASS" if r["cosine"] >= threshold and r["top1_match"] else "FAIL"
        if has_padded:
            cos_p = f"{r.get('cosine_padded', float('nan')):.6f}" if "cosine_padded" in r else "n/a"
            top1_p = f"{r['top1_match_padded']}" if "top1_match_padded" in r else "n/a"
            lines.append(
                f"| {rid} ({status}) | {r['cosine']:.6f} | {cos_p} | {r['max_abs_diff']:.4e} | "
                f"{r['top1_match']} ({r['hf_top1']} vs {r['mb_top1']}) | {top1_p} | "
                f"{r['topk_overlap']}/{r['topk_k']} |"
            )
        else:
            lines.append(
                f"| {rid} ({status}) | {r['cosine']:.6f} | {r['max_abs_diff']:.4e} | "
                f"{r['top1_match']} ({r['hf_top1']} vs {r['mb_top1']}) | "
                f"{r['topk_overlap']}/{r['topk_k']} | {r['hf_vocab']} | {r['mb_vocab']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the comparator and exit non-zero if any prompt falls below threshold."""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    args = parse_args()

    hf_artifact = torch.load(args.hf, map_location="cpu", weights_only=False)
    mb_artifact = torch.load(args.mb, map_location="cpu", weights_only=False)

    hf_by_id = _index_by_id(hf_artifact["results"])
    mb_by_id = _index_by_id(mb_artifact["results"])

    common = sorted(set(hf_by_id) & set(mb_by_id))
    missing_in_mb = sorted(set(hf_by_id) - set(mb_by_id))
    missing_in_hf = sorted(set(mb_by_id) - set(hf_by_id))
    if missing_in_mb:
        logger.warning(f"Prompts missing in MB artifact: {missing_in_mb}")
    if missing_in_hf:
        logger.warning(f"Prompts missing in HF artifact: {missing_in_hf}")

    rows: list[tuple[str, dict]] = []
    fail_count = 0
    for rid in common:
        r = _compare_one(hf_by_id[rid], mb_by_id[rid])
        rows.append((rid, r))
        status = "PASS" if r["cosine"] >= args.threshold and r["top1_match"] else "FAIL"
        if status == "FAIL":
            fail_count += 1
        padded_str = f" cos@padded={r['cosine_padded']:.6f}" if "cosine_padded" in r else ""
        logger.info(
            f"{rid:>20s} cos@real={r['cosine']:.6f}{padded_str} max_abs_diff={r['max_abs_diff']:.4e} "
            f"top1_match={r['top1_match']} topk={r['topk_overlap']}/{r['topk_k']} [{status}]"
        )

    md = _format_md(rows, args.threshold, hf_artifact, mb_artifact)
    print(md)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(md)
        logger.info(f"Wrote report to {args.report}")

    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
