# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Verify DeepSeek-V4 HF ↔ Megatron weight-name + shape parity on a toy config.

Usage (toy, no GPU, runs anywhere):
    uv run python examples/models/deepseek_v4/verify_weight_mapping.py

With `--hf-index <path/to/model.safetensors.index.json>` the real Flash
checkpoint's key list is cross-checked against what the reference port
produces (sizes are not fetched).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch


# Load the three deepseek_v4 modules by file path so this script can run in a
# bare Python env (the top-level `megatron.bridge` package pulls in heavy
# Megatron-Core / TransformerEngine deps that aren't available on every box).
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PKG = _ROOT / "src" / "megatron" / "bridge" / "models" / "deepseek_v4"


def _load(mod_name: str, file_rel: str):
    spec = importlib.util.spec_from_file_location(mod_name, _PKG / file_rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


_reference = _load(
    "deepseek_v4_reference",
    "modeling_deepseek_v4/reference.py",
)
DeepSeekV4ModelArgs = _reference.DeepSeekV4ModelArgs
DeepSeekV4Transformer = _reference.DeepSeekV4Transformer

# The sibling modules import `from megatron.bridge...deepseek_v4.modeling_deepseek_v4.reference`;
# register our hand-loaded reference under that package path so the imports resolve
# without needing to build the full `megatron.bridge` package.
sys.modules.setdefault("megatron.bridge.models.deepseek_v4.modeling_deepseek_v4.reference", _reference)

_mirror = _load("deepseek_v4_mirror", "modeling_deepseek_v4/megatron_mirror.py")
DeepSeekV4MegatronModel = _mirror.DeepSeekV4MegatronModel

_bridge = _load("deepseek_v4_bridge", "deepseek_v4_bridge.py")
build_translation_table = _bridge.build_translation_table
translate_hf_to_megatron = _bridge.translate_hf_to_megatron


def toy_args() -> DeepSeekV4ModelArgs:
    """Small enough to instantiate on CPU in seconds; exercises every branch:
    hash layers present, dense + compressed + indexed attention layers, MTP."""
    return DeepSeekV4ModelArgs(
        max_batch_size=1,
        max_seq_len=64,
        vocab_size=256,
        dim=64,
        moe_inter_dim=64,
        n_layers=6,
        n_hash_layers=1,
        n_mtp_layers=1,
        n_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        q_lora_rank=128,
        head_dim=64,
        rope_head_dim=16,
        o_groups=2,
        o_lora_rank=128,
        window_size=16,
        # Mix of 0 (pure sliding), 4 (compress+index), and 128 (compress-only).
        # Length is n_layers + n_mtp_layers (matches real config convention).
        compress_ratios=(0, 0, 4, 128, 4, 0, 4),
        index_n_heads=4,
        index_head_dim=32,
        index_topk=8,
        hc_mult=4,
    )


def run(hf_index: Path | None) -> int:
    """Build both toy models and check name/shape/dtype parity (incl. fanouts)."""
    args = toy_args()
    print("[toy config]")
    for k, v in asdict(args).items():
        print(f"  {k}: {v}")
    print()

    print("[build reference (HF-side) model]")
    with torch.device("meta"):
        hf_model = DeepSeekV4Transformer(args)
    hf_keys = set(hf_model.state_dict().keys())
    print(f"  {len(hf_keys)} params")

    print("[build megatron-mirror model]")
    with torch.device("meta"):
        mg_model = DeepSeekV4MegatronModel(args)
    mg_keys = set(mg_model.state_dict().keys())
    print(f"  {len(mg_keys)} params")
    print()

    print("[enumerate expected HF keys via build_translation_table]")
    table = build_translation_table(args)
    enum_hf = {e.hf for e in table}
    enum_mg = {e.megatron for e in table}

    # Cross-checks
    status = 0
    missing_from_enum = hf_keys - enum_hf
    extra_in_enum = enum_hf - hf_keys
    if missing_from_enum:
        status = 1
        print(f"  ERROR: {len(missing_from_enum)} HF key(s) emitted by reference but NOT in build_translation_table:")
        for k in sorted(missing_from_enum)[:20]:
            print(f"    - {k}")
    if extra_in_enum:
        status = 1
        print(f"  ERROR: {len(extra_in_enum)} key(s) in build_translation_table but NOT emitted by reference:")
        for k in sorted(extra_in_enum)[:20]:
            print(f"    - {k}")

    mg_missing = enum_mg - mg_keys
    mg_extra = mg_keys - enum_mg
    if mg_missing:
        status = 1
        print(f"  ERROR: {len(mg_missing)} Megatron key(s) expected by table but NOT emitted by megatron_mirror:")
        for k in sorted(mg_missing)[:20]:
            print(f"    - {k}")
    if mg_extra:
        status = 1
        print(f"  ERROR: {len(mg_extra)} Megatron key(s) emitted by megatron_mirror but NOT expected by table:")
        for k in sorted(mg_extra)[:20]:
            print(f"    - {k}")

    print()
    print("[shape parity — honours 1→N fanouts]")
    hf_sd = hf_model.state_dict()
    mg_sd = mg_model.state_dict()
    shape_mismatches: list[tuple[object, tuple, tuple]] = []
    dtype_mismatches: list[tuple[object, object, object]] = []
    # Group table entries by HF key — multi-entry groups are fanouts.
    from collections import defaultdict

    by_hf: dict[str, list] = defaultdict(list)
    for e in table:
        by_hf[e.hf].append(e)

    for hf_key, group in by_hf.items():
        hf_t = hf_sd[hf_key]
        if len(group) == 1:
            mg_t = mg_sd[group[0].megatron]
            if tuple(hf_t.shape) != tuple(mg_t.shape):
                shape_mismatches.append((group[0], tuple(hf_t.shape), tuple(mg_t.shape)))
            if hf_t.dtype != mg_t.dtype:
                dtype_mismatches.append((group[0], hf_t.dtype, mg_t.dtype))
        else:
            # fanout: expect total numel(mg pieces) == numel(hf)
            mg_ts = [mg_sd[e.megatron] for e in group]
            total = sum(t.numel() for t in mg_ts)
            if total != hf_t.numel():
                shape_mismatches.append((group[0], tuple(hf_t.shape), f"fanout {len(group)} pieces, total={total}"))
            for e, mg_t in zip(group, mg_ts):
                if hf_t.dtype != mg_t.dtype:
                    dtype_mismatches.append((e, hf_t.dtype, mg_t.dtype))

    if shape_mismatches:
        status = 1
        print(f"  ERROR: {len(shape_mismatches)} shape mismatch(es):")
        for e, hs, ms in shape_mismatches[:20]:
            print(f"    {e.hf}  {hs}  vs  {e.megatron}  {ms}")
    else:
        print(f"  OK: all {len(by_hf)} HF params covered (with fanouts honoured)")

    print()
    print("[dtype parity]")
    if dtype_mismatches:
        status = 1
        print(f"  ERROR: {len(dtype_mismatches)} dtype mismatch(es):")
        for e, hd, md in dtype_mismatches[:20]:
            print(f"    {e.hf}  {hd}  vs  {e.megatron}  {md}")
    else:
        print(f"  OK: all {len(table)} (megatron-side) entries dtype-aligned")

    if hf_index is not None:
        print()
        print(f"[cross-check against real checkpoint index: {hf_index}]")
        idx = json.loads(hf_index.read_text())
        real_keys = set(idx["weight_map"].keys())
        status |= _cross_check_real_keys(real_keys)

    print()
    print("=" * 60)
    if status == 0:
        print("RESULT: all checks passed")
    else:
        print("RESULT: one or more checks failed")
    return status


def _cross_check_real_keys(real_keys: set[str]) -> int:
    """Translate every real HF key via `translate_hf_to_megatron`. Collect any
    untranslatable keys as gaps in the mapping. Returns non-zero if any gaps."""
    status = 0
    unmapped: dict[str, int] = {}
    translated = 0
    mg_keys: set[str] = set()
    for k in real_keys:
        try:
            mg_keys.add(translate_hf_to_megatron(k))
            translated += 1
        except KeyError:
            # Coarse-bucket so we don't print 200 identical-shape entries.
            bucket = _bucket_key(k)
            unmapped[bucket] = unmapped.get(bucket, 0) + 1

    total = len(real_keys)
    print(f"  translated {translated}/{total} real keys -> {len(mg_keys)} unique megatron keys")
    if unmapped:
        status = 1
        print(f"  unmapped buckets ({len(unmapped)}):")
        for bucket, count in sorted(unmapped.items(), key=lambda kv: -kv[1])[:20]:
            print(f"    [{count:>5}x]  {bucket}")
    return status


def _bucket_key(k: str) -> str:
    """Replace all digit runs with `*` so many similar keys collapse to one."""
    import re as _re

    return _re.sub(r"\d+", "*", k)


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--hf-index",
        type=Path,
        default=None,
        help="Path to model.safetensors.index.json from the real DSV4-Flash checkpoint.",
    )
    args = p.parse_args()
    return run(args.hf_index)


if __name__ == "__main__":
    sys.exit(main())
