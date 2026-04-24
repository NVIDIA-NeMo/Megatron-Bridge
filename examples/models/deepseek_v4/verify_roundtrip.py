# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Toy-proxy HF→Megatron→HF roundtrip.

Procedure:
  1. Build reference Transformer on a small config; fill every parameter with
     randn.
  2. "Import": apply `translate_hf_to_megatron` to move each tensor into a
     Megatron-mirror state_dict, then load_state_dict into the mirror model.
  3. "Export": walk the mirror's parameters and invert the mapping.
  4. Assert every exported tensor is bit-identical to the original HF tensor
     (same dtype, same device, elementwise equal).

No GPU, no Megatron-Core, no transformers — this exercises the bridge mapping
in isolation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


# ---- minimal loader (same pattern as verify_weight_mapping.py) -------------
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PKG = _ROOT / "src" / "megatron" / "bridge" / "models" / "deepseek_v4"


def _load(mod_name: str, file_rel: str):
    spec = importlib.util.spec_from_file_location(mod_name, _PKG / file_rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


_reference = _load("deepseek_v4_reference", "modeling_deepseek_v4/reference.py")
sys.modules["megatron.bridge.models.deepseek_v4.modeling_deepseek_v4.reference"] = _reference
_mirror = _load("deepseek_v4_mirror", "modeling_deepseek_v4/megatron_mirror.py")
_bridge = _load("deepseek_v4_bridge", "deepseek_v4_bridge.py")

DeepSeekV4ModelArgs = _reference.DeepSeekV4ModelArgs
DeepSeekV4Transformer = _reference.DeepSeekV4Transformer
DeepSeekV4MegatronModel = _mirror.DeepSeekV4MegatronModel
build_translation_table = _bridge.build_translation_table


def toy_args() -> DeepSeekV4ModelArgs:
    """Tiny config covering every V4 branch (hash layer, dense/compress/index attn, MTP)."""
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
        compress_ratios=(0, 0, 4, 128, 4, 0, 4),
        index_n_heads=4,
        index_head_dim=32,
        index_topk=8,
        hc_mult=4,
    )


def _seed_random(model: torch.nn.Module, seed: int = 0) -> None:
    gen = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        if p.dtype.is_floating_point:
            p.data = torch.randn(p.shape, generator=gen, dtype=p.dtype)
        elif p.dtype == torch.int32:
            p.data = torch.randint(0, 32, p.shape, generator=gen, dtype=p.dtype)


import_hf_to_megatron = _bridge.import_hf_state_dict
export_megatron_to_hf = _bridge.export_hf_state_dict


def run() -> int:
    """Seed randomly, import to mirror, export, assert bit-identical roundtrip."""
    args = toy_args()
    torch.manual_seed(42)

    print("[build HF-side reference with random weights]")
    hf_model = DeepSeekV4Transformer(args)
    _seed_random(hf_model, seed=1)
    hf_sd_original = {k: v.detach().clone() for k, v in hf_model.state_dict().items()}
    print(f"  {len(hf_sd_original)} params seeded")

    print("[build Megatron-mirror]")
    mg_model = DeepSeekV4MegatronModel(args)
    print(f"  {len(mg_model.state_dict())} params allocated (pre-load)")

    print("[HF -> Megatron import]")
    mg_sd_imported = import_hf_to_megatron(hf_sd_original, args)
    # Load into the mirror to exercise nn.Module load path end-to-end.
    missing, unexpected = mg_model.load_state_dict(mg_sd_imported, strict=False)
    if missing or unexpected:
        print(f"  missing={len(missing)}, unexpected={len(unexpected)}")
        for k in missing[:5]:
            print(f"    missing: {k}")
        for k in unexpected[:5]:
            print(f"    unexpected: {k}")
    mg_sd_post_load = dict(mg_model.state_dict())

    # Compare tensors pre-load and post-load to confirm the mirror stores them intact.
    post_load_mismatches = []
    for k, v in mg_sd_imported.items():
        if not torch.equal(mg_sd_post_load[k], v):
            post_load_mismatches.append(k)
    if post_load_mismatches:
        print(f"  ERROR: mirror mutated {len(post_load_mismatches)} tensors on load")
        for k in post_load_mismatches[:5]:
            print(f"    - {k}")
        return 1
    print(f"  mirror holds {len(mg_sd_post_load)} tensors intact")

    print("[Megatron -> HF export]")
    hf_sd_export = export_megatron_to_hf(mg_sd_post_load, args)
    print(f"  re-exported {len(hf_sd_export)} HF keys")

    print("[key set equality]")
    orig_keys = set(hf_sd_original)
    new_keys = set(hf_sd_export)
    missing_after_rt = orig_keys - new_keys
    extra_after_rt = new_keys - orig_keys
    if missing_after_rt or extra_after_rt:
        print(f"  ERROR: key drift (missing={len(missing_after_rt)}, extra={len(extra_after_rt)})")
        for k in sorted(missing_after_rt)[:5]:
            print(f"    - {k}")
        for k in sorted(extra_after_rt)[:5]:
            print(f"    + {k}")
        return 1
    print(f"  OK: same {len(orig_keys)} keys")

    print("[elementwise equality on every tensor]")
    mismatches = []
    for k, v_orig in hf_sd_original.items():
        v_new = hf_sd_export[k]
        if v_orig.dtype != v_new.dtype:
            mismatches.append((k, "dtype", v_orig.dtype, v_new.dtype))
            continue
        if v_orig.shape != v_new.shape:
            mismatches.append((k, "shape", tuple(v_orig.shape), tuple(v_new.shape)))
            continue
        if not torch.equal(v_orig, v_new):
            max_abs = (v_orig.float() - v_new.float()).abs().max().item()
            mismatches.append((k, "values", "", f"max|Δ|={max_abs}"))
    if mismatches:
        print(f"  ERROR: {len(mismatches)} tensor(s) differ after roundtrip")
        for k, kind, lhs, rhs in mismatches[:10]:
            print(f"    {kind:>6}  {k}  {lhs} vs {rhs}")
        return 1
    print(f"  OK: all {len(hf_sd_original)} tensors elementwise equal")

    print()
    print("=" * 60)
    print("RESULT: toy roundtrip passed")
    return 0


if __name__ == "__main__":
    sys.exit(run())
