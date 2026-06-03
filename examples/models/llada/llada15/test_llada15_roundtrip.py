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

"""Level-1 state-dict round-trip test for LLaDA1.5.

HF → Megatron in-memory → re-export to HF via :py:meth:`AutoBridge.export_hf_weights`,
then compare each exported tensor against the original HF weight tensor-by-tensor.

For round-trip conversions there is no floating-point arithmetic, only reshape /
concat / split / transpose, so the result should be **exact** (max_diff = 0.0).

Any per-parameter mismatch indicates a bug in
:py:class:`LLaDA15Bridge.mapping_registry` (e.g. wrong QKV ordering, swapped
gate/up, transposed projection, etc.) that the bf16 forward parity test can
mask.

Usage::

    PYTHONPATH=/opt/Megatron-Bridge/src python3 \\
        examples/models/llada/llada15/test_llada15_roundtrip.py \\
        --hf-path /path/to/huggingface/hub/models--GSAI-ML--LLaDA-1.5/snapshots/<commit-hash>
"""

import argparse
import os
from collections import defaultdict

import torch
import torch.distributed as dist

from megatron.bridge import AutoBridge

# Side effect: registers LLaDA15Bridge with AutoBridge.
from megatron.bridge.diffusion.conversion.llada15 import llada15_bridge  # noqa: F401


def setup_distributed_single_gpu():
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", world_size=1, rank=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--atol", type=float, default=1e-6, help="Per-tensor allclose tolerance.")
    parser.add_argument("--show-first-n", type=int, default=10, help="Print details for the first N mismatches.")
    args = parser.parse_args()

    setup_distributed_single_gpu()

    print(f"Loading HF model: {args.hf_path}")
    bridge = AutoBridge.from_hf_pretrained(args.hf_path, trust_remote_code=True)

    print("Building Megatron model and loading weights...")
    megatron_model = bridge.to_megatron_model(wrap_with_ddp=False)

    # ------------------------------------------------------------------
    # Re-export the Megatron model back to HF format and diff each weight.
    # ------------------------------------------------------------------
    n_match = 0
    n_mismatch = 0
    mismatches = []
    by_category = defaultdict(lambda: [0, 0])  # category -> [match, mismatch]

    def categorize(name: str) -> str:
        if "q_proj" in name or "k_proj" in name or "v_proj" in name:
            return "qkv"
        if "attn_out" in name:
            return "attn_out"
        if "ff_proj" in name or "up_proj" in name:
            return "mlp_gate_up"
        if "ff_out" in name and "transformer.ff_out" not in name:
            return "mlp_down"
        if "attn_norm" in name:
            return "attn_norm"
        if "ff_norm" in name:
            return "ff_norm"
        if "ln_f" in name:
            return "final_norm"
        if "wte" in name:
            return "embed"
        if "transformer.ff_out" in name:
            return "lm_head"
        return "other"

    for name, mc_tensor in bridge.export_hf_weights(megatron_model, show_progress=True):
        hf_tensor = bridge.hf_pretrained.state[name]
        hf_tensor = hf_tensor.to(mc_tensor.device, mc_tensor.dtype)
        max_diff = (mc_tensor - hf_tensor).abs().max().item()
        match = torch.allclose(mc_tensor, hf_tensor, atol=args.atol, rtol=0)
        cat = categorize(name)
        if match:
            n_match += 1
            by_category[cat][0] += 1
        else:
            n_mismatch += 1
            by_category[cat][1] += 1
            mismatches.append((name, tuple(mc_tensor.shape), max_diff))

    print("\n=== Round-trip result ===")
    print(f"Matched: {n_match}, Mismatched: {n_mismatch}")
    print("Per-category breakdown (match/total):")
    for cat in sorted(by_category):
        m, mm = by_category[cat]
        print(f"  {cat}: {m}/{m + mm}")

    if mismatches:
        print(f"\nFirst {min(args.show_first_n, len(mismatches))} mismatches:")
        for name, shape, mdiff in mismatches[: args.show_first_n]:
            print(f"  {name}  shape={shape}  max_diff={mdiff:.6e}")

    verdict = "PASS" if n_mismatch == 0 else "FAIL"
    print(f"\nverdict: {verdict}")
    if n_mismatch > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    if dist.is_initialized():
        dist.destroy_process_group()
