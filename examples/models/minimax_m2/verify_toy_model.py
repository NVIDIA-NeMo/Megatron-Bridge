# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Verify MiniMax-M2 HF <-> Megatron conversion using real pretrained weights.

Extracts the first N layers from the full MiniMax-M2 FP8 checkpoint,
dequantizes to bf16 using block-wise scale factors, and saves a clean
bf16 model.  Then runs compare.py to verify conversion + forward pass.

Usage:
    uv run python examples/models/minimax_m2/verify_toy_model.py
    uv run python examples/models/minimax_m2/verify_toy_model.py --tp 2
    uv run python examples/models/minimax_m2/verify_toy_model.py --num-layers 1
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch


HF_MODEL_ID = "MiniMaxAI/MiniMax-M2"
DEFAULT_NUM_LAYERS = 1
FP8_BLOCK_SIZE = 128


def _dequant_blockwise(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Block-wise FP8 dequantization: weight_bf16 = fp8_val * scale_inv per block."""
    if weight.ndim == 1:
        return weight.float().to(torch.bfloat16)

    M, N = weight.shape
    B = FP8_BLOCK_SIZE
    w = weight.float()
    out = torch.empty_like(w)

    sM, sN = scale_inv.shape
    for bi in range(sM):
        for bj in range(sN):
            r0, r1 = bi * B, min((bi + 1) * B, M)
            c0, c1 = bj * B, min((bj + 1) * B, N)
            out[r0:r1, c0:c1] = w[r0:r1, c0:c1] * scale_inv[bi, bj]

    return out.to(torch.bfloat16)


def create_toy_model(output_dir: str, num_layers: int = DEFAULT_NUM_LAYERS):
    """Extract first N layers from MiniMax-M2, dequantize FP8 -> bf16, save."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    from transformers import AutoConfig, AutoTokenizer

    print(f"Creating sliced MiniMax-M2 ({num_layers} layers) with FP8 dequantization...")

    config = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    print(f"  Original: {config.num_hidden_layers} layers, {config.num_local_experts} experts")

    cache_dir = snapshot_download(HF_MODEL_ID, allow_patterns=["*.json", "*.safetensors"])
    print(f"  Cache: {cache_dir}")

    with open(os.path.join(cache_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    layer_re = re.compile(r"^model\.layers\.(\d+)\.")

    needed = set()
    for key in weight_map:
        m = layer_re.match(key)
        if m is None or int(m.group(1)) < num_layers:
            needed.add(key)

    for key in list(needed):
        sinv = key + "_scale_inv"
        if sinv in weight_map:
            needed.add(sinv)

    files_to_keys = defaultdict(list)
    for key in needed:
        files_to_keys[weight_map[key]].append(key)

    print(f"  Loading {len(needed)} tensors from {len(files_to_keys)} shard(s)...")
    raw = {}
    for fn, keys in sorted(files_to_keys.items()):
        print(f"    {fn} ({len(keys)} tensors)")
        data = load_file(os.path.join(cache_dir, fn))
        for k in keys:
            raw[k] = data[k]

    state_dict = {}
    n_dequant = 0
    for key, t in raw.items():
        if key.endswith("_scale_inv"):
            continue
        if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            sinv_key = key + "_scale_inv"
            if sinv_key in raw and t.ndim == 2:
                t = _dequant_blockwise(t, raw[sinv_key])
                n_dequant += 1
            else:
                t = t.float().to(torch.bfloat16)
        state_dict[key] = t

    print(f"  Dequantized {n_dequant} FP8 tensors (block {FP8_BLOCK_SIZE}x{FP8_BLOCK_SIZE})")

    os.makedirs(output_dir, exist_ok=True)

    cfg = config.__class__.from_dict(config.to_dict())
    cfg.num_hidden_layers = num_layers
    for attr in ("quantization_config", "auto_map"):
        try:
            delattr(cfg, attr)
        except AttributeError:
            pass
    cfg.torch_dtype = "bfloat16"
    if not hasattr(cfg, "rope_parameters") or cfg.rope_parameters is None:
        cfg.rope_parameters = {
            "rope_type": "default",
            "rope_theta": getattr(cfg, "rope_theta", 10000.0),
        }
    cfg.save_pretrained(output_dir)

    AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True).save_pretrained(output_dir)

    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))

    n_params = sum(t.numel() for t in state_dict.values())
    print(f"  Params: {n_params:,} | Layers: {num_layers} | Experts: {config.num_local_experts}")
    _verify_qk_norm(state_dict, cfg)
    _print_shapes(state_dict)
    print(f"  Saved to: {output_dir}")


def _verify_qk_norm(sd, cfg):
    hd, nh, nkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
    for i in range(cfg.num_hidden_layers):
        qk = f"model.layers.{i}.self_attn.q_norm.weight"
        kk = f"model.layers.{i}.self_attn.k_norm.weight"
        assert qk in sd, f"Missing {qk}"
        assert kk in sd, f"Missing {kk}"
        assert sd[qk].shape[0] == nh * hd, f"q_norm shape {sd[qk].shape[0]} != {nh * hd}"
        assert sd[kk].shape[0] == nkv * hd, f"k_norm shape {sd[kk].shape[0]} != {nkv * hd}"
    print(f"  QK norm OK: q=[{nh}*{hd}={nh * hd}], k=[{nkv}*{hd}={nkv * hd}]")


def _print_shapes(sd):
    print("  Key shapes (layer 0):")
    for k in [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.block_sparse_moe.gate.weight",
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
    ]:
        if k in sd:
            print(f"    {k}: {list(sd[k].shape)} {sd[k].dtype}")


def run_compare(model_dir: str, tp: int, pp: int, ep: int):
    """Run compare.py to verify HF <-> Megatron round-trip conversion."""
    script = str(
        Path(__file__).resolve().parent.parent.parent / "conversion" / "compare_hf_and_megatron" / "compare.py"
    )
    nproc = tp * pp * ep
    args = ["--hf_model_path", model_dir, "--prompt", "Hello", "--trust_remote_code"]

    if nproc == 1:
        cmd = [sys.executable, script] + args
    else:
        cmd = (
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={nproc}",
                script,
            ]
            + args
            + ["--tp", str(tp), "--pp", str(pp), "--ep", str(ep)]
        )

    print(f"\n{'=' * 60}")
    print(f"Running compare.py  TP={tp} PP={pp} EP={ep} (nproc={nproc})")
    print(f"{'=' * 60}")
    print(f"  cmd: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent.parent.parent))
    if result.returncode != 0:
        print(f"\n[FAIL] compare.py exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n[OK] TP={tp} PP={pp} EP={ep} comparison passed")


def main():
    """CLI entrypoint: create a toy model and run compare.py."""
    parser = argparse.ArgumentParser(description="Verify MiniMax-M2 conversion with real weights")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--force-recreate", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir or os.path.join(tempfile.gettempdir(), "minimax_m2_toy")

    if args.force_recreate and os.path.exists(model_dir):
        import shutil

        shutil.rmtree(model_dir)

    if not os.path.exists(os.path.join(model_dir, "config.json")):
        create_toy_model(model_dir, num_layers=args.num_layers)
    else:
        print(f"Reusing existing model at: {model_dir}")

    run_compare(model_dir, args.tp, args.pp, args.ep)


if __name__ == "__main__":
    main()
