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
Create a toy MiniMax-M2 model and verify HF ↔ Megatron conversion + forward pass.

The toy model is saved to /tmp/minimax_m2_toy and reused across runs.

Usage:
    uv run python examples/models/minimax_m2/verify_toy_model.py
    uv run python examples/models/minimax_m2/verify_toy_model.py --tp 2
    uv run python examples/models/minimax_m2/verify_toy_model.py --ep 2
    uv run python examples/models/minimax_m2/verify_toy_model.py --tp 2 --ep 2
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


HF_MODEL_ID = "MiniMaxAI/MiniMax-M2"


def create_toy_model(output_dir: str):
    """Create and save a toy MiniMax-M2 model with random weights."""
    from transformers import AutoTokenizer, MiniMaxM2Config, MiniMaxM2ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    vocab_size = len(tokenizer)
    print(f"  Tokenizer vocab_size: {vocab_size}")

    config_dict = {
        "architectures": ["MiniMaxM2ForCausalLM"],
        "model_type": "minimax_m2",
        "hidden_size": 512,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "hidden_act": "silu",
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-06,
        "rope_theta": 5000000,
        "rotary_dim": 32,
        "vocab_size": vocab_size,
        "tie_word_embeddings": False,
        "attention_dropout": 0.0,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "scoring_func": "sigmoid",
        "use_routing_bias": True,
        "use_qk_norm": True,
        "qk_norm_type": "per_layer",
        "router_aux_loss_coef": 0.001,
        "router_jitter_noise": 0.0,
        "output_router_logits": False,
        "torch_dtype": "bfloat16",
    }

    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating toy MiniMax-M2 model at {output_dir} ...")
    config = MiniMaxM2Config(**config_dict)
    config.torch_dtype = torch.bfloat16
    model = MiniMaxM2ForCausalLM(config).bfloat16()

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    param_count = sum(p.numel() for p in model.parameters())
    print(
        f"  Params: {param_count:,} | Experts: {config_dict['num_local_experts']} | Top-K: {config_dict['num_experts_per_tok']}"
    )
    print(f"  Saved to: {output_dir}")


def run_compare(model_dir: str, tp: int, pp: int, ep: int):
    """Run compare.py against the toy model."""
    script = str(
        Path(__file__).resolve().parent.parent.parent / "conversion" / "compare_hf_and_megatron" / "compare.py"
    )

    nproc = tp * pp * ep
    prompt = "Hello"

    if nproc == 1:
        cmd = [
            sys.executable,
            script,
            "--hf_model_path",
            model_dir,
            "--prompt",
            prompt,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            script,
            "--hf_model_path",
            model_dir,
            "--prompt",
            prompt,
            "--tp",
            str(tp),
            "--pp",
            str(pp),
            "--ep",
            str(ep),
        ]

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
    """Verify MiniMax-M2 toy model conversion + forward pass."""
    parser = argparse.ArgumentParser(description="Verify MiniMax-M2 toy model conversion + forward pass")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Reuse an existing toy model directory instead of creating a new one",
    )
    args = parser.parse_args()

    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = os.path.join(tempfile.gettempdir(), "minimax_m2_toy")

    if not os.path.exists(os.path.join(model_dir, "config.json")):
        create_toy_model(model_dir)
    else:
        print(f"Reusing existing toy model at: {model_dir}")

    run_compare(model_dir, args.tp, args.pp, args.ep)


if __name__ == "__main__":
    main()
