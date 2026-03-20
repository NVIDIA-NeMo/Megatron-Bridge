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
HuggingFace <-> Megatron-Core roundtrip conversion for GLM-5 (MoE + MLA + DSA).

Single-GPU (toy model):
    uv run python examples/models/glm5/glm5_roundtrip.py \\
        --hf-model-id /path/to/glm5 \\
        --output-dir /tmp/glm5_out

Multi-GPU (TP=2, PP=2, EP=2):
    uv run python -m torch.distributed.run --nproc_per_node=8 \\
        examples/conversion/hf_megatron_roundtrip_multi_gpu.py \\
        --hf-model-id zai-org/GLM-5 \\
        --output-dir /tmp/glm5_out \\
        --tp 2 --pp 2 --ep 2

Creating a toy model for testing:
    uv run python examples/models/glm5/glm5_roundtrip.py --create-toy-model \\
        --toy-model-dir /tmp/glm5_toy \\
        --output-dir /tmp/glm5_out

Requirements: transformers >= 5.2.0
"""

import argparse
import json
import os
from pathlib import Path

import torch
from rich.console import Console

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion import weights_verification_table
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo


console = Console()

# Default real model path; override with --hf-model-id
DEFAULT_HF_MODEL_ID = "zai-org/GLM-5"

# Toy model config for quick testing (reduced dimensions)
GLM5_TOY_CONFIG = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "model_type": "glm_moe_dsa",
    "hidden_size": 1024,
    "intermediate_size": 2048,
    "moe_intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 64,
    "qk_head_dim": 128,
    "qk_nope_head_dim": 96,
    "qk_rope_head_dim": 32,
    "v_head_dim": 128,
    "index_head_dim": 128,
    "index_n_heads": 8,
    "index_topk": 256,
    "indexer_rope_interleave": True,
    "q_lora_rank": 256,
    "kv_lora_rank": 128,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "moe_layer_freq": 1,
    "first_k_dense_replace": 1,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "mlp_layer_types": ["dense", "sparse"],
    "max_position_embeddings": 8192,
    "rope_interleave": True,
    "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
    "hidden_act": "silu",
    "rms_norm_eps": 1e-05,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "vocab_size": 154880,
    "bos_token_id": 0,
    "eos_token_id": [154820, 154827, 154829],
    "pad_token_id": 154820,
    "ep_size": 1,
    "num_nextn_predict_layers": 1,
    "initializer_range": 0.02,
    "tie_word_embeddings": False,
    "use_cache": True,
    "dtype": "bfloat16",
    "pretraining_tp": 1,
    "transformers_version": "5.2.0.dev0",
}


def create_toy_model(toy_dir: Path) -> None:
    """Create a small GLM-5 toy model with random weights for local testing."""
    from transformers import AutoConfig, AutoTokenizer, GlmMoeDsaForCausalLM

    toy_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Creating GLM-5 toy model at {toy_dir}...[/bold]")

    config = AutoConfig.from_pretrained(DEFAULT_HF_MODEL_ID)
    for key, value in GLM5_TOY_CONFIG.items():
        setattr(config, key, value)
    config.torch_dtype = torch.bfloat16

    model = GlmMoeDsaForCausalLM(config).bfloat16()
    # e_score_correction_bias must be fp32
    for _name, buf in model.named_buffers():
        if "e_score_correction_bias" in _name:
            buf.data = buf.data.to(torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_HF_MODEL_ID)
    tokenizer.save_pretrained(toy_dir)
    model.save_pretrained(toy_dir, safe_serialization=True)

    # Overwrite config.json with minimal toy config so AutoConfig loads correctly
    with open(toy_dir / "config.json", "w") as f:
        json.dump(GLM5_TOY_CONFIG, f, indent=2)

    console.print(f"[green]Toy model saved to {toy_dir}[/green]")


def main(
    hf_model_id: str = DEFAULT_HF_MODEL_ID,
    output_dir: str | None = None,
    trust_remote_code: bool | None = None,
) -> None:
    """Perform HF -> Megatron -> HF roundtrip for GLM-5."""
    model_name = Path(hf_model_id).name
    save_path = os.path.join(output_dir, model_name) if output_dir else model_name

    console.print(f"[bold]Loading GLM-5 bridge from {hf_model_id}...[/bold]")
    bridge = AutoBridge.from_hf_pretrained(
        hf_model_id,
        trust_remote_code=is_safe_repo(
            trust_remote_code=trust_remote_code,
            hf_path=hf_model_id,
        ),
    )
    megatron_model = bridge.to_megatron_model(wrap_with_ddp=False)
    console.print(weights_verification_table(bridge, megatron_model))

    console.print(f"[bold]Saving converted HF checkpoint to {save_path}...[/bold]")
    bridge.save_hf_pretrained(megatron_model, save_path)
    console.print(f"[green]Roundtrip complete. Output at: {save_path}[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLM-5 HF <-> Megatron roundtrip conversion")
    parser.add_argument("--hf-model-id", type=str, default=DEFAULT_HF_MODEL_ID)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--create-toy-model",
        action="store_true",
        help="Create a small toy model for testing (no GPU required for creation).",
    )
    parser.add_argument(
        "--toy-model-dir",
        type=str,
        default="/tmp/glm5_toy",
        help="Directory to save/load the toy model (used with --create-toy-model).",
    )
    args = parser.parse_args()

    if args.create_toy_model:
        toy_dir = Path(args.toy_model_dir)
        create_toy_model(toy_dir)
        # Use toy model for the roundtrip
        args.hf_model_id = str(toy_dir)

    main(args.hf_model_id, args.output_dir, args.trust_remote_code)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
