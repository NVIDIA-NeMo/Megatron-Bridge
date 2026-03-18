#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Convert HuggingFace Llama/Vicuna weights to Megatron GPTModel format.

Produces per-TP-rank .pt files that can be loaded into a Megatron
GPTModel with ``load_megatron_llm_weights()``.

Supports: Llama-2-7B, Vicuna-7B-v1.5, or any HF model with the same
LlamaForCausalLM architecture (including GQA variants).

Usage:
    # Convert (run once on any single GPU):
    python convert_hf_llama_to_megatron.py \
        --hf-model lmsys/vicuna-7b-v1.5 \
        --output /path/to/vicuna_megatron_ckpt \
        --tensor-parallel-size 4 \
        --use-te \
        --megatron-vocab-size 32256

    # Then in your training script, after model construction:
    from convert_hf_llama_to_megatron import load_megatron_llm_weights
    load_megatron_llm_weights(gpt_model, "/path/to/vicuna_megatron_ckpt", tp_rank, tp_size)
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM


# ---------------------------------------------------------------------------
# QKV interleaving for MHA / GQA
# ---------------------------------------------------------------------------


def _build_qkv_interleave_indices(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Build indices to interleave separate Q/K/V into Megatron's fused QKV layout.

    Handles both MHA (num_heads == num_kv_heads) and GQA (num_heads > num_kv_heads).

    Megatron expects per-group interleaving:
        For each KV group g:
            [Q_h0, Q_h1, ..., Q_h(ratio-1), K_g, V_g]
        where ratio = num_heads / num_kv_heads.

    When MHA (ratio=1): [Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ...].
    """
    q_dim = num_heads * head_dim
    k_dim = num_kv_heads * head_dim
    # offsets into the concatenated [Q; K; V] tensor
    k_offset = q_dim
    v_offset = q_dim + k_dim
    ratio = num_heads // num_kv_heads

    indices = []
    for g in range(num_kv_heads):
        # Q heads belonging to this KV group
        for r in range(ratio):
            h = g * ratio + r
            indices.append(torch.arange(h * head_dim, (h + 1) * head_dim))
        # K head for this group
        indices.append(torch.arange(k_offset + g * head_dim, k_offset + (g + 1) * head_dim))
        # V head for this group
        indices.append(torch.arange(v_offset + g * head_dim, v_offset + (g + 1) * head_dim))

    return torch.cat(indices)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def convert_hf_llama_to_megatron(
    hf_model_name: str = "lmsys/vicuna-7b-v1.5",
    output_path: str = "./vicuna_megatron_ckpt",
    tensor_parallel_size: int = 1,
    use_te: bool = True,
    megatron_vocab_size: int | None = None,
) -> None:
    """Download HF Llama/Vicuna weights and save as per-TP-rank Megatron .pt files.

    Args:
        hf_model_name: HuggingFace model identifier or local path.
        output_path: Directory for output checkpoint files.
        tensor_parallel_size: Target tensor parallelism size.
        use_te: If True, use Transformer Engine layer naming (fused layernorm
                inside linear_qkv / linear_fc1).
        megatron_vocab_size: If set, pad embedding/output weights to this size.
            Common for LLaVA where extra tokens are added (e.g. 32256).
    """
    print(f"Loading HuggingFace model: {hf_model_name}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=torch.float32,
    )
    hf_config = hf_model.config
    state_dict = hf_model.state_dict()

    hidden_size = hf_config.hidden_size
    num_heads = hf_config.num_attention_heads
    num_kv_heads = getattr(hf_config, "num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads
    hf_vocab_size = hf_config.vocab_size
    target_vocab_size = megatron_vocab_size or hf_vocab_size

    print(
        f"  hidden_size={hidden_size}, num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
        f"head_dim={head_dim}, hf_vocab={hf_vocab_size}, target_vocab={target_vocab_size}"
    )

    indices = _build_qkv_interleave_indices(num_heads, num_kv_heads, head_dim)

    new_state_dicts = [{"model": {}} for _ in range(tensor_parallel_size)]

    def _pad_vocab(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Zero-pad along *dim* from hf_vocab_size → target_vocab_size."""
        if target_vocab_size <= hf_vocab_size:
            return tensor
        pad_size = target_vocab_size - tensor.size(dim)
        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_size
        return torch.cat([tensor, torch.zeros(pad_shape, dtype=tensor.dtype)], dim=dim)

    for name, tensor in state_dict.items():
        new_name = ""
        new_tensor = tensor.float()
        chunk_dim = None  # dimension to split for TP

        # --- Embedding ---
        if name == "model.embed_tokens.weight":
            new_name = "embedding.word_embeddings.weight"
            new_tensor = _pad_vocab(new_tensor, dim=0)
            chunk_dim = 0  # VocabParallelEmbedding splits along vocab dim

        # --- Final RMSNorm ---
        elif name == "model.norm.weight":
            new_name = "decoder.final_layernorm.weight"

        # --- LM head ---
        elif name == "lm_head.weight":
            new_name = "output_layer.weight"
            new_tensor = _pad_vocab(new_tensor, dim=0)
            chunk_dim = 0

        # --- Transformer layers ---
        elif name.startswith("model.layers."):
            parts = name.split(".")
            layer_idx = parts[2]
            base = f"decoder.layers.{layer_idx}"
            suffix = ".".join(parts[3:])

            # --- Self-attention QKV (separate → fused, interleaved) ---
            if suffix == "self_attn.q_proj.weight":
                k_key = name.replace("q_proj", "k_proj")
                v_key = name.replace("q_proj", "v_proj")
                q = new_tensor
                k = state_dict[k_key].float()
                v = state_dict[v_key].float()
                qkv = torch.cat([q, k, v], dim=0)[indices]
                new_name = f"{base}.self_attention.linear_qkv.weight"
                new_tensor = qkv
                chunk_dim = 0
            elif suffix in ("self_attn.k_proj.weight", "self_attn.v_proj.weight"):
                continue  # fused above

            # --- Output projection ---
            elif suffix == "self_attn.o_proj.weight":
                new_name = f"{base}.self_attention.linear_proj.weight"
                chunk_dim = 1

            # --- RMSNorm (no bias for Llama) ---
            elif suffix == "input_layernorm.weight":
                if use_te:
                    new_name = f"{base}.self_attention.linear_qkv.layer_norm_weight"
                else:
                    new_name = f"{base}.input_layernorm.weight"
            elif suffix == "post_attention_layernorm.weight":
                if use_te:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_weight"
                else:
                    new_name = f"{base}.pre_mlp_layernorm.weight"

            # --- MLP: SwiGLU (gate + up → fused linear_fc1) ---
            elif suffix == "mlp.gate_proj.weight":
                up_key = name.replace("gate_proj", "up_proj")
                gate = new_tensor  # [ffn_hidden, hidden]
                up = state_dict[up_key].float()  # [ffn_hidden, hidden]
                fused = torch.cat([gate, up], dim=0)  # [2*ffn_hidden, hidden]
                new_name = f"{base}.mlp.linear_fc1.weight"
                new_tensor = fused
                chunk_dim = 0
            elif suffix == "mlp.up_proj.weight":
                continue  # fused above
            elif suffix == "mlp.down_proj.weight":
                new_name = f"{base}.mlp.linear_fc2.weight"
                chunk_dim = 1

        if new_name == "":
            print(f"  [WARN] skipping unmapped key: {name}")
            continue

        # Split for tensor parallelism
        if chunk_dim is None:
            chunks = [new_tensor] * tensor_parallel_size
        else:
            chunks = torch.chunk(new_tensor, tensor_parallel_size, dim=chunk_dim)

        for tp in range(tensor_parallel_size):
            new_state_dicts[tp]["model"][new_name] = chunks[tp].clone()

            # TE layers need _extra_state placeholders for FP8 compatibility
            if use_te:
                te_layers = ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2")
                if any(layer in new_name for layer in te_layers):
                    layer_key = new_name.split(".")[-2]
                    if layer_key in te_layers:
                        extra_key = new_name[: new_name.rfind(".") + 1] + "_extra_state"
                        new_state_dicts[tp]["model"][extra_key] = None

    # Save per-TP-rank files
    for tp in range(tensor_parallel_size):
        output_dir_tp = os.path.join(output_path, f"tp_rank_{tp:02d}")
        os.makedirs(output_dir_tp, exist_ok=True)
        output_file = os.path.join(output_dir_tp, "model_weights.pt")
        torch.save(new_state_dicts[tp], output_file)
        n_params = len(new_state_dicts[tp]["model"])
        print(f"  Saved TP rank {tp}: {output_file} ({n_params} params)")

    print(f"Conversion complete → {output_path}")


# ---------------------------------------------------------------------------
# Loading helper: load converted weights into a Megatron GPTModel
# ---------------------------------------------------------------------------


def _get_llm_tp_concat_dim(param_name: str):
    """Return the concat dimension for a TP-sharded LLM parameter, or None if replicated.

    Must match the chunk_dim logic in convert_hf_llama_to_megatron().
    """
    # Column-parallel (chunk_dim=0)
    if "linear_qkv.weight" in param_name:
        return 0
    if "linear_fc1.weight" in param_name:
        return 0
    if "word_embeddings.weight" in param_name:
        return 0
    if "output_layer.weight" in param_name:
        return 0
    # Row-parallel (chunk_dim=1)
    if "linear_proj.weight" in param_name:
        return 1
    if "linear_fc2.weight" in param_name:
        return 1
    # Everything else is replicated (layernorm, final_layernorm)
    return None


def load_megatron_llm_weights(
    gpt_model: torch.nn.Module,
    ckpt_dir: str,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> None:
    """Load converted LLM weights into a Megatron GPTModel.

    Supports loading from a checkpoint saved with a *different* TP size.
    When the checkpoint TP size differs from the model's TP size, all shards
    are merged and optionally re-split.

    Args:
        gpt_model: The Megatron GPTModel instance.
        ckpt_dir: Directory produced by ``convert_hf_llama_to_megatron()``.
        tp_rank: This rank's tensor-parallel index.
        tp_size: Total tensor-parallel size (the *model's* TP, not the ckpt's).
    """
    # Discover how many TP shards exist in the checkpoint
    ckpt_tp_dirs = sorted(
        d for d in os.listdir(ckpt_dir) if d.startswith("tp_rank_") and os.path.isdir(os.path.join(ckpt_dir, d))
    )
    ckpt_tp_size = len(ckpt_tp_dirs)
    if ckpt_tp_size == 0:
        raise FileNotFoundError(f"No tp_rank_* directories found in {ckpt_dir}")

    if ckpt_tp_size == tp_size:
        # Simple case: sizes match, load the single shard for this rank
        ckpt_file = os.path.join(ckpt_dir, f"tp_rank_{tp_rank:02d}", "model_weights.pt")
        saved = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        state_dict = {k: v for k, v in saved["model"].items() if v is not None}
    else:
        # Merge all TP shards into a single unsharded state dict
        print(f"  Merging {ckpt_tp_size} TP shards into TP={tp_size} model")
        all_shards = []
        for tp_dir in ckpt_tp_dirs:
            f = os.path.join(ckpt_dir, tp_dir, "model_weights.pt")
            saved = torch.load(f, map_location="cpu", weights_only=True)
            all_shards.append({k: v for k, v in saved["model"].items() if v is not None})

        state_dict = {}
        for key in all_shards[0]:
            concat_dim = _get_llm_tp_concat_dim(key)
            if concat_dim is not None:
                state_dict[key] = torch.cat([s[key] for s in all_shards], dim=concat_dim)
            else:
                state_dict[key] = all_shards[0][key]

        # If model TP > 1, re-split to get the right shard for this rank
        if tp_size > 1:
            new_sd = {}
            for key, tensor in state_dict.items():
                concat_dim = _get_llm_tp_concat_dim(key)
                if concat_dim is not None:
                    chunks = torch.chunk(tensor, tp_size, dim=concat_dim)
                    new_sd[key] = chunks[tp_rank].clone()
                else:
                    new_sd[key] = tensor
            state_dict = new_sd

    incompatible = gpt_model.load_state_dict(state_dict, strict=False)
    unexpected = [k for k in incompatible.unexpected_keys if "_extra_state" not in k]
    missing = [k for k in incompatible.missing_keys if "_extra_state" not in k]
    if unexpected or missing:
        raise RuntimeError(f"State dict mismatch.\n  Missing:    {missing}\n  Unexpected: {unexpected}")
    print(f"Loaded LLM weights from {ckpt_dir} (ckpt_tp={ckpt_tp_size}, model tp_rank={tp_rank}/{tp_size})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace Llama/Vicuna to Megatron GPTModel format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default="lmsys/vicuna-7b-v1.5",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for Megatron checkpoint files",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Target tensor parallel size",
    )
    parser.add_argument(
        "--use-te",
        action="store_true",
        help="Use Transformer Engine layer naming (required when using TE specs)",
    )
    parser.add_argument(
        "--megatron-vocab-size",
        type=int,
        default=None,
        help="Pad embedding/output to this vocab size (e.g. 32256 for LLaVA)",
    )
    args = parser.parse_args()

    convert_hf_llama_to_megatron(
        hf_model_name=args.hf_model,
        output_path=args.output,
        tensor_parallel_size=args.tensor_parallel_size,
        use_te=args.use_te,
        megatron_vocab_size=args.megatron_vocab_size,
    )
    print("Done.")
