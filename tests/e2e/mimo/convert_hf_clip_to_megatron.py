#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Convert HuggingFace CLIP ViT-L/14-336 weights to Megatron format.

Produces per-TP-rank .pt files that can be loaded into a Megatron
CLIPViTModel with ``load_megatron_clip_weights()``.

Usage:
    python convert_hf_clip_to_megatron.py \
        --output /path/to/clip_megatron_ckpt \
        --tensor-parallel-size 4 \
        --use-te

    # Then in your training script, after model construction:
    from convert_hf_clip_to_megatron import load_megatron_clip_weights
    load_megatron_clip_weights(clip_vit_model, "/path/to/clip_megatron_ckpt", tp_rank, tp_size)
"""

import argparse
import os

import torch
from transformers import CLIPVisionModel


# ---------------------------------------------------------------------------
# Key mapping: HuggingFace → Megatron
# ---------------------------------------------------------------------------


def _build_qkv_interleave_indices(hidden_dim: int = 1024, num_heads: int = 16) -> torch.Tensor:
    """Build indices to interleave separate Q/K/V into Megatron's fused QKV layout.

    Megatron expects: [Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ...]
    HuggingFace stores: [Q_all, K_all, V_all] when concatenated.
    """
    kv_channels = hidden_dim // num_heads
    indices = []
    for i in range(num_heads):
        lb = i * kv_channels
        ub = (i + 1) * kv_channels
        indices.append(torch.arange(lb, ub))  # Q for head i
        indices.append(torch.arange(hidden_dim + lb, hidden_dim + ub))  # K for head i
        indices.append(torch.arange(2 * hidden_dim + lb, 2 * hidden_dim + ub))  # V for head i
    return torch.cat(indices)


def convert_hf_clip_to_megatron(
    hf_model_name: str = "openai/clip-vit-large-patch14-336",
    output_path: str = "./clip_megatron_ckpt",
    tensor_parallel_size: int = 1,
    use_te: bool = True,
) -> None:
    """Download HF CLIP weights and save as per-TP-rank Megatron .pt files.

    Args:
        hf_model_name: HuggingFace model identifier.
        output_path: Directory for output checkpoint files.
        tensor_parallel_size: Target tensor parallelism size.
        use_te: If True, use Transformer Engine layer naming (fused layernorm
                inside linear_qkv / linear_fc1).
    """
    print(f"Loading HuggingFace model: {hf_model_name}")
    hf_model = CLIPVisionModel.from_pretrained(hf_model_name)
    state_dict = hf_model.state_dict()

    hidden_dim = hf_model.config.hidden_size  # 1024
    num_heads = hf_model.config.num_attention_heads  # 16
    indices = _build_qkv_interleave_indices(hidden_dim, num_heads)

    new_state_dicts = [{"model": {}} for _ in range(tensor_parallel_size)]

    for name, tensor in state_dict.items():
        # Skip post_layernorm (not used by Megatron CLIPViTModel)
        if "post_layernorm" in name:
            continue

        new_name = ""
        new_tensor = tensor.float()  # convert to fp32 for saving
        chunk_dim = None  # dimension to chunk for TP

        # --- Embeddings ---
        if name == "vision_model.embeddings.class_embedding":
            new_name = "class_token"
            new_tensor = new_tensor.unsqueeze(0).unsqueeze(0)  # [hidden] → [1, 1, hidden]
        elif name == "vision_model.embeddings.position_embedding.weight":
            new_name = "position_embeddings.weight"
        elif name == "vision_model.embeddings.patch_embedding.weight":
            new_name = "conv1.weight"
        elif name == "vision_model.embeddings.patch_embedding.bias":
            # Megatron CLIPViTModel (clip subtype) uses conv_bias=False, skip
            continue

        # --- Pre-LayerNorm ---
        elif name == "vision_model.pre_layrnorm.weight" or name == "vision_model.pre_layernorm.weight":
            new_name = "ln_pre.weight"
        elif name == "vision_model.pre_layrnorm.bias" or name == "vision_model.pre_layernorm.bias":
            new_name = "ln_pre.bias"

        # --- Encoder layers ---
        elif "vision_model.encoder.layers." in name:
            parts = name.split(".")
            layer_idx = parts[3]
            base = f"decoder.layers.{layer_idx}"
            suffix = ".".join(parts[4:])  # e.g. "self_attn.q_proj.weight"

            # Self-attention QKV (separate → fused)
            if suffix == "self_attn.q_proj.weight":
                # Collect q, k, v and fuse them
                k_name = name.replace("q_proj", "k_proj")
                v_name = name.replace("q_proj", "v_proj")
                q = new_tensor
                k = state_dict[k_name].float()
                v = state_dict[v_name].float()
                qkv = torch.cat([q, k, v], dim=0)[indices]
                new_name = f"{base}.self_attention.linear_qkv.weight"
                new_tensor = qkv
                chunk_dim = 0
            elif suffix in ("self_attn.k_proj.weight", "self_attn.v_proj.weight"):
                continue  # handled by q_proj above
            elif suffix == "self_attn.q_proj.bias":
                k_name = name.replace("q_proj", "k_proj")
                v_name = name.replace("q_proj", "v_proj")
                q = new_tensor
                k = state_dict[k_name].float()
                v = state_dict[v_name].float()
                qkv = torch.cat([q, k, v], dim=0)[indices]
                new_name = f"{base}.self_attention.linear_qkv.bias"
                new_tensor = qkv
                chunk_dim = 0
            elif suffix in ("self_attn.k_proj.bias", "self_attn.v_proj.bias"):
                continue  # handled by q_proj above

            # Output projection
            elif suffix == "self_attn.out_proj.weight":
                new_name = f"{base}.self_attention.linear_proj.weight"
                chunk_dim = 1
            elif suffix == "self_attn.out_proj.bias":
                new_name = f"{base}.self_attention.linear_proj.bias"

            # Layer norms
            elif suffix == "layer_norm1.weight":
                if use_te:
                    new_name = f"{base}.self_attention.linear_qkv.layer_norm_weight"
                else:
                    new_name = f"{base}.input_layernorm.weight"
            elif suffix == "layer_norm1.bias":
                if use_te:
                    new_name = f"{base}.self_attention.linear_qkv.layer_norm_bias"
                else:
                    new_name = f"{base}.input_layernorm.bias"
            elif suffix == "layer_norm2.weight":
                if use_te:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_weight"
                else:
                    new_name = f"{base}.pre_mlp_layernorm.weight"
            elif suffix == "layer_norm2.bias":
                if use_te:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_bias"
                else:
                    new_name = f"{base}.pre_mlp_layernorm.bias"

            # MLP
            elif suffix == "mlp.fc1.weight":
                new_name = f"{base}.mlp.linear_fc1.weight"
                chunk_dim = 0
            elif suffix == "mlp.fc1.bias":
                new_name = f"{base}.mlp.linear_fc1.bias"
                chunk_dim = 0
            elif suffix == "mlp.fc2.weight":
                new_name = f"{base}.mlp.linear_fc2.weight"
                chunk_dim = 1
            elif suffix == "mlp.fc2.bias":
                new_name = f"{base}.mlp.linear_fc2.bias"

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
# Loading helper: load converted weights into a Megatron CLIPViTModel
# ---------------------------------------------------------------------------


def _get_tp_concat_dim(param_name: str):
    """Return the concat dimension for a TP-sharded parameter, or None if replicated.

    Must match the chunk_dim logic in convert_hf_clip_to_megatron().
    """
    # Column-parallel (chunk_dim=0): QKV weight/bias, FC1 weight/bias
    if "linear_qkv.weight" in param_name or "linear_qkv.bias" in param_name:
        return 0
    if "linear_fc1.weight" in param_name or "linear_fc1.bias" in param_name:
        return 0
    # Row-parallel (chunk_dim=1): proj weight, FC2 weight
    if "linear_proj.weight" in param_name:
        return 1
    if "linear_fc2.weight" in param_name:
        return 1
    # Everything else is replicated (layernorm, class_token, position_embeddings, conv1, biases for proj/fc2)
    return None


def load_megatron_clip_weights(
    clip_model: torch.nn.Module,
    ckpt_dir: str,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> None:
    """Load converted CLIP weights into a Megatron CLIPViTModel.

    Supports loading from a checkpoint saved with a *different* TP size.
    When the checkpoint TP size exceeds the model's TP size (e.g. ckpt TP=4,
    model TP=1), all shards are loaded and concatenated along the appropriate
    dimension.  When they match, only the requested tp_rank is loaded.

    Args:
        clip_model: The Megatron CLIPViTModel instance.
        ckpt_dir: Directory produced by ``convert_hf_clip_to_megatron()``.
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
            concat_dim = _get_tp_concat_dim(key)
            if concat_dim is not None:
                # Concatenate sharded tensors along the split dimension
                state_dict[key] = torch.cat([s[key] for s in all_shards], dim=concat_dim)
            else:
                # Replicated tensor — take from first shard
                state_dict[key] = all_shards[0][key]

        # If model TP > 1, re-split to get the right shard for this rank
        if tp_size > 1:
            new_sd = {}
            for key, tensor in state_dict.items():
                concat_dim = _get_tp_concat_dim(key)
                if concat_dim is not None:
                    chunks = torch.chunk(tensor, tp_size, dim=concat_dim)
                    new_sd[key] = chunks[tp_rank].clone()
                else:
                    new_sd[key] = tensor
            state_dict = new_sd

    incompatible = clip_model.load_state_dict(state_dict, strict=False)
    unexpected = [k for k in incompatible.unexpected_keys if "_extra_state" not in k]
    missing = [k for k in incompatible.missing_keys if "_extra_state" not in k]
    if unexpected or missing:
        raise RuntimeError(f"State dict mismatch. Missing: {missing}, Unexpected: {unexpected}")
    print(f"Loaded CLIP weights from {ckpt_dir} (ckpt_tp={ckpt_tp_size}, model tp_rank={tp_rank}/{tp_size})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace CLIP ViT-L/14-336 to Megatron format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default="openai/clip-vit-large-patch14-336",
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
    args = parser.parse_args()

    convert_hf_clip_to_megatron(
        hf_model_name=args.hf_model,
        output_path=args.output,
        tensor_parallel_size=args.tensor_parallel_size,
        use_te=args.use_te,
    )
    print("Done.")
