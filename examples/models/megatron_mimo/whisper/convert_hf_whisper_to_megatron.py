#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Convert HuggingFace Whisper encoder weights to Megatron format.

Produces per-TP-rank .pt files that can be loaded into a Megatron
WhisperEncoder with ``load_megatron_whisper_weights()``.

Usage:
    python convert_hf_whisper_to_megatron.py \
        --hf-model openai/whisper-base \
        --output /path/to/whisper_megatron_ckpt \
        --tensor-parallel-size 4 \
        --use-te

    # Then in your training script, after model construction:
    from convert_hf_whisper_to_megatron import load_megatron_whisper_weights
    load_megatron_whisper_weights(whisper_model, "/path/to/whisper_megatron_ckpt", tp_rank, tp_size)
"""

import argparse
import os

import torch
from transformers import WhisperModel


# ---------------------------------------------------------------------------
# Key mapping: HuggingFace -> Megatron
# ---------------------------------------------------------------------------


def _build_qkv_interleave_indices(hidden_dim: int, num_heads: int) -> torch.Tensor:
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


def convert_hf_whisper_to_megatron(
    hf_model_name: str = "openai/whisper-base",
    output_path: str = "./whisper_megatron_ckpt",
    tensor_parallel_size: int = 1,
    use_te: bool = True,
) -> None:
    """Download HF Whisper weights and save encoder as per-TP-rank Megatron .pt files.

    Args:
        hf_model_name: HuggingFace model identifier or local path.
        output_path: Directory for output checkpoint files.
        tensor_parallel_size: Target tensor parallelism size.
        use_te: If True, use Transformer Engine layer naming (fused layernorm
                inside linear_qkv / linear_fc1).
    """
    print(f"Loading HuggingFace model: {hf_model_name}")
    hf_model = WhisperModel.from_pretrained(hf_model_name)
    state_dict = hf_model.state_dict()

    hidden_dim = hf_model.config.d_model
    num_heads = hf_model.config.encoder_attention_heads
    indices = _build_qkv_interleave_indices(hidden_dim, num_heads)

    print(
        f"  d_model={hidden_dim}, encoder_attention_heads={num_heads}, "
        f"encoder_ffn_dim={hf_model.config.encoder_ffn_dim}, "
        f"encoder_layers={hf_model.config.encoder_layers}"
    )

    new_state_dicts = [{"model": {}} for _ in range(tensor_parallel_size)]

    for name, tensor in state_dict.items():
        # Only convert encoder weights; skip decoder weights
        if not name.startswith("encoder."):
            continue

        new_name = ""
        new_tensor = tensor.float()  # convert to fp32 for saving
        chunk_dim = None  # dimension to chunk for TP

        # --- Convolutional embeddings ---
        if name == "encoder.conv1.weight":
            new_name = "conv1.weight"
        elif name == "encoder.conv1.bias":
            new_name = "conv1.bias"
        elif name == "encoder.conv2.weight":
            new_name = "conv2.weight"
        elif name == "encoder.conv2.bias":
            new_name = "conv2.bias"

        # --- Positional embeddings (sinusoidal) ---
        elif name == "encoder.embed_positions.weight":
            new_name = "position_embeddings.weight"

        # --- Final layer norm (after all encoder layers) ---
        # WhisperEncoder uses post_process=False in TransformerBlock, so
        # decoder.final_layernorm doesn't exist.  The model's own ln_post handles this.
        elif name == "encoder.layer_norm.weight":
            new_name = "ln_post.weight"
        elif name == "encoder.layer_norm.bias":
            new_name = "ln_post.bias"

        # --- Encoder layers ---
        elif name.startswith("encoder.layers."):
            parts = name.split(".")
            layer_idx = parts[2]
            base = f"decoder.layers.{layer_idx}"
            suffix = ".".join(parts[3:])  # e.g. "self_attn.q_proj.weight"

            # Self-attention QKV (separate -> fused)
            # Note: HF Whisper k_proj has no bias (hardcoded bias=False).
            # When fusing QKV biases, use zeros for the K portion.
            if suffix == "self_attn.q_proj.weight":
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
                k = state_dict.get(k_name, torch.zeros_like(new_tensor)).float()
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
            elif suffix == "self_attn_layer_norm.weight":
                if use_te:
                    new_name = f"{base}.self_attention.linear_qkv.layer_norm_weight"
                else:
                    new_name = f"{base}.input_layernorm.weight"
            elif suffix == "self_attn_layer_norm.bias":
                if use_te:
                    new_name = f"{base}.self_attention.linear_qkv.layer_norm_bias"
                else:
                    new_name = f"{base}.input_layernorm.bias"
            elif suffix == "final_layer_norm.weight":
                if use_te:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_weight"
                else:
                    new_name = f"{base}.pre_mlp_layernorm.weight"
            elif suffix == "final_layer_norm.bias":
                if use_te:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_bias"
                else:
                    new_name = f"{base}.pre_mlp_layernorm.bias"

            # MLP
            elif suffix == "fc1.weight":
                new_name = f"{base}.mlp.linear_fc1.weight"
                chunk_dim = 0
            elif suffix == "fc1.bias":
                new_name = f"{base}.mlp.linear_fc1.bias"
                chunk_dim = 0
            elif suffix == "fc2.weight":
                new_name = f"{base}.mlp.linear_fc2.weight"
                chunk_dim = 1
            elif suffix == "fc2.bias":
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

    print(f"Conversion complete -> {output_path}")


# ---------------------------------------------------------------------------
# Loading helper: load converted weights into a Megatron WhisperEncoder
# ---------------------------------------------------------------------------


def _get_tp_concat_dim(param_name: str):
    """Return the concat dimension for a TP-sharded parameter, or None if replicated.

    Must match the chunk_dim logic in convert_hf_whisper_to_megatron().
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
    # Everything else is replicated (layernorm, conv, position_embeddings, biases for proj/fc2)
    return None


def load_megatron_whisper_weights(
    whisper_model: torch.nn.Module,
    ckpt_dir: str,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> None:
    """Load converted Whisper encoder weights into a Megatron WhisperEncoder.

    Supports loading from a checkpoint saved with a *different* TP size.
    When the checkpoint TP size exceeds the model's TP size (e.g. ckpt TP=4,
    model TP=1), all shards are loaded and concatenated along the appropriate
    dimension.  When they match, only the requested tp_rank is loaded.

    Args:
        whisper_model: The Megatron WhisperEncoder instance.
        ckpt_dir: Directory produced by ``convert_hf_whisper_to_megatron()``.
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
                # Replicated tensor -- take from first shard
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

    incompatible = whisper_model.load_state_dict(state_dict, strict=False)
    unexpected = [k for k in incompatible.unexpected_keys if "_extra_state" not in k]
    missing = [k for k in incompatible.missing_keys if "_extra_state" not in k]
    if unexpected or missing:
        raise RuntimeError(f"State dict mismatch. Missing: {missing}, Unexpected: {unexpected}")
    print(f"Loaded Whisper weights from {ckpt_dir} (ckpt_tp={ckpt_tp_size}, model tp_rank={tp_rank}/{tp_size})")


# ---------------------------------------------------------------------------
# Verification: load back and check shapes
# ---------------------------------------------------------------------------


def verify_conversion(
    output_path: str,
    hf_model_name: str = "openai/whisper-base",
    tensor_parallel_size: int = 1,
) -> bool:
    """Verify the converted checkpoint by loading back and checking tensor shapes.

    Args:
        output_path: Directory with per-TP-rank checkpoint files.
        hf_model_name: HuggingFace model identifier (for reference shapes).
        tensor_parallel_size: TP size the checkpoint was converted with.

    Returns:
        True if all checks pass.
    """
    print(f"\nVerifying conversion in {output_path} (TP={tensor_parallel_size})")

    # Load HF model for reference shapes
    hf_model = WhisperModel.from_pretrained(hf_model_name)
    hf_config = hf_model.config
    hidden_dim = hf_config.d_model
    encoder_ffn_dim = hf_config.encoder_ffn_dim
    num_layers = hf_config.encoder_layers
    num_mel_bins = hf_config.num_mel_bins
    max_source_positions = hf_config.max_source_positions
    del hf_model

    passed = True

    # Expected shapes (full, before TP sharding)
    expected_shapes = {
        "conv1.weight": (hidden_dim, num_mel_bins, 3),
        "conv1.bias": (hidden_dim,),
        "conv2.weight": (hidden_dim, hidden_dim, 3),
        "conv2.bias": (hidden_dim,),
        "position_embeddings.weight": (max_source_positions, hidden_dim),
        "ln_post.weight": (hidden_dim,),
        "ln_post.bias": (hidden_dim,),
    }

    # Per-layer expected shapes
    for i in range(num_layers):
        base = f"decoder.layers.{i}"
        # QKV fused: 3 * hidden_dim (before TP split on dim 0)
        expected_shapes[f"{base}.self_attention.linear_qkv.weight"] = (
            3 * hidden_dim // tensor_parallel_size,
            hidden_dim,
        )
        expected_shapes[f"{base}.self_attention.linear_qkv.bias"] = (3 * hidden_dim // tensor_parallel_size,)
        # Output projection: split on dim 1
        expected_shapes[f"{base}.self_attention.linear_proj.weight"] = (
            hidden_dim,
            hidden_dim // tensor_parallel_size,
        )
        expected_shapes[f"{base}.self_attention.linear_proj.bias"] = (hidden_dim,)
        # LayerNorm (not sharded)
        expected_shapes[f"{base}.self_attention.linear_qkv.layer_norm_weight"] = (hidden_dim,)
        expected_shapes[f"{base}.self_attention.linear_qkv.layer_norm_bias"] = (hidden_dim,)
        # FC1: split on dim 0
        expected_shapes[f"{base}.mlp.linear_fc1.weight"] = (
            encoder_ffn_dim // tensor_parallel_size,
            hidden_dim,
        )
        expected_shapes[f"{base}.mlp.linear_fc1.bias"] = (encoder_ffn_dim // tensor_parallel_size,)
        # FC2: split on dim 1
        expected_shapes[f"{base}.mlp.linear_fc2.weight"] = (
            hidden_dim,
            encoder_ffn_dim // tensor_parallel_size,
        )
        expected_shapes[f"{base}.mlp.linear_fc2.bias"] = (hidden_dim,)
        # MLP LayerNorm (not sharded)
        expected_shapes[f"{base}.mlp.linear_fc1.layer_norm_weight"] = (hidden_dim,)
        expected_shapes[f"{base}.mlp.linear_fc1.layer_norm_bias"] = (hidden_dim,)

    for tp in range(tensor_parallel_size):
        ckpt_file = os.path.join(output_path, f"tp_rank_{tp:02d}", "model_weights.pt")
        if not os.path.exists(ckpt_file):
            print(f"  [FAIL] Missing file: {ckpt_file}")
            passed = False
            continue

        saved = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        model_dict = {k: v for k, v in saved["model"].items() if v is not None}

        print(f"\n  TP rank {tp}: {len(model_dict)} parameters")

        for key, expected_shape in expected_shapes.items():
            if key not in model_dict:
                print(f"    [FAIL] Missing key: {key}")
                passed = False
                continue
            actual_shape = tuple(model_dict[key].shape)
            if actual_shape != expected_shape:
                print(f"    [FAIL] {key}: expected {expected_shape}, got {actual_shape}")
                passed = False
            else:
                # Only print first/last layer to avoid excessive output
                parts = key.split(".")
                if len(parts) >= 3 and parts[2].isdigit():
                    layer_idx = int(parts[2])
                    if layer_idx not in (0, num_layers - 1):
                        continue
                print(f"    [OK]   {key}: {actual_shape}")

        # Check for unexpected keys (ignore _extra_state)
        expected_keys = set(expected_shapes.keys())
        actual_keys = set(model_dict.keys())
        unexpected = actual_keys - expected_keys
        if unexpected:
            print(f"    [WARN] Unexpected keys: {unexpected}")

    status = "PASSED" if passed else "FAILED"
    print(f"\nVerification: {status}")
    return passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace Whisper encoder to Megatron format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default="openai/whisper-base",
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
        "--verify",
        action="store_true",
        help="Run verification after conversion (loads back and checks shapes)",
    )
    args = parser.parse_args()

    convert_hf_whisper_to_megatron(
        hf_model_name=args.hf_model,
        output_path=args.output,
        tensor_parallel_size=args.tensor_parallel_size,
        use_te=args.use_te,
    )

    if args.verify:
        ok = verify_conversion(
            output_path=args.output,
            hf_model_name=args.hf_model,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        if not ok:
            raise RuntimeError("Verification failed!")

    print("Done.")
