#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Verify Whisper encoder weight conversion: HuggingFace vs Megatron.

Loads the same pretrained weights into both HF WhisperModel.encoder and Megatron
WhisperEncoder, runs the same input, and compares hidden-state outputs.

Supports verification across different TP (tensor-parallel) sizes.
Each TP rank loads its own weight shard; outputs are compared against HF on rank 0.

Usage:
    # Convert first:
    python convert_hf_whisper_to_megatron.py \
        --hf-model openai/whisper-base --output /tmp/whisper_ckpt --tensor-parallel-size 1 --use-te

    # TP=1:
    torchrun --nproc-per-node=1 verify_whisper_conversion.py \
        --checkpoint-dir /tmp/whisper_ckpt --hf-model openai/whisper-base

    # TP=2:
    python convert_hf_whisper_to_megatron.py \
        --hf-model openai/whisper-base --output /tmp/whisper_ckpt_tp2 --tensor-parallel-size 2 --use-te
    torchrun --nproc-per-node=2 verify_whisper_conversion.py \
        --checkpoint-dir /tmp/whisper_ckpt_tp2 --hf-model openai/whisper-base --tensor-parallel-size 2
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
from convert_hf_whisper_to_megatron import load_megatron_whisper_weights
from megatron.core import parallel_state as ps
from megatron.core.transformer.transformer_config import TransformerConfig
from whisper_layer_specs import get_whisper_layer_with_transformer_engine_spec
from whisper_model import WhisperEncoder


# ---------------------------------------------------------------------------
# Megatron init (supports TP > 1)
# ---------------------------------------------------------------------------


def _init_megatron(tp_size: int = 1):
    """Initialize Megatron parallel state for the given TP size."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()
    if world_size != tp_size:
        raise RuntimeError(
            f"World size ({world_size}) must equal --tensor-parallel-size ({tp_size}). "
            f"Use: torchrun --nproc-per-node={tp_size}"
        )

    ps.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
    )


# ---------------------------------------------------------------------------
# Build Megatron config from HF config
# ---------------------------------------------------------------------------


def _make_whisper_config(hf_config, dtype: torch.dtype) -> TransformerConfig:
    """Build a Megatron TransformerConfig matching the HF Whisper encoder architecture."""
    is_bf16 = dtype == torch.bfloat16
    cfg = TransformerConfig(
        num_layers=hf_config.encoder_layers,
        hidden_size=hf_config.d_model,
        ffn_hidden_size=hf_config.encoder_ffn_dim,
        num_attention_heads=hf_config.encoder_attention_heads,
        use_cpu_initialization=True,
        pipeline_dtype=dtype,
        bf16=is_bf16,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
    )
    cfg.add_bias_linear = True
    cfg.add_qkv_bias = True
    cfg.hidden_dropout = 0.0
    cfg.attention_dropout = 0.0
    cfg.gated_linear_unit = False
    cfg.layernorm_zero_centered_gamma = False
    cfg.apply_query_key_layer_scaling = False
    cfg.bias_activation_fusion = False
    cfg.bias_dropout_fusion = False
    cfg.attention_softmax_in_fp32 = True
    cfg.normalization = "LayerNorm"
    cfg.apply_rope_fusion = False
    cfg.activation_func = torch.nn.functional.gelu
    return cfg


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_outputs(
    hf_out: torch.Tensor,
    meg_out: torch.Tensor,
    label: str = "Whisper",
) -> bool:
    """Compare two tensors and print diagnostics. Returns True if passed."""
    hf_f = hf_out.float()
    meg_f = meg_out.float()
    diff = (hf_f - meg_f).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()

    # Cosine similarity (per-sample, flattened)
    cos = torch.nn.functional.cosine_similarity(hf_f.flatten(1), meg_f.flatten(1), dim=1).mean().item()

    print(f"\n{'=' * 60}")
    print(f"{label} Verification Results")
    print(f"{'=' * 60}")
    print(f"  HF output shape:       {tuple(hf_out.shape)}")
    print(f"  Megatron output shape:  {tuple(meg_out.shape)}")
    print(f"  Mean abs diff:          {mean_diff:.6e}")
    print(f"  Max abs diff:           {max_diff:.6e}")
    print(f"  Cosine similarity:      {cos:.8f}")

    # Tolerances: TE attention kernels cause small numerical diffs vs HF.
    # For fp32: mean ~1e-4, max ~0.1 is typical. For bf16: slightly higher.
    passed = mean_diff < 1e-2 and max_diff < 1.0 and cos > 0.9999
    status = "PASSED" if passed else "FAILED"
    print(f"\n  Status: {status}")
    print(f"{'=' * 60}\n")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """CLI entrypoint for verifying a Whisper HF→Megatron conversion."""
    parser = argparse.ArgumentParser(description="Verify Whisper HF→Megatron conversion.")
    parser.add_argument("--checkpoint-dir", required=True, help="Megatron Whisper checkpoint dir")
    parser.add_argument("--hf-model", default="openai/whisper-base", help="HF model name or path")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="TP size (must match --nproc-per-node)",
    )
    parser.add_argument(
        "--mel-frames",
        type=int,
        default=3000,
        help="Number of mel spectrogram frames (default: 3000 = 30s at 100 fps)",
    )
    args = parser.parse_args()

    tp_size = args.tensor_parallel_size
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    _init_megatron(tp_size)

    rank = dist.get_rank()
    tp_rank = ps.get_tensor_model_parallel_rank()

    # --- Load HF config to derive Megatron config ---
    from transformers import WhisperConfig

    hf_config = WhisperConfig.from_pretrained(args.hf_model)

    # --- Deterministic input (same on all ranks) ---
    torch.manual_seed(42)
    input_features = torch.randn(1, hf_config.num_mel_bins, args.mel_frames, dtype=dtype, device="cuda")

    # --- Megatron model (all ranks participate for TP communication) ---
    if rank == 0:
        print(f"Building Megatron WhisperEncoder (TP={tp_size})")
        print(
            f"  d_model={hf_config.d_model}, layers={hf_config.encoder_layers}, "
            f"heads={hf_config.encoder_attention_heads}, ffn={hf_config.encoder_ffn_dim}"
        )

    whisper_config = _make_whisper_config(hf_config, dtype)
    meg_model = WhisperEncoder(
        transformer_config=whisper_config,
        transformer_layer_spec=get_whisper_layer_with_transformer_engine_spec(),
        num_mel_bins=hf_config.num_mel_bins,
        max_source_positions=hf_config.max_source_positions,
    )
    load_megatron_whisper_weights(meg_model, args.checkpoint_dir, tp_rank=tp_rank, tp_size=tp_size)
    meg_model.cuda().to(dtype).eval()

    if rank == 0:
        print("Running Megatron forward pass...")
    with torch.no_grad():
        meg_out = meg_model(input_features)  # [1, S, d_model]

    # --- HF comparison on rank 0 only ---
    if rank == 0:
        print(f"Loading HF model: {args.hf_model}")
        from transformers import WhisperModel

        hf_model = WhisperModel.from_pretrained(args.hf_model, torch_dtype=dtype)
        hf_encoder = hf_model.encoder.cuda().eval()

        with torch.no_grad():
            hf_out = hf_encoder(input_features).last_hidden_state  # [1, S, d_model]

        del hf_model, hf_encoder
        torch.cuda.empty_cache()

        model_name = args.hf_model.split("/")[-1]
        passed = compare_outputs(hf_out, meg_out, label=f"Whisper {model_name} (TP={tp_size})")
    else:
        passed = True

    # --- Broadcast pass/fail to all ranks ---
    result = torch.tensor([1 if passed else 0], device="cuda")
    dist.broadcast(result, src=0)
    passed = result.item() == 1

    # --- Cleanup ---
    del meg_model
    torch.cuda.empty_cache()
    ps.destroy_model_parallel()
    dist.destroy_process_group()

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
