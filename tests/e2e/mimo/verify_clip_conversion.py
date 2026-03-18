#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Verify CLIP ViT-L/14-336 weight conversion: HuggingFace vs Megatron.

Loads the same pretrained weights into both HF CLIPVisionModel and Megatron
CLIPViTModel, runs the same input, and compares hidden-state outputs.

Usage:
    # First convert weights (TP=1 for verification):
    python convert_hf_clip_to_megatron.py --output /tmp/clip_ckpt --tensor-parallel-size 1 --use-te

    # Then verify:
    torchrun --nproc-per-node=1 verify_clip_conversion.py --checkpoint-dir /tmp/clip_ckpt
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
from convert_hf_clip_to_megatron import load_megatron_clip_weights
from megatron.core import parallel_state as ps
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import CLIPVisionModel


# ---------------------------------------------------------------------------
# Megatron single-GPU init
# ---------------------------------------------------------------------------


def _init_megatron_single_gpu():
    """Lightweight Megatron init for single-GPU inference (TP=1, PP=1)."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )


# ---------------------------------------------------------------------------
# Vision config (mirrors test_mimo_training_llava.py _make_vision_config)
# ---------------------------------------------------------------------------

HF_MODEL = "openai/clip-vit-large-patch14-336"
IMG_SIZE = 336
PATCH_DIM = 14


def _make_vision_config(dtype: torch.dtype) -> TransformerConfig:
    """CLIP ViT-L/14 config matching the HF model architecture."""
    is_bf16 = dtype == torch.bfloat16
    cfg = TransformerConfig(
        num_layers=24,
        hidden_size=1024,
        ffn_hidden_size=4096,
        num_attention_heads=16,
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
    # CLIP uses "quick_gelu", not standard gelu
    cfg.activation_func = lambda x: x * torch.sigmoid(1.702 * x)
    return cfg


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_outputs(
    hf_out: torch.Tensor,
    meg_out: torch.Tensor,
    label: str = "CLIP",
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

    # Tolerances (TE attention kernels cause small numerical diffs)
    passed = mean_diff < 0.1 and max_diff < 50
    status = "PASSED" if passed else "FAILED"
    print(f"\n  Status: {status}")
    print(f"{'=' * 60}\n")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Verify CLIP HF→Megatron conversion.")
    parser.add_argument("--checkpoint-dir", required=True, help="Megatron CLIP checkpoint dir")
    parser.add_argument("--hf-model", default=HF_MODEL, help="HF model name or path")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    _init_megatron_single_gpu()

    # --- HuggingFace model ---
    print(f"Loading HF model: {args.hf_model}")
    hf_model = CLIPVisionModel.from_pretrained(args.hf_model, torch_dtype=dtype)
    hf_model.cuda().eval()

    # --- Megatron model ---
    print("Building Megatron CLIPViTModel")
    vision_config = _make_vision_config(dtype)
    meg_model = CLIPViTModel(
        transformer_config=vision_config,
        transformer_layer_spec=get_vit_layer_with_transformer_engine_spec(),
        patch_dim=PATCH_DIM,
        img_h=IMG_SIZE,
        img_w=IMG_SIZE,
    )
    load_megatron_clip_weights(meg_model, args.checkpoint_dir, tp_rank=0, tp_size=1)
    meg_model.cuda().to(dtype).eval()

    # --- Deterministic input ---
    torch.manual_seed(42)
    pixel_values = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, dtype=dtype, device="cuda")

    # --- Forward passes ---
    print("Running forward passes...")
    with torch.no_grad():
        hf_out = hf_model(pixel_values).last_hidden_state  # [1, 577, 1024]
        meg_out = meg_model(pixel_values)  # [1, 577, 1024]

    passed = compare_outputs(hf_out, meg_out, label="CLIP ViT-L/14-336")

    # --- Cleanup ---
    ps.destroy_model_parallel()
    dist.destroy_process_group()

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
