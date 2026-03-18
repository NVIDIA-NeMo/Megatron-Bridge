#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Verify Llama/Vicuna weight conversion: HuggingFace vs Megatron.

Loads the same pretrained weights into both HF AutoModelForCausalLM and
Megatron GPTModel, runs the same token input, and compares logits.

Usage:
    # First convert weights (TP=1 for verification):
    python convert_hf_llama_to_megatron.py \
        --hf-model lmsys/vicuna-7b-v1.5 \
        --output /tmp/vicuna_ckpt \
        --tensor-parallel-size 1 \
        --use-te \
        --megatron-vocab-size 32256

    # Then verify:
    torchrun --nproc-per-node=1 verify_llama_conversion.py --checkpoint-dir /tmp/vicuna_ckpt
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
from convert_hf_llama_to_megatron import load_megatron_llm_weights
from megatron.core import parallel_state as ps
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_MODEL = "lmsys/vicuna-7b-v1.5"
HF_VOCAB_SIZE = 32000
MEGATRON_VOCAB_SIZE = 32256
MAX_SEQ_LENGTH = 4096


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
# Language config (mirrors test_mimo_training_llava.py _make_language_config)
# ---------------------------------------------------------------------------


def _make_language_config(dtype: torch.dtype) -> TransformerConfig:
    """Vicuna-7B / Llama-2-7B config."""
    is_bf16 = dtype == torch.bfloat16
    cfg = TransformerConfig(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        use_cpu_initialization=True,
    )
    cfg.ffn_hidden_size = 11008
    cfg.activation_func = torch.nn.functional.silu
    cfg.gated_linear_unit = True

    cfg.normalization = "RMSNorm"
    cfg.rms_norm_eps = 1e-5

    cfg.position_embedding_type = "rope"
    cfg.rotary_base = 10000
    cfg.rotary_percent = 1.0

    cfg.seq_length = MAX_SEQ_LENGTH
    cfg.max_position_embeddings = MAX_SEQ_LENGTH

    cfg.attention_dropout = 0.0
    cfg.hidden_dropout = 0.0

    cfg.num_query_groups = 32
    cfg.add_bias_linear = False
    cfg.untie_embeddings_and_output_weights = False

    cfg.bias_activation_fusion = True
    cfg.masked_softmax_fusion = True
    cfg.persist_layer_norm = True
    cfg.bias_dropout_fusion = True
    cfg.apply_rope_fusion = True

    cfg.pipeline_dtype = dtype
    cfg.bf16 = is_bf16
    cfg.cross_entropy_loss_fusion = True
    cfg.variable_seq_lengths = True

    return cfg


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_outputs(
    hf_logits: torch.Tensor,
    meg_logits: torch.Tensor,
    hf_vocab_size: int,
    label: str = "LLaMA",
) -> bool:
    """Compare logits and print diagnostics. Returns True if passed."""
    # Trim Megatron logits to HF vocab size (padded rows are zeros)
    meg_trimmed = meg_logits[:, :, :hf_vocab_size]
    hf_f = hf_logits.float()
    meg_f = meg_trimmed.float()
    diff = (hf_f - meg_f).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()

    # Cosine similarity (per-sample, flattened)
    cos = torch.nn.functional.cosine_similarity(hf_f.flatten(1), meg_f.flatten(1), dim=1).mean().item()

    # Check padded logits are near-zero
    padded = meg_logits[:, :, hf_vocab_size:].float()
    padded_mean = padded.abs().mean().item()
    padded_max = padded.abs().max().item()

    print(f"\n{'=' * 60}")
    print(f"{label} Verification Results")
    print(f"{'=' * 60}")
    print(f"  HF logits shape:        {tuple(hf_logits.shape)}")
    print(f"  Megatron logits shape:   {tuple(meg_logits.shape)}")
    print(f"  Compared range:          [:, :, :{hf_vocab_size}]")
    print(f"  Mean abs diff:           {mean_diff:.6e}")
    print(f"  Max abs diff:            {max_diff:.6e}")
    print(f"  Cosine similarity:       {cos:.8f}")
    print(f"  Padded logits (expect~0): mean={padded_mean:.6e}, max={padded_max:.6e}")

    # Tolerances (TE kernels + RoPE fusion + 32 layers accumulate diffs)
    passed = mean_diff < 0.5 and cos > 0.99
    status = "PASSED" if passed else "FAILED"
    print(f"\n  Status: {status}")
    print(f"{'=' * 60}\n")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Verify Llama/Vicuna HF→Megatron conversion.")
    parser.add_argument("--checkpoint-dir", required=True, help="Megatron LLM checkpoint dir")
    parser.add_argument("--hf-model", default=HF_MODEL, help="HF model name or path")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--seq-len", type=int, default=32, help="Token sequence length")
    parser.add_argument("--megatron-vocab-size", type=int, default=MEGATRON_VOCAB_SIZE)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    _init_megatron_single_gpu()

    # --- Deterministic input (create before loading any model) ---
    torch.manual_seed(42)
    seq_len = args.seq_len
    input_ids = torch.randint(0, HF_VOCAB_SIZE, (1, seq_len), device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)

    # --- Run Megatron model first, save logits, then free GPU memory ---
    print("Building Megatron GPTModel")
    language_config = _make_language_config(dtype)
    meg_model = GPTModel(
        config=language_config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=args.megatron_vocab_size,
        max_sequence_length=MAX_SEQ_LENGTH,
        position_embedding_type="rope",
        parallel_output=False,
    )
    load_megatron_llm_weights(meg_model, args.checkpoint_dir, tp_rank=0, tp_size=1)
    meg_model.cuda().to(dtype).eval()

    print(f"Running Megatron forward pass (seq_len={seq_len})...")
    with torch.no_grad():
        meg_out = meg_model(input_ids, position_ids, attention_mask=None)  # [1, seq, meg_vocab]
    meg_out = meg_out.cpu()  # move to CPU before freeing GPU

    del meg_model
    torch.cuda.empty_cache()
    print("Megatron model freed from GPU")

    # --- Now load and run HuggingFace model ---
    print(f"Loading HF model: {args.hf_model}")
    hf_model = AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=dtype)
    hf_model.cuda().eval()
    hf_vocab = hf_model.config.vocab_size

    print(f"Running HF forward pass (seq_len={seq_len})...")
    with torch.no_grad():
        hf_out = hf_model(input_ids).logits  # [1, seq, hf_vocab]
    hf_out = hf_out.cpu()

    del hf_model
    torch.cuda.empty_cache()
    print("HF model freed from GPU")

    # --- Compare (both tensors on CPU) ---
    passed = compare_outputs(hf_out, meg_out, hf_vocab, label="Vicuna-7B")

    # --- Cleanup ---
    ps.destroy_model_parallel()
    dist.destroy_process_group()

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
