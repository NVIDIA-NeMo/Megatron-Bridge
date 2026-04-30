#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Verify Llama/Vicuna weight conversion: HuggingFace vs Megatron.

Loads the same pretrained weights into both HF AutoModelForCausalLM and
Megatron GPTModel, runs the same token input, and compares logits.

Supports verification across different TP (tensor-parallel) sizes.
Each TP rank loads its own weight shard; logits are compared against HF on rank 0.

Usage:
    # TP=1:
    python convert_hf_llama_to_megatron.py \
        --hf-model lmsys/vicuna-7b-v1.5 \
        --output /tmp/vicuna_ckpt \
        --tensor-parallel-size 1 \
        --use-te \
        --megatron-vocab-size 32256
    torchrun --nproc-per-node=1 verify_llama_conversion.py --checkpoint-dir /tmp/vicuna_ckpt

    # TP=4:
    python convert_hf_llama_to_megatron.py \
        --hf-model lmsys/vicuna-7b-v1.5 \
        --output /tmp/vicuna_ckpt_tp4 \
        --tensor-parallel-size 4 \
        --use-te \
        --megatron-vocab-size 32256
    torchrun --nproc-per-node=4 verify_llama_conversion.py \
        --checkpoint-dir /tmp/vicuna_ckpt_tp4 --tensor-parallel-size 4
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
# Megatron init (supports TP > 1)
# ---------------------------------------------------------------------------


def _init_megatron(tp_size: int = 1):
    """Initialize Megatron parallel state for the given TP size.

    When tp_size > 1, expects to be launched via torchrun with
    --nproc-per-node equal to tp_size.
    """
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
# Language config (mirrors megatron_mimo_training_llava.py _make_language_config)
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
    """Verify Llama/Vicuna HF→Megatron conversion."""
    parser = argparse.ArgumentParser(description="Verify Llama/Vicuna HF→Megatron conversion.")
    parser.add_argument("--checkpoint-dir", required=True, help="Megatron LLM checkpoint dir")
    parser.add_argument("--hf-model", default=HF_MODEL, help="HF model name or path")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--seq-len", type=int, default=32, help="Token sequence length")
    parser.add_argument("--megatron-vocab-size", type=int, default=MEGATRON_VOCAB_SIZE)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="TP size (must match --nproc-per-node)",
    )
    args = parser.parse_args()

    tp_size = args.tensor_parallel_size
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    _init_megatron(tp_size)

    rank = dist.get_rank()
    tp_rank = ps.get_tensor_model_parallel_rank()

    # --- Deterministic input (same on all ranks) ---
    torch.manual_seed(42)
    seq_len = args.seq_len
    input_ids = torch.randint(0, HF_VOCAB_SIZE, (1, seq_len), device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)

    # --- Megatron model (all ranks participate for TP communication) ---
    if rank == 0:
        print(f"Building Megatron GPTModel (TP={tp_size})")
    language_config = _make_language_config(dtype)
    meg_model = GPTModel(
        config=language_config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=args.megatron_vocab_size,
        max_sequence_length=MAX_SEQ_LENGTH,
        position_embedding_type="rope",
        parallel_output=False,
    )
    load_megatron_llm_weights(meg_model, args.checkpoint_dir, tp_rank=tp_rank, tp_size=tp_size)
    meg_model.cuda().to(dtype).eval()

    if rank == 0:
        print(f"Running Megatron forward pass (seq_len={seq_len})...")
    with torch.no_grad():
        meg_out = meg_model(input_ids, position_ids, attention_mask=None)  # [1, seq, meg_vocab]
    meg_out_cpu = meg_out.cpu()

    del meg_model
    torch.cuda.empty_cache()

    # --- HF comparison on rank 0 only ---
    if rank == 0:
        print("Megatron model freed from GPU")
        print(f"Loading HF model: {args.hf_model}")
        hf_model = AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=dtype)
        hf_model.cuda().eval()
        hf_vocab = hf_model.config.vocab_size

        print(f"Running HF forward pass (seq_len={seq_len})...")
        with torch.no_grad():
            hf_out = hf_model(input_ids).logits  # [1, seq, hf_vocab]
        hf_out_cpu = hf_out.cpu()

        del hf_model
        torch.cuda.empty_cache()
        print("HF model freed from GPU")

        passed = compare_outputs(hf_out_cpu, meg_out_cpu, hf_vocab, label=f"Vicuna-7B (TP={tp_size})")
    else:
        passed = True

    # --- Broadcast pass/fail to all ranks ---
    result = torch.tensor([1 if passed else 0], device="cuda")
    dist.broadcast(result, src=0)
    passed = result.item() == 1

    # --- Cleanup ---
    ps.destroy_model_parallel()
    dist.destroy_process_group()

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
