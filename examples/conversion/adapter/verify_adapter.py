#!/usr/bin/env python3
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
Verify an exported HuggingFace PEFT adapter by loading it with the PEFT
library and comparing logits against the Megatron checkpoint.

Supports both CPU-only (single process) and multi-GPU (torchrun) modes.

Verification criteria (configurable with ``--top-k``):
  * PEFT model logits must differ from the base model (adapter has effect).
  * When ``--lora-checkpoint`` is given, the top-k predicted tokens
    from the PEFT model must match those from the Megatron model with merged
    weights.

CPU mode (no GPU required)::

    uv run python examples/conversion/adapter/verify_adapter.py \\
        --hf-model-path Qwen/Qwen3-0.6B \\
        --hf-adapter-path ./my_adapter \\
        --lora-checkpoint /path/to/finetune_ckpt/iter_0000020 \\
        --cpu

GPU mode (single GPU)::

    uv run python examples/conversion/adapter/verify_adapter.py \\
        --hf-model-path Qwen/Qwen3-0.6B \\
        --hf-adapter-path ./my_adapter \\
        --lora-checkpoint /path/to/finetune_ckpt/iter_0000020

Multi-GPU mode (TP=2)::

    uv run python -m torch.distributed.run --nproc_per_node=2 \\
        examples/conversion/adapter/verify_adapter.py \\
        --hf-model-path Qwen/Qwen3-0.6B \\
        --hf-adapter-path ./my_adapter \\
        --lora-checkpoint /path/to/finetune_ckpt/iter_0000020 \\
        --tp 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify exported HF PEFT adapter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hf-model-path", required=True, help="HF base model name or path.")
    parser.add_argument("--hf-adapter-path", required=True, help="Exported HF PEFT adapter directory.")
    parser.add_argument(
        "--lora-checkpoint",
        default=None,
        help="Megatron PEFT checkpoint (iter dir). Required for Megatron-side verification.",
    )
    parser.add_argument("--prompt", default="The capital of France is", help="Prompt for the forward pass.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top tokens to compare.")

    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--cpu", action="store_true", help="Run entirely on CPU (no GPU required)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _forward_logits(model, tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    """Single forward pass, return last-token logits as float32 on CPU."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    return logits.cpu().float()


def _top_k_info(logits: torch.Tensor, tokenizer, k: int) -> tuple[list[int], list[str], list[float]]:
    vals, ids = torch.topk(logits, k)
    token_ids = ids.tolist()
    tokens = [tokenizer.decode([i]) for i in token_ids]
    scores = vals.tolist()
    return token_ids, tokens, scores


def _print_top_k(label: str, logits: torch.Tensor, tokenizer, k: int) -> None:
    _, tokens, scores = _top_k_info(logits, tokenizer, k)
    pairs = list(zip(tokens, [f"{v:.4f}" for v in scores]))
    print(f"  {label} top-{k}: {pairs}")


def _compare_top_k(
    label: str,
    ref_logits: torch.Tensor,
    cand_logits: torch.Tensor,
    tokenizer,
    k: int,
) -> bool:
    """Return True if the top-k token IDs match between ref and cand."""
    ref_ids, ref_tok, _ = _top_k_info(ref_logits, tokenizer, k)
    cand_ids, cand_tok, _ = _top_k_info(cand_logits, tokenizer, k)
    match = ref_ids == cand_ids
    diff = (ref_logits - cand_logits).abs()
    status = "PASS" if match else "FAIL"
    print(f"\n  {label}")
    print(f"    top-{k} tokens ref : {ref_tok}")
    print(f"    top-{k} tokens cand: {cand_tok}")
    print(f"    max logit diff: {diff.max().item():.6e}  mean: {diff.mean().item():.6e}")
    print(f"    => {status}")
    return match


# ---------------------------------------------------------------------------
# Build Megatron model with LoRA from checkpoint
# ---------------------------------------------------------------------------


def _build_megatron_lora_model(hf_model_path, peft_checkpoint, trust_remote_code, *, tp=1, pp=1, ep=1, cpu=False):
    from megatron.core import dist_checkpointing

    from megatron.bridge.models.conversion.auto_bridge import AutoBridge
    from megatron.bridge.peft.lora import LoRA, VLMLoRA
    from megatron.bridge.training.checkpointing import (
        _generate_model_state_dict,
        apply_peft_adapter_filter_to_state_dict,
    )
    from megatron.bridge.training.model_load_save import temporary_distributed_context
    from megatron.bridge.training.utils.checkpoint_utils import read_run_config

    bridge = AutoBridge.from_hf_pretrained(hf_model_path, trust_remote_code=trust_remote_code)

    ckpt_path = Path(peft_checkpoint).expanduser().resolve()
    peft_class: type = LoRA
    peft_cfg: dict = {}
    cfg_file = ckpt_path / "run_config.yaml"
    if not cfg_file.exists() and ckpt_path.parent != ckpt_path:
        cfg_file = ckpt_path.parent / "run_config.yaml"
    if cfg_file.exists():
        run_cfg_dict = read_run_config(str(cfg_file))
        peft_cfg = run_cfg_dict.get("peft", {}) or {}
        if "VLMLoRA" in peft_cfg.get("_target_", ""):
            peft_class = VLMLoRA
        allowed = {
            "target_modules",
            "dim",
            "alpha",
            "dropout",
            "dropout_position",
            "freeze_language_model",
            "freeze_vision_model",
            "freeze_vision_projection",
        }
        peft_cfg = {k: v for k, v in peft_cfg.items() if k in allowed}

    lora = peft_class(**peft_cfg)
    print(f"  LoRA config: class={peft_class.__name__}, dim={lora.dim}, alpha={lora.alpha}")

    provider = bridge.to_megatron_provider(load_weights=True)
    provider.pipeline_dtype = torch.float32
    provider.params_dtype = torch.float32
    provider.tensor_model_parallel_size = tp
    provider.pipeline_model_parallel_size = pp
    provider.expert_model_parallel_size = ep
    provider.finalize()
    provider.register_pre_wrap_hook(lambda chunks: lora(chunks, training=False))

    dist_ctx = None
    if not torch.distributed.is_initialized():
        backend = "gloo" if cpu else "nccl"
        dist_ctx = temporary_distributed_context(backend=backend)
        dist_ctx.__enter__()
    else:
        provider.initialize_model_parallel(seed=0)

    model = provider.provide_distributed_model(
        wrap_with_ddp=False,
        use_cpu_initialization=cpu,
        init_model_with_meta_device=False,
    )

    sharded_sd = _generate_model_state_dict(model, {})
    sharded_sd = apply_peft_adapter_filter_to_state_dict(sharded_sd, lora)
    loaded_sd = dist_checkpointing.load(sharded_sd, str(ckpt_path))
    model_key = "model" if "model" in loaded_sd else next(k for k in loaded_sd if k.startswith("model"))
    model[0].load_state_dict(loaded_sd[model_key], strict=False)

    return bridge, model, lora, dist_ctx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run adapter verification checks."""
    args = parse_args()
    k = args.top_k
    use_gpu = not args.cpu
    is_multi_gpu = args.tp > 1 or args.pp > 1 or args.ep > 1

    if is_multi_gpu and args.cpu:
        print("ERROR: TP/PP/EP > 1 requires GPU; do not use --cpu", file=sys.stderr)
        sys.exit(1)

    if is_multi_gpu and os.environ.get("WORLD_SIZE") is None:
        print(
            "ERROR: TP/PP/EP > 1 requires torchrun. Run with:\n"
            f"  python -m torch.distributed.run --nproc_per_node=<gpus> {sys.argv[0]} ...",
            file=sys.stderr,
        )
        sys.exit(1)

    if use_gpu and os.environ.get("WORLD_SIZE") is not None and not torch.distributed.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")

    rank = int(os.environ.get("RANK", 0))
    is_rank_0 = rank == 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}") if use_gpu else torch.device("cpu")

    all_pass = True
    base_logits = None
    peft_logits = None
    tokenizer = None

    # ------------------------------------------------------------------
    # Steps 0-2: HF model operations (rank 0 only)
    # ------------------------------------------------------------------
    if is_rank_0:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=args.trust_remote_code)

        # 0) Read adapter_config.json
        adapter_cfg_path = Path(args.hf_adapter_path) / "adapter_config.json"
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        print(
            f"\nadapter_config.json: r={adapter_cfg['r']}, lora_alpha={adapter_cfg['lora_alpha']}, "
            f"target_modules={adapter_cfg.get('target_modules')}"
        )

        # 1) HF base model logits
        print("\n[Step 1] Loading HF base model ...")
        hf_base = AutoModelForCausalLM.from_pretrained(
            args.hf_model_path,
            torch_dtype=torch.float32,
            trust_remote_code=args.trust_remote_code,
        ).to(device)
        base_logits = _forward_logits(hf_base, tokenizer, args.prompt, device)
        _print_top_k("HF base (no adapter)", base_logits, tokenizer, k)
        del hf_base

        # 2) PEFT library loading check
        print("\n[Step 2] Loading adapter with PEFT library ...")
        from peft import PeftModel

        peft_base = AutoModelForCausalLM.from_pretrained(
            args.hf_model_path,
            torch_dtype=torch.float32,
            trust_remote_code=args.trust_remote_code,
        ).to(device)
        peft_model = PeftModel.from_pretrained(peft_base, args.hf_adapter_path)
        peft_model.eval()
        peft_logits = _forward_logits(peft_model, tokenizer, args.prompt, device)
        _print_top_k("HF PEFT", peft_logits, tokenizer, k)
        del peft_model, peft_base

        peft_vs_base = (peft_logits - base_logits).abs().max().item()
        if peft_vs_base < 1e-6:
            print("\n  FAIL: PEFT model logits are identical to base model.")
            print("  PEFT failed to load the adapter weights from the safetensors file.")
            all_pass = False
        else:
            print(f"\n  Adapter effect on logits: max diff from base = {peft_vs_base:.6e}  PASS")

    if not args.lora_checkpoint:
        if is_rank_0:
            print("\n\nSkipping Megatron-side checks (--lora-checkpoint not provided).")
            if all_pass:
                print("PASSED")
            else:
                raise SystemExit("FAILED: see details above")
        _cleanup_distributed(args.cpu)
        return

    # Synchronize before Megatron work (all ranks must participate together)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # ------------------------------------------------------------------
    # 3) Megatron: load model with LoRA, export merged weights (all ranks)
    # ------------------------------------------------------------------
    if is_rank_0:
        print("\n[Step 3] Building Megatron model with LoRA from checkpoint ...")

    bridge, mg_model, lora, dist_ctx = _build_megatron_lora_model(
        args.hf_model_path,
        args.lora_checkpoint,
        args.trust_remote_code,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        cpu=args.cpu,
    )

    try:
        mg_merged_sd: dict[str, torch.Tensor] = {}
        for name, tensor in bridge.export_hf_weights(mg_model, cpu=True, merge_adapter_weights=True):
            mg_merged_sd[name] = tensor
    finally:
        if dist_ctx:
            dist_ctx.__exit__(None, None, None)

    # ------------------------------------------------------------------
    # 4) Logit-level verification (top-k) — rank 0 only
    # ------------------------------------------------------------------
    if is_rank_0:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=args.trust_remote_code)

        print(f"\n[Step 4] Top-{k} logit verification ...")

        mg_hf = AutoModelForCausalLM.from_pretrained(
            args.hf_model_path,
            torch_dtype=torch.float32,
            trust_remote_code=args.trust_remote_code,
        )
        mg_hf.load_state_dict(mg_merged_sd, strict=True)
        mg_hf = mg_hf.to(device)
        mg_logits = _forward_logits(mg_hf, tokenizer, args.prompt, device)
        _print_top_k("Megatron merged", mg_logits, tokenizer, k)
        del mg_hf

        if not _compare_top_k("PEFT vs Megatron merged", peft_logits, mg_logits, tokenizer, k):
            all_pass = False

        # Result
        print(f"\n{'=' * 70}")
        if all_pass:
            print("  PASSED: adapter export is correct")
        else:
            print("  FAILED: see details above")
        print(f"{'=' * 70}")

        if not all_pass:
            raise SystemExit("Adapter verification failed")

    _cleanup_distributed(args.cpu)


def _cleanup_distributed(cpu: bool) -> None:
    """Destroy the process group if it was initialized by us (not by temporary_distributed_context)."""
    if not cpu and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
