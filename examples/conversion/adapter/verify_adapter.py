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

Verification criteria (configurable with ``--top-k``):
  * PEFT model logits must differ from the base model (adapter has effect).
  * When ``--megatron-peft-checkpoint`` is given, the top-k predicted tokens
    from the PEFT model must match those from the Megatron model with merged
    weights.

Usage::

    uv run python examples/conversion/adapter/verify_adapter.py \\
        --hf-model-id Qwen/Qwen3-0.6B \\
        --hf-adapter-path ./my_adapter \\
        --megatron-peft-checkpoint /path/to/finetune_ckpt/iter_0000020
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify exported HF PEFT adapter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hf-model-id", required=True, help="HF base model name or path.")
    parser.add_argument("--hf-adapter-path", required=True, help="Exported HF PEFT adapter directory.")
    parser.add_argument(
        "--megatron-peft-checkpoint",
        default=None,
        help="Megatron PEFT checkpoint (iter dir). Required for Megatron-side verification.",
    )
    parser.add_argument("--prompt", default="The capital of France is", help="Prompt for the forward pass.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top tokens to compare.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _forward_logits(model, tokenizer, prompt: str) -> torch.Tensor:
    """Single forward pass, return last-token logits as float32 on CPU."""
    inputs = tokenizer(prompt, return_tensors="pt")
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


def _build_megatron_lora_model(hf_model_id, peft_checkpoint, trust_remote_code):
    from megatron.core import dist_checkpointing

    from megatron.bridge.models.conversion.auto_bridge import AutoBridge
    from megatron.bridge.peft.lora import LoRA, VLMLoRA
    from megatron.bridge.training.checkpointing import (
        _generate_model_state_dict,
        apply_peft_adapter_filter_to_state_dict,
    )
    from megatron.bridge.training.model_load_save import temporary_distributed_context
    from megatron.bridge.training.utils.checkpoint_utils import read_run_config

    bridge = AutoBridge.from_hf_pretrained(hf_model_id, trust_remote_code=trust_remote_code)

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
    provider.finalize()
    provider.register_pre_wrap_hook(lambda chunks: lora(chunks, training=False))

    ctx = temporary_distributed_context(backend="gloo")
    ctx.__enter__()

    model = provider.provide_distributed_model(
        wrap_with_ddp=False,
        use_cpu_initialization=True,
        init_model_with_meta_device=False,
    )

    sharded_sd = _generate_model_state_dict(model, {})
    sharded_sd = apply_peft_adapter_filter_to_state_dict(sharded_sd, lora)
    loaded_sd = dist_checkpointing.load(sharded_sd, str(ckpt_path))
    model_key = "model" if "model" in loaded_sd else next(k for k in loaded_sd if k.startswith("model"))
    model[0].load_state_dict(loaded_sd[model_key], strict=False)

    return bridge, model, lora, ctx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run adapter verification checks."""
    args = parse_args()
    k = args.top_k
    all_pass = True

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id, trust_remote_code=args.trust_remote_code)

    # ------------------------------------------------------------------
    # 0) Read adapter_config.json
    # ------------------------------------------------------------------
    adapter_cfg_path = Path(args.hf_adapter_path) / "adapter_config.json"
    with open(adapter_cfg_path) as f:
        adapter_cfg = json.load(f)
    print(
        f"\nadapter_config.json: r={adapter_cfg['r']}, lora_alpha={adapter_cfg['lora_alpha']}, "
        f"target_modules={adapter_cfg.get('target_modules')}"
    )

    # ------------------------------------------------------------------
    # 1) HF base model logits
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading HF base model ...")
    hf_base = AutoModelForCausalLM.from_pretrained(
        args.hf_model_id,
        torch_dtype=torch.float32,
        trust_remote_code=args.trust_remote_code,
    )
    base_logits = _forward_logits(hf_base, tokenizer, args.prompt)
    _print_top_k("HF base (no adapter)", base_logits, tokenizer, k)
    del hf_base

    # ------------------------------------------------------------------
    # 2) PEFT library loading check
    # ------------------------------------------------------------------
    print("\n[Step 2] Loading adapter with PEFT library ...")
    from peft import PeftModel

    peft_base = AutoModelForCausalLM.from_pretrained(
        args.hf_model_id,
        torch_dtype=torch.float32,
        trust_remote_code=args.trust_remote_code,
    )
    peft_model = PeftModel.from_pretrained(peft_base, args.hf_adapter_path)
    peft_model.eval()
    peft_logits = _forward_logits(peft_model, tokenizer, args.prompt)
    _print_top_k("HF PEFT", peft_logits, tokenizer, k)
    del peft_model, peft_base

    peft_vs_base = (peft_logits - base_logits).abs().max().item()
    if peft_vs_base < 1e-6:
        print("\n  FAIL: PEFT model logits are identical to base model.")
        print("  PEFT failed to load the adapter weights from the safetensors file.")
        all_pass = False
    else:
        print(f"\n  Adapter effect on logits: max diff from base = {peft_vs_base:.6e}  PASS")

    if not args.megatron_peft_checkpoint:
        print("\n\nSkipping Megatron-side checks (--megatron-peft-checkpoint not provided).")
        if all_pass:
            print("PASSED")
        else:
            raise SystemExit("FAILED: see details above")
        return

    # ------------------------------------------------------------------
    # 3) Megatron: load model with LoRA, export merged weights
    # ------------------------------------------------------------------
    print("\n[Step 3] Building Megatron model with LoRA from checkpoint ...")
    bridge, mg_model, lora, dist_ctx = _build_megatron_lora_model(
        args.hf_model_id,
        args.megatron_peft_checkpoint,
        args.trust_remote_code,
    )

    try:
        mg_merged_sd: dict[str, torch.Tensor] = {}
        for name, tensor in bridge.export_hf_weights(mg_model, cpu=True, merge_adapter_weights=True):
            mg_merged_sd[name] = tensor
    finally:
        dist_ctx.__exit__(None, None, None)

    # ------------------------------------------------------------------
    # 4) Logit-level verification (top-k)
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Top-{k} logit verification ...")

    mg_hf = AutoModelForCausalLM.from_pretrained(
        args.hf_model_id,
        torch_dtype=torch.float32,
        trust_remote_code=args.trust_remote_code,
    )
    mg_hf.load_state_dict(mg_merged_sd, strict=True)
    mg_logits = _forward_logits(mg_hf, tokenizer, args.prompt)
    _print_top_k("Megatron merged", mg_logits, tokenizer, k)
    del mg_hf

    if not _compare_top_k("PEFT vs Megatron merged", peft_logits, mg_logits, tokenizer, k):
        all_pass = False

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    if all_pass:
        print("  PASSED: adapter export is correct")
    else:
        print("  FAILED: see details above")
    print(f"{'=' * 70}")

    if not all_pass:
        raise SystemExit("Adapter verification failed")


if __name__ == "__main__":
    main()
