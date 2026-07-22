#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Unified block-diffusion inference entry point.

A single CLI that routes to the right block-diffusion model's generation loop
via ``--model``. The two models keep their own generation loops (different
attention semantics: LLaDA1.5 is fully bidirectional; NemotronLabsDiffusion is
block-causal with a KV cache and also supports an autoregressive ``--mode ar``).
Only the per-step sampling math is shared, in
:mod:`megatron.bridge.diffusion.common.dllm`.

This dispatcher lives at the example layer (not in ``src/.../common``) on
purpose: examples are allowed to import model packages, whereas making a
``common`` module import upward into both model families would invert the
dependency direction and risk circular / heavy imports. Each model's modules are
imported **lazily** inside its adapter, so selecting one model never loads the
other's dependencies (e.g. picking ``llada15`` never imports NemotronLabs'
``flex_attention`` / ``torch.compile`` stack).

Examples:
    LLaDA1.5 (pure diffusion)::

        python examples/models/diffusion/inference_dllm.py \\
            --model llada15 \\
            --hf-model /path/to/GSAI-ML/LLaDA-1.5 \\
            --megatron-path /path/to/llada15_megatron_ckpt \\
            --prompts "The capital of France is" \\
            --gen-length 64 --block-length 32 --steps 64

    NemotronLabsDiffusion (block diffusion)::

        torchrun --nproc_per_node=4 examples/models/diffusion/inference_dllm.py \\
            --model nemotron_labs_diffusion \\
            --hf-model mistralai/Ministral-3-8B-Base-2512 \\
            --megatron-path /path/to/ar_to_dlm_8b \\
            --prompts "The capital of France is" \\
            --gen-length 256 --block-length 32 --steps 256 --tp 4

    NemotronLabsDiffusion autoregressive mode::

        python examples/models/diffusion/inference_dllm.py \\
            --model nemotron_labs_diffusion --mode ar \\
            --hf-model mistralai/Ministral-3-3B-Base-2512 \\
            --megatron-path /path/to/ar_to_dlm_3b \\
            --prompts "Once upon a time" --max-new-tokens 128
"""

import argparse
import os

import torch
import torch.distributed as dist


# Per-model default mask token id (LLaDA1.5 uses 126336; NemotronLabs uses 100).
_DEFAULT_MASK_ID = {"llada15": 126336, "nemotron_labs_diffusion": 100}


# ---------------------------------------------------------------------------
# LLaDA1.5 adapter  (pure diffusion, single-GPU, fully bidirectional)
# ---------------------------------------------------------------------------


def _llada15_load(args):
    """Build/load a LLaDA1.5 Megatron GPTModel from a Megatron checkpoint."""
    from megatron.bridge import AutoBridge
    from megatron.bridge.diffusion.conversion.llada15 import llada15_bridge  # noqa: F401

    bridge = AutoBridge.from_hf_pretrained(args.hf_model, trust_remote_code=True)
    model = bridge.load_megatron_model(args.megatron_path, wrap_with_ddp=False)
    if isinstance(model, list):
        model = model[0]
    return model.cuda().eval()


def _llada15_generate(model, prompt_ids, args):
    """Run LLaDA1.5 block-diffusion generation; returns full token ids."""
    from megatron.bridge.diffusion.models.llada15.inference_llada15 import generate_block_diffusion

    return generate_block_diffusion(
        model=model,
        input_ids=prompt_ids,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        temperature=args.temperature,
        mask_token_id=args.mask_token_id,
        eos_token_id=args.eos_token_id,
        eos_early_stop=not args.no_eos_early_stop,
        pad_token_id=args.pad_token_id,
    )


# ---------------------------------------------------------------------------
# NemotronLabsDiffusion adapter  (AR + dLLM, optional TP)
# ---------------------------------------------------------------------------


def _nemotron_load(args):
    """Load a NemotronLabsDiffusion Megatron model and wire the TP group."""
    from megatron.bridge import AutoBridge
    from megatron.bridge.diffusion.conversion.nemotron_labs_diffusion import (  # noqa: F401
        nemotron_labs_diffusion_bridge,
    )

    bridge = AutoBridge.from_hf_pretrained(args.hf_model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = args.tp
    provider.pipeline_model_parallel_size = 1
    provider.pipeline_dtype = torch.bfloat16
    provider.params_dtype = torch.bfloat16
    provider.seq_length = args.seq_length
    provider.finalize()
    provider.initialize_model_parallel(seed=0)

    from megatron.bridge.training.model_load_save import build_and_load_model

    models = build_and_load_model(checkpoint_path=args.megatron_path, model_cfg=provider, skip_temp_dist_context=True)
    model = models[0] if isinstance(models, list) else models
    model = model.cuda().eval()

    if args.tp > 1:
        from megatron.core import parallel_state as mpu

        from megatron.bridge.diffusion.models.nemotron_labs_diffusion.inference_nemotron_labs_diffusion import (
            set_tp_group,
        )

        rank = int(os.getenv("RANK", 0))
        set_tp_group(mpu.get_tensor_model_parallel_group(), src_global_rank=(rank // args.tp) * args.tp)
    return model


def _nemotron_generate(model, prompt_ids, args):
    """Run NemotronLabsDiffusion generation (AR or dLLM); returns full token ids."""
    from megatron.bridge.diffusion.models.nemotron_labs_diffusion.inference_nemotron_labs_diffusion import (
        generate_ar,
        generate_dllm,
    )

    if args.mode == "ar":
        return generate_ar(
            model=model,
            prompt=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            eos_token_id=args.eos_token_id,
        )
    output, _nfe, _timing = generate_dllm(
        model=model,
        prompt=prompt_ids,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        temperature=args.temperature,
        mask_id=args.mask_token_id,
    )
    return output


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTERS = {
    "llada15": {"load": _llada15_load, "generate": _llada15_generate, "supports_ar": False},
    "nemotron_labs_diffusion": {"load": _nemotron_load, "generate": _nemotron_generate, "supports_ar": True},
}


def parse_args():
    """Parse command-line arguments for the unified dLLM inference CLI."""
    p = argparse.ArgumentParser(description="Unified block-diffusion inference")
    p.add_argument("--model", required=True, choices=sorted(ADAPTERS), help="Which diffusion model to run")
    p.add_argument("--hf-model", required=True, help="HF model id/path (config + tokenizer + bridge dispatch)")
    p.add_argument("--megatron-path", required=True, help="Megatron checkpoint directory")
    p.add_argument("--prompts", action="append", required=True, help="Prompt(s); repeat for multiple")
    p.add_argument("--mode", choices=["dllm", "ar"], default="dllm", help="ar is NemotronLabs-only")
    p.add_argument("--gen-length", type=int, default=64, help="Tokens to generate (dLLM)")
    p.add_argument("--block-length", type=int, default=32, help="Denoising block size (dLLM)")
    p.add_argument("--steps", type=int, default=64, help="Total denoising steps (dLLM)")
    p.add_argument("--max-new-tokens", type=int, default=128, help="Tokens to generate (AR mode)")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    p.add_argument("--mask-token-id", type=int, default=None, help="Override mask token id (model default if unset)")
    p.add_argument("--eos-token-id", type=int, default=None, help="EOS id (tokenizer default if unset)")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallel degree (NemotronLabs)")
    p.add_argument("--seq-length", type=int, default=4096, help="Max sequence length (NemotronLabs load)")
    p.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable the tokenizer chat template (use for base models). By default the "
        "chat template is applied when the tokenizer defines one (e.g. LLaDA1.5-Instruct).",
    )
    p.add_argument(
        "--no-eos-early-stop",
        action="store_true",
        help="Disable early stopping at EOS (LLaDA1.5). Use for short-answer prompts that "
        "otherwise emit EOS at the first position and return empty.",
    )
    return p.parse_args()


def _encode_prompts(tokenizer, prompts, use_chat_template, model):
    """Tokenize prompts, optionally wrapping each in the tokenizer's chat template.

    LLaDA uses **left**-padding because Megatron applies RoPE by physical sequence
    index. This shifts a short prompt and its generated suffix together, preserving
    their relative positions; right-padding would insert a positional gap. Other
    adapters retain their existing right-padding behavior.
    """
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        texts = [
            tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
    else:
        texts = prompts
    padding_side = "left" if model == "llada15" else "right"
    return tokenizer(texts, return_tensors="pt", padding=True, padding_side=padding_side)


def main():
    """Route to the selected model's loader + generation loop and print replies."""
    args = parse_args()
    adapter = ADAPTERS[args.model]

    if args.mode == "ar" and not adapter["supports_ar"]:
        raise SystemExit(f"--mode ar is not supported by --model {args.model} (pure diffusion).")

    # EOS early-stop is batch-global (it returns as soon as ANY row emits EOS),
    # which truncates longer rows in a mixed-length batch. Auto-disable it for
    # multi-prompt batches; per-row trimming still happens at decode time.
    if len(args.prompts) > 1 and not args.no_eos_early_stop:
        print("[info] multiple prompts: disabling EOS early-stop (it is batch-global).")
        args.no_eos_early_stop = True

    if args.mask_token_id is None:
        args.mask_token_id = _DEFAULT_MASK_ID[args.model]

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    elif not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        torch.cuda.set_device(0)
        dist.init_process_group(backend="nccl", world_size=1, rank=0)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.eos_token_id is None:
        args.eos_token_id = tokenizer.eos_token_id
    # Padding id used to build the key-padding mask for batched generation.
    args.pad_token_id = tokenizer.pad_token_id

    model = adapter["load"](args)

    inputs = _encode_prompts(
        tokenizer,
        args.prompts,
        use_chat_template=not args.no_chat_template,
        model=args.model,
    )
    prompt_ids = inputs.input_ids.cuda()
    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        output = adapter["generate"](model, prompt_ids, args)

    if rank == 0:
        generated = output[:, prompt_len:]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for i, (prompt, text) in enumerate(zip(args.prompts, texts)):
            print(f"\n--- Prompt {i + 1}/{len(args.prompts)} ({args.model}, mode={args.mode}) ---")
            print(f"USER: {prompt}")
            print(f"OUTPUT: {text.strip()}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
