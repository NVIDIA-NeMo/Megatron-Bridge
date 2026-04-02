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
Nemotron Diffusion inference script.

Runs text generation over one or more prompts using a Megatron-format
NemotronDiffusion checkpoint. Supports both dLLM (block diffusion) and AR modes.

Examples:
    Single prompt, dLLM mode (default):
        $ torchrun --nproc_per_node=4 examples/diffusion/recipes/nemotron_diffusion/inference_nemotron.py \\
            --megatron-path /path/to/checkpoints/ar_to_dlm_8b \\
            --hf-model mistralai/Ministral-3-8B-Base-2512 \\
            --prompts "The capital of France is"

    AR mode:
        $ python examples/diffusion/recipes/nemotron_diffusion/inference_nemotron.py \\
            --megatron-path /path/to/checkpoints/ar_to_dlm_3b \\
            --hf-model mistralai/Ministral-3-3B-Base-2512 \\
            --mode ar \\
            --max-new-tokens 128 \\
            --prompts "Once upon a time"

    Multiple prompts with custom diffusion settings:
        $ torchrun --nproc_per_node=4 examples/diffusion/recipes/nemotron_diffusion/inference_nemotron.py \\
            --megatron-path /path/to/checkpoints/ar_to_dlm_8b \\
            --hf-model mistralai/Ministral-3-8B-Base-2512 \\
            --prompts "Prompt one" --prompts "Prompt two" \\
            --gen-length 256 --block-length 32 --diffusion-steps 256
"""

import argparse
import os

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

# Register NemotronDiffusionBridge so AutoBridge uses NemotronDiffusionAttention
import megatron.bridge.diffusion.conversion.nemotron_diffusion.nemotron_diffusion_bridge  # noqa: F401
from megatron.bridge import AutoBridge
from megatron.bridge.diffusion.models.nemotron_diffusion.inference_nemotron_diffusion import (
    generate_ar,
    generate_dllm,
    set_tp_group,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Nemotron Diffusion inference")
    parser.add_argument(
        "--megatron-path",
        type=str,
        required=True,
        help="Path to the Megatron-Bridge checkpoint directory (e.g. .../ar_to_dlm_8b)",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path (used for config and tokenizer)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        action="append",
        required=True,
        help="Input prompt(s). Can be specified multiple times.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dllm", "ar"],
        default="dllm",
        help="Generation mode: 'dllm' for block diffusion (default), 'ar' for autoregressive",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Number of tokens to generate (AR mode)",
    )
    parser.add_argument(
        "--gen-length",
        type=int,
        default=256,
        help="Total tokens to generate (dLLM mode, must be divisible by --block-length)",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=32,
        help="Denoising block size (dLLM mode)",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=256,
        help="Total denoising steps across all blocks (dLLM mode)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--mask-token-id",
        type=int,
        default=100,
        help="Mask token ID used during diffusion (default: 100)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallelism degree (must match the saved checkpoint)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )
    return parser.parse_args()


def load_model(args):
    """Load the NemotronDiffusion model from a Megatron checkpoint via AutoBridge."""
    bridge = AutoBridge.from_hf_pretrained(args.hf_model, torch_dtype=torch.bfloat16)
    model_provider = bridge.to_megatron_provider(load_weights=False)
    model_provider.tensor_model_parallel_size = args.tp
    model_provider.pipeline_model_parallel_size = 1
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.params_dtype = torch.bfloat16
    model_provider.seq_length = args.seq_length
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)

    from megatron.bridge.training.model_load_save import build_and_load_model

    megatron_models = build_and_load_model(
        checkpoint_path=args.megatron_path,
        model_cfg=model_provider,
        skip_temp_dist_context=True,
        dist_ckpt_strictness="ignore_all",
    )
    model = megatron_models[0] if isinstance(megatron_models, list) else megatron_models
    return model.cuda().eval()


def main():
    """Entry point for Nemotron Diffusion inference."""
    args = parse_args()

    # Distributed setup
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = load_model(args)

    # Wire up TP group for broadcast (no-op when tp=1)
    if args.tp > 1:
        from megatron.core import parallel_state as mpu

        tp_group = mpu.get_tensor_model_parallel_group()
        tp_src = (rank // args.tp) * args.tp
        set_tp_group(tp_group, src_global_rank=tp_src)

    # Tokenize prompts
    inputs = tokenizer(args.prompts, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids = inputs.input_ids.cuda()
    prompt_len = prompt_ids.shape[1]

    # Generate
    with torch.no_grad():
        if args.mode == "ar":
            output = generate_ar(
                model=model,
                prompt=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            output, nfe, _ = generate_dllm(
                model=model,
                prompt=prompt_ids,
                gen_length=args.gen_length,
                block_length=args.block_length,
                steps=args.diffusion_steps,
                temperature=args.temperature,
                mask_id=args.mask_token_id,
            )

    # Decode and print (rank 0 only)
    if rank == 0 or (args.tp > 1 and rank % args.tp == 0):
        generated_ids = output[:, prompt_len:]
        texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for i, (prompt, text) in enumerate(zip(args.prompts, texts)):
            print(f"\n--- Prompt {i + 1} ---")
            print(f"Input:  {prompt}")
            print(f"Output: {text}")
        if args.mode == "dllm" and rank == 0:
            print(f"\nNFE: {nfe}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
