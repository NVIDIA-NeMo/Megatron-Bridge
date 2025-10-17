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

"""Test Llama Nemotron generation with multi-GPU support."""

import argparse

import torch
import torch.distributed as dist
from megatron.core import parallel_state

from megatron.bridge import AutoBridge


def initialize_distributed(tp=1, pp=1):
    """Initialize distributed environment for multi-GPU inference."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Set device
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    # Initialize model parallel
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
    )

    return rank, world_size, device


def test_generation(model_id: str, tp: int = 1, pp: int = 1, prompt: str = "Hello, how are you?"):
    """Test generation with a Llama Nemotron model.

    Args:
        model_id: HuggingFace model ID or local path
        tp: Tensor parallelism size
        pp: Pipeline parallelism size
        prompt: Input prompt for generation
    """
    rank, world_size, device = initialize_distributed(tp=tp, pp=pp)

    if rank == 0:
        print("=" * 80)
        print("Testing Llama Nemotron Generation")
        print("=" * 80)
        print(f"Model: {model_id}")
        print(f"World Size: {world_size}")
        print(f"TP: {tp}, PP: {pp}")
        print(f"Device: cuda:{device}")
        print("=" * 80)

    try:
        # Load model via AutoBridge
        if rank == 0:
            print(f"\n[Rank {rank}] Loading model from HuggingFace...")

        # For heterogeneous models, need trust_remote_code
        trust_remote_code = "Super" in model_id or "Ultra" in model_id

        bridge = AutoBridge.from_hf_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
        )

        if rank == 0:
            print(f"[Rank {rank}] Converting to Megatron provider...")

        provider = bridge.to_megatron_provider(load_weights=True)

        # Set parallelism
        provider.tensor_model_parallel_size = tp
        provider.pipeline_model_parallel_size = pp

        if rank == 0:
            print(f"[Rank {rank}] Provider config:")
            print(f"  - Layers: {provider.num_layers}")
            print(f"  - Hidden: {provider.hidden_size}")
            print(f"  - Heads: {provider.num_attention_heads}")
            print(f"  - KV Channels: {provider.kv_channels}")
            print(f"  - Seq Length: {provider.seq_length}")

        # Finalize and create model
        provider.finalize()

        if rank == 0:
            print(f"\n[Rank {rank}] Building distributed model...")

        models = provider.provide_distributed_model(wrap_with_ddp=False)

        # Handle both single model and list of models (PP)
        if isinstance(models, list):
            for m in models:
                m.eval()
            model = models[0]  # Use first model for forward pass
        else:
            models.eval()
            model = models

        if rank == 0:
            print(f"[Rank {rank}] Model loaded successfully!")
            print(f"[Rank {rank}] Model type: {type(model).__name__}")

        # Tokenize input
        tokenizer = bridge.hf_pretrained.tokenizer
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()

        if rank == 0:
            print(f"\n[Rank {rank}] Running generation...")
            print(f"[Rank {rank}] Input: '{prompt}'")
            print(f"[Rank {rank}] Input tokens: {input_ids.shape}")

        # Simple forward pass (not actual generation, just verification)
        with torch.no_grad():
            # Get model output for the input
            # Unwrap DDP if needed
            test_model = model.module if hasattr(model, "module") else model
            output = test_model(input_ids)

        if rank == 0:
            print(f"[Rank {rank}] Forward pass successful!")
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            print(f"[Rank {rank}] Output logits shape: {logits.shape}")

            # Decode next token
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            next_word = tokenizer.decode(next_token[0])

            print(f"[Rank {rank}] Next predicted token: '{next_word}'")
            print("\n" + "=" * 80)
            print("✅ GENERATION TEST PASSED!")
            print("=" * 80)

        # Cleanup
        parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()

        return True

    except Exception as e:
        if rank == 0:
            print(f"\n❌ Error during generation test: {e}")
            import traceback

            traceback.print_exc()

        if dist.is_initialized():
            dist.destroy_process_group()

        return False


def main():
    parser = argparse.ArgumentParser(description="Test Llama Nemotron generation")
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="HuggingFace model ID or local path (e.g., nvidia/Llama-3.1-Nemotron-70B-Instruct-HF)",
    )
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallelism size (default: 2 for 2-GPU system)")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size (default: 1)")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Input prompt for generation")

    args = parser.parse_args()

    # Validate
    world_size = args.tp * args.pp
    if world_size > torch.cuda.device_count():
        print(f"Error: Requested {world_size} processes but only {torch.cuda.device_count()} GPUs available")
        return 1

    success = test_generation(model_id=args.model_id, tp=args.tp, pp=args.pp, prompt=args.prompt)

    return 0 if success else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
