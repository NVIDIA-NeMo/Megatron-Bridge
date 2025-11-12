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
Mixtral text generation demonstrating Expert Parallelism (EP) for MoE models.

This script showcases how to use Expert Model Parallelism with Mixtral,
a Mixture-of-Experts (MoE) model. For basic text generation without MoE-specific
features, see examples/conversion/hf_to_megatron_generate_text.py

Key MoE-specific features demonstrated:
- Expert Model Parallelism (EP): Distributes experts across GPUs
- Proper parallelism configuration for MoE models
- Loading and running Mixtral with distributed experts

Examples:
    # Single GPU generation
    python examples/mixtral/generate_text.py \\
        --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \\
        --prompt="What is quantum computing?"

    # Multi-GPU with Expert Parallelism (recommended for large MoE models)
    torchrun --nproc_per_node=2 examples/mixtral/generate_text.py \\
        --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \\
        --expert_model_parallel_size=2 \\
        --prompt="Explain machine learning"

    # Load from pre-converted Megatron checkpoint
    torchrun --nproc_per_node=2 examples/mixtral/generate_text.py \\
        --load="/path/to/megatron/checkpoint" \\
        --expert_model_parallel_size=2 \\
        --prompt="Tell me about AI"
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from transformers import AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.models import MixtralModelProvider
from megatron.bridge.utils.common_utils import get_last_rank, print_rank_0


class SingleBatchIterator:
    """Iterator that yields a single batch for inference."""

    def __init__(self, input_ids, position_ids):
        self.batch = dict(
            tokens=input_ids,
            position_ids=position_ids,
        )
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def forward_step(data_iterator, model, **kwargs) -> torch.Tensor:
    """Forward step for text generation."""
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }

    def loss_func(output_tensor, non_loss_data=False, **kwargs):
        if non_loss_data:
            return output_tensor
        else:
            return output_tensor, torch.tensor(0.0, device=output_tensor.device)

    return model(**forward_args), loss_func


def generate_text(model, tokenizer, prompt: str, max_tokens: int = 100) -> str:
    """Generate text from a prompt using greedy decoding.

    Args:
        model: The Megatron model
        tokenizer: The tokenizer
        prompt: Input prompt text
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text
    """
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    generated_ids = input_ids.clone()

    print_rank_0(f"Generating {max_tokens} tokens...")

    for step in range(max_tokens):
        # Prepare batch
        position_ids = torch.arange(
            generated_ids.shape[1], device=generated_ids.device
        ).unsqueeze(0)

        data_iterator = SingleBatchIterator(generated_ids, position_ids)

        # Forward pass
        forward_backward_func = get_forward_backward_func()
        output = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=iter(data_iterator),
            model=model,
            num_microbatches=1,
            seq_length=generated_ids.shape[1],
            micro_batch_size=1,
            forward_only=True,
            collect_non_loss_data=True,
        )

        # Get next token from last pipeline stage (greedy decoding)
        if parallel_state.is_pipeline_last_stage():
            logits = output[0]
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        else:
            next_token = torch.zeros((1, 1), dtype=torch.long, device="cuda")

        # Broadcast next token to all ranks
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            dist.broadcast(next_token, src=get_last_rank())

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Append to generated sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        if step % 10 == 0:
            print_rank_0(f"  Step {step}/{max_tokens}")

    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def main(args) -> None:
    """Main function for Mixtral text generation with Expert Parallelism."""

    # Validate arguments
    if not args.hf_model_path and not args.load:
        raise ValueError("Either --hf_model_path or --load must be provided")

    if not args.prompt:
        print_rank_0("Error: --prompt is required")
        sys.exit(1)

    # Initialize distributed environment
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    # Set CUDA device for this rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Print parallelism configuration (important for MoE models)
    print_rank_0("\n" + "=" * 80)
    print_rank_0("Mixtral Generation with Expert Parallelism")
    print_rank_0("=" * 80)
    print_rank_0(f"Parallelism configuration:")
    print_rank_0(f"  Tensor Parallelism (TP):   {args.tensor_model_parallel_size}")
    print_rank_0(f"  Pipeline Parallelism (PP): {args.pipeline_model_parallel_size}")
    print_rank_0(f"  Expert Parallelism (EP):   {args.expert_model_parallel_size} <- MoE-specific")
    print_rank_0("")
    print_rank_0("Note: EP distributes the 8 experts across GPUs for memory efficiency")
    print_rank_0("=" * 80 + "\n")

    if args.load:
        # Load from pre-converted Megatron checkpoint
        print_rank_0(f"Loading Mixtral model from checkpoint: {args.load}...")

        # Create provider with Mixtral 8x7B config
        provider = MixtralModelProvider(
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            expert_model_parallel_size=args.expert_model_parallel_size,
        )

        # Skip weight initialization since we're loading from checkpoint
        provider.perform_initialization = False

        print_rank_0(f"\nModel configuration:")
        print_rank_0(f"  Hidden size: {provider.hidden_size}")
        print_rank_0(f"  Num layers: {provider.num_layers}")
        print_rank_0(f"  Num experts: {provider.num_moe_experts}")
        print_rank_0(f"  Router top-k: {provider.moe_router_topk}\n")

        # Finalize provider configuration
        provider.finalize()

        # Create model (provide_distributed_model handles parallelism initialization)
        model = provider.provide_distributed_model(
            ddp_config=None,
            wrap_with_ddp=False,
        )

        # Load checkpoint
        print_rank_0(f"Loading weights from {args.load}...")
        from pathlib import Path

        checkpoint_base = Path(args.load)

        # Read iteration from tracker file
        tracker_filename = checkpoint_base / "latest_checkpointed_iteration.txt"
        if tracker_filename.exists():
            with open(tracker_filename, 'r') as f:
                iteration_str = f.read().strip()
                if iteration_str == 'release':
                    iteration_dir = checkpoint_base / 'release'
                else:
                    iteration = int(iteration_str)
                    iteration_dir = checkpoint_base / f"iter_{iteration:07d}"
        else:
            # Fallback: try to find iteration directory
            iter_dirs = sorted(checkpoint_base.glob("iter_*"))
            if iter_dirs:
                iteration_dir = iter_dirs[-1]
            else:
                iteration_dir = checkpoint_base

        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        ep_rank = parallel_state.get_expert_model_parallel_rank()

        checkpoint_name = f"mp_rank_{tp_rank:02d}_{pp_rank:03d}_{ep_rank:03d}"
        checkpoint_path = iteration_dir / checkpoint_name / "model_optim_rng.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model[0].load_state_dict(checkpoint['model'])
        print_rank_0(f"✓ Loaded checkpoint from {checkpoint_path}")

        model[0].eval()

    else:
        # Load from HuggingFace
        print_rank_0(f"Loading Mixtral model from HuggingFace: {args.hf_model_path}...")

        bridge = AutoBridge.from_hf_pretrained(
            args.hf_model_path,
            trust_remote_code=args.trust_remote_code,
        )

        # Convert to Megatron provider
        provider = bridge.to_megatron_provider()

        # Override parallelism settings
        # NOTE: Expert parallelism is crucial for large MoE models
        provider.expert_model_parallel_size = args.expert_model_parallel_size
        provider.tensor_model_parallel_size = args.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size

        print_rank_0(f"\nModel configuration:")
        print_rank_0(f"  Hidden size: {provider.hidden_size}")
        print_rank_0(f"  Num layers: {provider.num_layers}")
        print_rank_0(f"  Num experts: {provider.num_moe_experts}")
        print_rank_0(f"  Router top-k: {provider.moe_router_topk}\n")

        # Finalize provider configuration
        provider.finalize()

        # Load distributed model
        model = provider.provide_distributed_model(
            ddp_config=None,
            wrap_with_ddp=False,
        )

        model[0].eval()

    # Load tokenizer
    tokenizer_path = args.hf_model_path or "mistralai/Mixtral-8x7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )

    # Generate text
    print_rank_0(f"Prompt: {args.prompt}\n")

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )

    print_rank_0(f"\n{'=' * 80}")
    print_rank_0("Generated text:")
    print_rank_0(f"{'=' * 80}")
    print_rank_0(f"{generated_text}")
    print_rank_0(f"{'=' * 80}\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Mixtral text generation demonstrating Expert Parallelism (EP) for MoE models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--hf_model_path",
        type=str,
        default=None,
        help="HuggingFace model path or identifier (for loading from HF)",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to Megatron checkpoint directory (for loading pre-converted checkpoint)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading from HuggingFace",
    )

    # Generation arguments
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )

    # Parallelism arguments (MoE-specific: expert_model_parallel_size)
    parser.add_argument(
        "--tensor_model_parallel_size",
        type=int,
        default=1,
        help="Tensor model parallel size",
    )
    parser.add_argument(
        "--pipeline_model_parallel_size",
        type=int,
        default=1,
        help="Pipeline model parallel size",
    )
    parser.add_argument(
        "--expert_model_parallel_size",
        type=int,
        default=1,
        help="Expert model parallel size (MoE-specific: distributes experts across GPUs)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
