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
Mixtral fine-tuning with Megatron-Bridge.

This script demonstrates how to fine-tune Mixtral models using Megatron-Bridge,
with support for distributed training across multiple GPUs and nodes.

Examples:
    # Single node, 8 GPUs
    torchrun --nproc_per_node=8 examples/mixtral/train_mixtral.py \\
        --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \\
        --data_path=/path/to/data \\
        --output_path=/path/to/checkpoints \\
        --expert_model_parallel_size=8

    # Multi-node (2 nodes, 8 GPUs each)
    # Node 0:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \\
        --master_addr=<node0_addr> --master_port=6000 \\
        examples/mixtral/train_mixtral.py \\
        --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \\
        --data_path=/path/to/data \\
        --output_path=/path/to/checkpoints

    # Node 1:
    torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \\
        --master_addr=<node0_addr> --master_port=6000 \\
        examples/mixtral/train_mixtral.py \\
        --hf_model_path="mistralai/Mixtral-8x7B-v0.1" \\
        --data_path=/path/to/data \\
        --output_path=/path/to/checkpoints
"""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from transformers import AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.utils.common_utils import print_rank_0


def setup_distributed():
    """Initialize distributed training environment."""
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)


def load_model(args):
    """Load and configure Mixtral model."""
    if args.load:
        # Load from pre-converted Megatron checkpoint
        print_rank_0(f"Loading Mixtral model from checkpoint: {args.load}...")

        # Create provider directly with Mixtral 8x7B config
        from megatron.bridge.models import MixtralModelProvider
        provider = MixtralModelProvider(
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            expert_model_parallel_size=args.expert_model_parallel_size,
            context_parallel_size=args.context_parallel_size,
            seq_length=args.seq_length,
            params_dtype=torch.bfloat16 if args.bf16 else torch.float32,
            bf16=args.bf16,
            fp16=args.fp16,
            # Sequence parallelism required when using TP > 1 with MoE
            sequence_parallel=args.tensor_model_parallel_size > 1,
            # Enable activation checkpointing for memory efficiency
            # Selective recompute checkpoints attention and MoE activations
            recompute_granularity="selective" if args.recompute_activations else None,
            recompute_modules=["core_attn", "moe"] if args.recompute_activations else None,
        )

        # Skip weight initialization since we're loading from checkpoint
        provider.perform_initialization = False

    else:
        # Load from HuggingFace
        print_rank_0(f"Loading Mixtral model from {args.hf_model_path}...")

        # Load model using AutoBridge
        bridge = AutoBridge.from_hf_pretrained(
            args.hf_model_path,
            trust_remote_code=args.trust_remote_code,
        )

        # Convert to Megatron provider
        provider = bridge.to_megatron_provider()

        # Configure parallelism
        provider.tensor_model_parallel_size = args.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
        provider.expert_model_parallel_size = args.expert_model_parallel_size
        provider.context_parallel_size = args.context_parallel_size

        # Override training-specific settings
        provider.seq_length = args.seq_length
        provider.params_dtype = torch.bfloat16 if args.bf16 else torch.float32
        provider.bf16 = args.bf16
        provider.fp16 = args.fp16

        # Sequence parallelism required when using TP > 1 with MoE
        provider.sequence_parallel = args.tensor_model_parallel_size > 1

        # Enable activation checkpointing if requested
        if args.recompute_activations:
            provider.recompute_granularity = "selective"
            provider.recompute_modules = ["core_attn", "moe"]

        # MoE-specific settings
        if args.moe_aux_loss_coeff is not None:
            provider.moe_aux_loss_coeff = args.moe_aux_loss_coeff

    print_rank_0(f"\nModel configuration:")
    print_rank_0(f"  Hidden size: {provider.hidden_size}")
    print_rank_0(f"  Num layers: {provider.num_layers}")
    print_rank_0(f"  Num experts: {provider.num_moe_experts}")
    print_rank_0(f"  Router top-k: {provider.moe_router_topk}")
    print_rank_0(f"  Sequence length: {provider.seq_length}")
    print_rank_0(f"  Parallelism: TP={args.tensor_model_parallel_size}, "
                 f"PP={args.pipeline_model_parallel_size}, "
                 f"EP={args.expert_model_parallel_size}, "
                 f"CP={args.context_parallel_size}")

    return provider


def create_dataloader(args, tokenizer):
    """Create training dataloader.

    Supports both mock data (for testing) and real data (for production).
    Mock data generates random tokens, useful for testing without real datasets.
    """
    from torch.utils.data import DataLoader, Dataset

    if args.mock_data:
        # Mock dataset for testing without real data
        class MockDataset(Dataset):
            """Mock dataset that generates random tokens."""

            def __init__(self, seq_length, vocab_size, num_samples=1000):
                self.seq_length = seq_length
                self.vocab_size = vocab_size
                self.num_samples = num_samples

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                # Generate random token IDs
                input_ids = torch.randint(
                    0, self.vocab_size, (self.seq_length,), dtype=torch.long
                )
                return {
                    "input_ids": input_ids,
                    "labels": input_ids.clone(),
                }

        print_rank_0("Using mock data for testing")
        dataset = MockDataset(
            seq_length=args.seq_length,
            vocab_size=args.vocab_size,
            num_samples=args.train_iters * args.global_batch_size,
        )
    else:
        # Real dataset for production training
        class SimpleTextDataset(Dataset):
            """Simple text dataset for demonstration."""

            def __init__(self, data_path, tokenizer, seq_length):
                self.tokenizer = tokenizer
                self.seq_length = seq_length
                # In production, load your actual dataset here
                # For now, using dummy data for demonstration
                self.data = [
                    "This is example training text.",
                    "Another example for fine-tuning.",
                ] * 1000  # Repeat for demonstration

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                text = self.data[idx]
                tokens = self.tokenizer(
                    text,
                    max_length=self.seq_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                return {
                    "input_ids": tokens["input_ids"].squeeze(0),
                    "labels": tokens["input_ids"].squeeze(0),
                }

        print_rank_0(f"Loading data from {args.data_path}")
        dataset = SimpleTextDataset(args.data_path, tokenizer, args.seq_length)

    # Use DistributedSampler for multi-GPU training
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        num_workers=args.num_workers if not args.mock_data else 0,
        pin_memory=True,
    )

    return dataloader


def train(args):
    """Main training function."""
    # Validate arguments
    if not args.mock_data and not args.data_path:
        raise ValueError("Either --mock_data or --data_path must be specified")
    if not args.load and not args.hf_model_path:
        raise ValueError("Either --load or --hf_model_path must be specified")

    # Setup distributed
    setup_distributed()

    # Load model
    # Note: provide_distributed_model() handles model parallelism initialization
    provider = load_model(args)

    # Finalize provider configuration
    provider.finalize()

    # Configure DDP based on args (actual DP size determined by provider)
    # Distributed optimizer requires DP > 1, but we let provider handle this
    use_distributed_optimizer = args.use_distributed_optimizer
    overlap_grad_reduce = args.overlap_grad_reduce and use_distributed_optimizer

    if args.overlap_grad_reduce and not use_distributed_optimizer:
        print_rank_0("Note: overlap_grad_reduce disabled (requires distributed optimizer)")

    # gradient_accumulation_fusion requires distributed optimizer (uses main_grad buffers)
    if not use_distributed_optimizer and provider.gradient_accumulation_fusion:
        print_rank_0("Note: gradient_accumulation_fusion disabled (requires distributed optimizer)")
        provider.gradient_accumulation_fusion = False

    print_rank_0(f"Distributed optimizer: {'enabled' if use_distributed_optimizer else 'disabled'}")
    print_rank_0(f"Overlap grad reduce: {'enabled' if overlap_grad_reduce else 'disabled'}")
    print_rank_0(f"Gradient accumulation fusion: {'enabled' if provider.gradient_accumulation_fusion else 'disabled'}")

    # Configure DDP
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=overlap_grad_reduce,
        use_distributed_optimizer=use_distributed_optimizer,
    )

    # Get distributed model (this initializes model parallelism)
    print_rank_0("\nInitializing distributed model...")
    # Provider determines if DDP is needed based on data parallel world size
    # We always wrap with DDP - provider will handle it correctly
    model = provider.provide_distributed_model(
        ddp_config=ddp_config,
        wrap_with_ddp=True,
    )

    # Now parallel state is initialized - print actual configuration
    ep_size = parallel_state.get_expert_model_parallel_world_size()
    if ep_size > 1:
        dp_size = parallel_state.get_expert_data_parallel_world_size()
        print_rank_0(f"\nParallel configuration:")
        print_rank_0(f"  Expert model parallel size: {ep_size}")
        print_rank_0(f"  Expert data parallel size: {dp_size}")
    else:
        dp_size = parallel_state.get_data_parallel_world_size()
        print_rank_0(f"\nParallel configuration:")
        print_rank_0(f"  Data parallel size: {dp_size}")

    # Load checkpoint weights if loading from pre-converted checkpoint
    if args.load:
        print_rank_0(f"Loading weights from {args.load}...")
        from pathlib import Path

        checkpoint_base = Path(args.load)

        # Read iteration from tracker file (Megatron format)
        tracker_filename = checkpoint_base / "latest_checkpointed_iteration.txt"
        if tracker_filename.exists():
            with open(tracker_filename, 'r') as f:
                iteration_str = f.read().strip()
                if iteration_str == 'release':
                    iteration_dir = checkpoint_base / 'release'
                else:
                    iteration = int(iteration_str)
                    iteration_dir = checkpoint_base / f"iter_{iteration:07d}"
            print_rank_0(f"  Loading from iteration: {iteration_str}")
        else:
            # Fallback: try to find iteration directory
            print_rank_0(f"  Warning: Tracker file not found, looking for iteration directories...")
            iter_dirs = sorted(checkpoint_base.glob("iter_*"))
            if iter_dirs:
                iteration_dir = iter_dirs[-1]  # Use latest
                print_rank_0(f"  Found iteration directory: {iteration_dir.name}")
            else:
                raise FileNotFoundError(f"No checkpoint iterations found in {checkpoint_base}")

        # Load using distributed checkpointing (torch_dist format)
        print_rank_0(f"  Loading from {iteration_dir}...")
        from megatron.core import dist_checkpointing
        from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy

        # Build sharded state dict for loading
        state_dict = {
            "model": model[0].sharded_state_dict(),
        }

        # Load using distributed checkpointing
        load_strategy = get_default_load_sharded_strategy(str(iteration_dir))
        state_dict = dist_checkpointing.load(
            state_dict,
            str(iteration_dir),
            load_strategy,
        )

        # Load the model weights
        model[0].load_state_dict(state_dict["model"], strict=True)
        print_rank_0(f"✓ Loaded checkpoint from {iteration_dir}")

        # Set to training mode
        model[0].train()

    # Load tokenizer (only needed for real data)
    if not args.mock_data:
        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_model_path,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        tokenizer = None  # Not needed for mock data

    # Create dataloader
    print_rank_0("Creating dataloader...")
    dataloader = create_dataloader(args, tokenizer)

    # Setup optimizer
    print_rank_0("Setting up optimizer...")
    if args.optimizer == "adamw":
        print_rank_0(f"Using AdamW optimizer (requires ~2× param memory for optimizer states)")
        optimizer = torch.optim.AdamW(
            model[0].parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "sgd":
        print_rank_0(f"Using SGD optimizer with momentum={args.momentum} (memory-efficient)")
        optimizer = torch.optim.SGD(
            model[0].parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Setup learning rate scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.train_iters,
        eta_min=args.min_lr,
    )

    # Training loop
    print_rank_0(f"\nStarting training for {args.train_iters} iterations...")
    print_rank_0(f"Global batch size: {args.global_batch_size}")
    print_rank_0(f"Micro batch size: {args.micro_batch_size}")

    global_step = 0
    accumulation_steps = args.global_batch_size // (
        args.micro_batch_size * parallel_state.get_data_parallel_world_size()
    )

    model[0].train()

    for epoch in range(args.num_epochs):
        dataloader.sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(dataloader):
            if global_step >= args.train_iters:
                break

            # Move batch to GPU
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()

            # Create position_ids and attention_mask
            batch_size, seq_length = input_ids.shape
            position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            attention_mask = torch.ones(batch_size, seq_length, device=input_ids.device, dtype=torch.bool)

            # Forward pass
            outputs = model[0](
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs

            # Compute loss using vocab-parallel cross entropy
            # Shift for causal language modeling (predict next token)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # vocab_parallel_cross_entropy expects [seq_len, batch, vocab_size]
            # but we have [batch, seq_len, vocab_size], so transpose
            from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy

            shift_logits_transposed = shift_logits.transpose(0, 1).contiguous()  # [seq_len, batch, vocab]
            shift_labels_transposed = shift_labels.transpose(0, 1).contiguous()  # [seq_len, batch]

            loss_tensor = vocab_parallel_cross_entropy(
                shift_logits_transposed,
                shift_labels_transposed,
            )
            # Average loss over all tokens
            loss = loss_tensor.mean()

            # Backward pass
            loss = loss / accumulation_steps
            loss.backward()

            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model[0].parameters(), args.clip_grad)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % args.log_interval == 0:
                    print_rank_0(
                        f"Step {global_step}/{args.train_iters} | "
                        f"Loss: {loss.item() * accumulation_steps:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    )

                # Checkpointing
                if global_step % args.save_interval == 0:
                    save_checkpoint(model, optimizer, global_step, args)

        if global_step >= args.train_iters:
            break

    # Save final checkpoint
    print_rank_0("\nTraining complete. Saving final checkpoint...")
    save_checkpoint(model, optimizer, global_step, args, final=True)


def save_checkpoint(model, optimizer, global_step, args, final=False):
    """Save training checkpoint."""
    if parallel_state.get_data_parallel_rank() != 0:
        return

    checkpoint_dir = Path(args.output_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = "final_checkpoint" if final else f"checkpoint_{global_step}"
    checkpoint_path = checkpoint_dir / checkpoint_name

    print_rank_0(f"Saving checkpoint to {checkpoint_path}...")

    # In production, use proper distributed checkpointing
    # For now, simplified version
    torch.save(
        {
            "global_step": global_step,
            "model_state_dict": model[0].state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    print_rank_0(f"Checkpoint saved successfully.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Mixtral fine-tuning with Megatron-Bridge",
        epilog="Example: torchrun --nproc_per_node=2 train_mixtral.py "
               "--hf_model_path='mistralai/Mixtral-8x7B-v0.1' --mock_data",
    )

    # Model arguments
    parser.add_argument(
        "--hf_model_path",
        type=str,
        default=None,
        help="HuggingFace model path or identifier (not needed if using --load)",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to pre-converted Megatron checkpoint directory",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading from HuggingFace",
    )

    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to training data (not required if using --mock_data)",
    )
    parser.add_argument(
        "--mock_data",
        action="store_true",
        help="Use mock/synthetic data for testing (no real dataset required)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Vocabulary size (used with mock data)",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=4096,
        help="Sequence length",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )

    # Training arguments
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=1,
        help="Micro batch size per GPU",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=256,
        help="Global batch size",
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=10000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Optimizer arguments
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "sgd"],
        help="Optimizer type (adamw or sgd). Use sgd for memory-constrained scenarios.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD optimizer",
    )

    # Parallelism arguments
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
        help="Expert model parallel size",
    )
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Context parallel size",
    )

    # DDP arguments
    parser.add_argument(
        "--use_distributed_optimizer",
        action="store_true",
        help="Use distributed optimizer",
    )
    parser.add_argument(
        "--overlap_grad_reduce",
        action="store_true",
        default=True,
        help="Overlap gradient reduction with computation",
    )
    parser.add_argument(
        "--recompute_activations",
        action="store_true",
        help="Enable selective activation checkpointing to save memory (checkpoints attention and MoE activations)",
    )

    # MoE arguments
    parser.add_argument(
        "--moe_aux_loss_coeff",
        type=float,
        default=None,
        help="MoE auxiliary loss coefficient",
    )

    # Precision arguments
    parser.add_argument("--bf16", action="store_true", default=True, help="Use BF16 precision")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")

    # Logging and checkpointing
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for checkpoints",
    )
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save interval")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
