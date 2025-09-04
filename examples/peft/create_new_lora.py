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
This example demonstrates how to create a new LoRA configuration and apply it to a base model.

The process is as follows:
1. Load a base HuggingFace model using AutoBridge
2. Create a new LoRA configuration with custom parameters
3. Apply the LoRA transform to create a PEFT-enabled Megatron model
4. Show parameter statistics and adapter management
5. Optionally save the model configuration for future training

This workflow is useful for:
- Starting new fine-tuning projects with custom LoRA configurations
- Experimenting with different LoRA hyperparameters
- Setting up distributed training with PEFT adapters
- Understanding the parameter efficiency of different LoRA settings
"""

import argparse

import torch
from rich.console import Console

from megatron.bridge import AutoBridge
from megatron.bridge.peft import get_peft_model
from megatron.bridge.peft.lora.lora import LoRA
from megatron.bridge.peft.lora.canonical_lora import CanonicalLoRA


console = Console()
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-1B"


def main(
    base_model_id: str = DEFAULT_BASE_MODEL,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_canonical: bool = False,
    target_modules: str = "all"
) -> None:
    """Create and apply a new LoRA configuration to a base model."""
    console.print(f"🚀 Loading base model: [bold blue]{base_model_id}[/bold blue]")
    base_bridge = AutoBridge.from_hf_pretrained(base_model_id)
    
    # Determine target modules
    if target_modules == "all":
        if use_canonical:
            targets = ["linear_q", "linear_k", "linear_v", "linear_proj", "linear_fc1_gate", "linear_fc1_up", "linear_fc2"]
        else:
            targets = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    elif target_modules == "attention":
        if use_canonical:
            targets = ["linear_q", "linear_k", "linear_v", "linear_proj"]
        else:
            targets = ["linear_qkv", "linear_proj"]
    elif target_modules == "mlp":
        if use_canonical:
            targets = ["linear_fc1_gate", "linear_fc1_up", "linear_fc2"]
        else:
            targets = ["linear_fc1", "linear_fc2"]
    else:
        targets = target_modules.split(",")
    
    # Create LoRA configuration
    console.print(f"\n🔧 Creating {'Canonical' if use_canonical else 'Fused'} LoRA configuration:")
    console.print(f"  • Rank (r): {lora_rank}")
    console.print(f"  • Alpha: {lora_alpha}")
    console.print(f"  • Dropout: {lora_dropout}")
    console.print(f"  • Target Modules: {', '.join(targets)}")
    
    if use_canonical:
        lora = CanonicalLoRA(
            target_modules=targets,
            dim=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout
        )
    else:
        lora = LoRA(
            target_modules=targets,
            dim=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout
        )
    
    console.print(f"\n⚙️  Applying LoRA to model provider...")
    provider = base_bridge.to_megatron_provider()
    peft_model = get_peft_model(provider, lora, training=True, wrap_with_ddp=False)
    
    console.print("\n📊 Parameter Statistics:")
    peft_model.print_trainable_parameters()
    
    console.print("\n🎛️  Demonstrating adapter management:")
    
    # Test adapter state dict extraction
    console.print("  • Extracting adapter-only state dict...")
    adapter_state = peft_model.adapter_state_dict()
    adapter_param_count = sum(p.numel() for p in adapter_state.values())
    console.print(f"    ✓ Found {len(adapter_state)} adapter parameters ({adapter_param_count:,} total elements)")
    
    # Show some example parameter names
    if adapter_state:
        sample_params = list(adapter_state.keys())[:3]
        console.print("    Example parameter names:")
        for param in sample_params:
            console.print(f"      - {param}")
    
    # Demonstrate enable/disable
    console.print("\n  • Testing adapter enable/disable:")
    peft_model.disable_adapters()
    console.print("    ✓ Adapters disabled")
    
    peft_model.enable_adapters()
    console.print("    ✓ Adapters re-enabled")
    
    console.print("\n💡 Next Steps:")
    console.print("   1. Use this PEFT model for training with your training loop")
    console.print("   2. Save adapter checkpoints during training")
    console.print("   3. Merge and unload adapters when training is complete")
    console.print("   4. Export adapters to HuggingFace format for sharing")
    
    console.print(f"\n✨ LoRA configuration created and applied successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and apply new LoRA configuration")
    parser.add_argument("--base-model-id", type=str, default=DEFAULT_BASE_MODEL, help="Base HuggingFace model ID")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (r) parameter")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling parameter")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--use-canonical", action="store_true", help="Use canonical LoRA instead of fused")
    parser.add_argument(
        "--target-modules",
        type=str, 
        default="all",
        help="Target modules: 'all', 'attention', 'mlp', or comma-separated list"
    )

    args = parser.parse_args()
    main(
        args.base_model_id,
        args.lora_rank,
        args.lora_alpha, 
        args.lora_dropout,
        args.use_canonical,
        args.target_modules
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()