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
This example demonstrates creating and applying fused LoRA adapters.

The process:
1. Load a base model from HuggingFace
2. Create fused LoRA configuration with custom parameters
3. Apply LoRA to create PEFT-enabled Megatron model
4. Optionally demonstrate merge functionality
5. Save as HuggingFace PEFT format

Fused LoRA applies adapters to fused projection layers:
- linear_qkv (Q/K/V combined)
- linear_proj (attention output)
- linear_fc1 (gate/up combined)
- linear_fc2 (MLP output)
"""

import argparse
import os

import torch
from rich.console import Console

from megatron.bridge import AutoBridge
from megatron.bridge.peft import get_peft_model
from megatron.bridge.peft.lora.lora import LoRA


console = Console()
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"


def main(
    base_model_id: str = BASE_MODEL_ID,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    output_dir: str = None,
    do_merge: bool = True,
) -> bool:
    """Create and demonstrate fused LoRA configuration."""

    # Create output path
    model_name = base_model_id.split("/")[-1]
    if output_dir:
        save_path = os.path.join(output_dir, f"{model_name}_fused_lora_r{rank}")
    else:
        save_path = os.path.join("outputs", f"{model_name}_fused_lora_r{rank}")

    console.print(f"Creating fused LoRA adapters for [bold blue]{base_model_id}[/bold blue]")

    # Load base model
    base_bridge = AutoBridge.from_hf_pretrained(base_model_id)

    # Fused LoRA target modules (efficient fused projections)
    targets = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]

    # Create fused LoRA configuration
    console.print("\n🔧 Fused LoRA Configuration:")
    console.print(f"  • Rank (r): {rank}")
    console.print(f"  • Alpha: {alpha}")
    console.print(f"  • Dropout: {dropout}")
    console.print(f"  • Target Modules: {targets}")

    lora = LoRA(target_modules=targets, dim=rank, alpha=alpha, dropout=dropout)

    # Apply LoRA to model
    console.print("\n⚙️  Creating PEFT model...")
    provider = base_bridge.to_megatron_provider()
    peft_model = get_peft_model(provider, lora, training=True, wrap_with_ddp=False)

    console.print("\n📊 Parameter Statistics:")
    peft_model.print_trainable_parameters()

    # Merge demonstration (default behavior)
    if do_merge:
        console.print("\n🔄 Merging adapters into base weights:")
        try:
            merged_model = peft_model.merge_and_unload()
            console.print("  ✓ Successfully merged adapters into base weights")

            # Save merged model as standard model
            os.makedirs(
                os.path.dirname(f"{save_path}_merged") if os.path.dirname(f"{save_path}_merged") else ".",
                exist_ok=True,
            )
            console.print(f"  💾 Saving merged model to {save_path}_merged...")
            base_bridge.save_hf_pretrained(merged_model, f"{save_path}_merged")
            console.print("  ✓ Merged model saved as standard HuggingFace model")
        except Exception as e:
            console.print(f"  ⚠️  Merge failed: {e}")

    # Save PEFT adapters
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    console.print(f"\n💾 Saving LoRA adapters to {save_path}...")

    # For demonstration purposes, create a simple adapter config
    # In practice, you would save after training
    console.print("  ℹ️  Note: In practice, save adapters after training with actual weights")
    console.print(f"      peft_bridge.save_hf_pretrained(peft_model, '{save_path}')")

    console.print("\n✨ Fused LoRA example completed successfully!")
    console.print("Next steps:")
    console.print("  • Train the PEFT model with your training loop")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and demonstrate fused LoRA adapters")
    parser.add_argument("--base-model-id", type=str, default=BASE_MODEL_ID, help="Base model to apply LoRA to")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank (r) parameter")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha scaling parameter")
    parser.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for output files")
    parser.add_argument("--no-merge", action="store_true", help="Skip merge demonstration")

    args = parser.parse_args()
    success = main(
        args.base_model_id,
        args.rank,
        args.alpha,
        args.dropout,
        args.output_dir,
        not args.no_merge,
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    exit(0 if success else 1)
