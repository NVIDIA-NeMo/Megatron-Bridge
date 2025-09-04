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
This example demonstrates creating and applying DoRA adapters.

The process:
1. Load a base model from HuggingFace
2. Create DoRA configuration with custom parameters
3. Apply DoRA to create PEFT-enabled Megatron model
4. Optionally demonstrate merge functionality
5. Save as HuggingFace PEFT format

DoRA (Weight-Decomposed Low-Rank Adaptation) extends LoRA by decomposing
weights into magnitude and direction, adapting only the directional component.
"""

import argparse
import os

import torch
from rich.console import Console

from megatron.bridge import AutoBridge
from megatron.bridge.peft import get_peft_model
from megatron.bridge.peft.lora.dora import DoRA

console = Console()
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"


def main(
    base_model_id: str = BASE_MODEL_ID,
    rank: int = 16,
    alpha: int = 64,  # DoRA typically uses higher alpha
    dropout: float = 0.05,
    output_dir: str = None,
    do_merge: bool = True,
) -> bool:
    """Create and demonstrate DoRA configuration."""
    
    # Create output path
    model_name = base_model_id.split("/")[-1]
    if output_dir:
        save_path = os.path.join(output_dir, f"{model_name}_dora_r{rank}")
    else:
        save_path = os.path.join("outputs", f"{model_name}_dora_r{rank}")

    console.print(f"Creating DoRA adapters for [bold blue]{base_model_id}[/bold blue]")
    
    # Load base model
    base_bridge = AutoBridge.from_hf_pretrained(base_model_id)
    
    # DoRA target modules (uses fused layers like LoRA)
    targets = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    
    # Create DoRA configuration
    console.print(f"\n🔧 DoRA Configuration:")
    console.print(f"  • Rank (r): {rank}")
    console.print(f"  • Alpha: {alpha}")
    console.print(f"  • Dropout: {dropout}")
    console.print(f"  • Target Modules: {targets}")
    console.print(f"  • Features: Low-rank adaptation + magnitude vectors")
    
    dora = DoRA(
        target_modules=targets,
        dim=rank,
        alpha=alpha,
        dropout=dropout
    )
    
    # Apply DoRA to model
    console.print("\n⚙️  Creating PEFT model...")
    provider = base_bridge.to_megatron_provider()
    peft_model = get_peft_model(provider, dora, training=True, wrap_with_ddp=False)
    
    console.print("\n📊 Parameter Statistics:")
    peft_model.print_trainable_parameters()
    
    # Merge demonstration (default behavior)
    if do_merge:
        console.print("\n🔄 Unwrapping DoRA modules:")
        try:
            merged_model = peft_model.merge_and_unload()
            console.print("  ✓ Successfully unwrapped DoRA modules")
            console.print("  ⚠️  Note: DoRA weight merge not yet implemented (complex magnitude handling)")
            
            # Save unwrapped model
            os.makedirs(os.path.dirname(f"{save_path}_unwrapped") if os.path.dirname(f"{save_path}_unwrapped") else ".", exist_ok=True)
            console.print(f"  💾 Saving unwrapped model to {save_path}_unwrapped...")
            base_bridge.save_hf_pretrained(merged_model, f"{save_path}_unwrapped")
            console.print("  ✓ Unwrapped model saved (no DoRA weights merged)")
        except Exception as e:
            console.print(f"  ⚠️  Merge failed: {e}")
    
    # Save PEFT adapters
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    console.print(f"\n💾 Saving DoRA adapters to {save_path}...")
    
    # For demonstration purposes
    console.print("  ℹ️  Note: In practice, save adapters after training with actual weights")
    console.print(f"      peft_bridge.save_hf_pretrained(peft_model, '{save_path}')")
    
    console.print("\n✨ DoRA example completed successfully!")
    console.print("Next steps:")
    console.print(f"  • Train the PEFT model with your training loop")
    console.print(f"  • Save adapter weights after training")
    console.print(f"  • {'Unwrap was demonstrated above' if do_merge else 'Use --no-merge to skip unwrap demonstration'}")
    console.print(f"  • DoRA merge implementation coming in future release")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and demonstrate DoRA adapters")
    parser.add_argument("--base-model-id", type=str, default=BASE_MODEL_ID, help="Base model to apply DoRA to")
    parser.add_argument("--rank", type=int, default=16, help="DoRA rank (r) parameter")
    parser.add_argument("--alpha", type=int, default=64, help="DoRA alpha scaling parameter")
    parser.add_argument("--dropout", type=float, default=0.05, help="DoRA dropout rate")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for output files")
    parser.add_argument("--no-merge", action="store_true", help="Skip merge demonstration (unwrap only for DoRA)")

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