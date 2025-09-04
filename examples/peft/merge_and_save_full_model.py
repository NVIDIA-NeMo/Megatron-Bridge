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
This example demonstrates how to merge trained LoRA adapters into the base model
and save the result as a complete fine-tuned model.

The process is as follows:
1. Load a base model and pretrained LoRA adapters
2. Apply the adapters to create a PEFT-enabled Megatron model
3. Merge the adapter weights into the base model weights
4. Save the merged model as a standard HuggingFace model (no adapters)
5. Verify that the merged model can be loaded and used normally

This workflow is useful for:
- Creating deployable models from trained adapters
- Sharing fine-tuned models without requiring PEFT library
- Converting adapter-based models to standard model format
- Preparing models for inference frameworks that don't support adapters
"""

import argparse
import os

import torch
from rich.console import Console

from megatron.bridge import AutoBridge
from megatron.bridge.peft import AutoPEFTBridge


console = Console()
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_ADAPTER_MODEL = "codelion/Llama-3.2-1B-Instruct-tool-calling-lora"


def main(
    base_model_id: str = DEFAULT_BASE_MODEL,
    adapter_model_id: str = DEFAULT_ADAPTER_MODEL,
    output_dir: str = None,
    verify_merge: bool = True,
) -> None:
    """Merge LoRA adapters into base model and save as complete model."""
    console.print("🚀 Merging LoRA adapters into base model")
    console.print(f"   Base model: [bold blue]{base_model_id}[/bold blue]")
    console.print(f"   Adapters: [bold green]{adapter_model_id}[/bold green]")

    # Load adapters (base model auto-detected)
    console.print("\n📥 Loading adapters with auto-detected base model...")
    peft_bridge = AutoPEFTBridge.from_hf_pretrained(adapter_model_id)

    # Display adapter information
    console.print("\n📋 Adapter Information:")
    config = peft_bridge.peft_config
    console.print(f"  • PEFT Type: {config.peft_type}")
    console.print(f"  • Rank (r): {config.r}")
    console.print(f"  • Alpha: {config.lora_alpha}")
    console.print(f"  • Target Modules: {len(config.target_modules)} modules")

    # Create PEFT model
    console.print("\n⚙️  Creating PEFT-enabled Megatron model...")
    peft_model = peft_bridge.to_megatron_model(wrap_with_ddp=False)

    console.print("\n📊 Original Parameter Statistics:")
    peft_model.print_trainable_parameters()

    # Optional: Store some adapter weights for verification
    if verify_merge:
        console.print("\n🔍 Storing adapter weights for verification...")
        adapter_state_before = peft_model.adapter_state_dict()
        sample_params = dict(list(adapter_state_before.items())[:3])
        console.print(f"   • Stored {len(sample_params)} sample parameters for comparison")

    # Merge adapters into base weights
    console.print("\n🔄 Merging adapters into base model weights...")
    try:
        merged_model = peft_model.merge_and_unload()
        console.print("   ✅ Merge completed successfully")

        # Verify that adapters are no longer present
        console.print("\n🧪 Verifying merge results...")

        # The merged model should be a list of standard Megatron modules (no adapters)
        has_adapters = False
        for stage in merged_model:
            for name, _ in stage.named_modules():
                if "adapter" in name.lower():
                    has_adapters = True
                    break
            if has_adapters:
                break

        if has_adapters:
            console.print("   ⚠️  Warning: Adapter modules still present after merge")
        else:
            console.print("   ✅ No adapter modules found - merge successful")

        # Save merged model
        model_name = adapter_model_id.split("/")[-1]
        if output_dir:
            save_path = os.path.join(output_dir, f"{model_name}_merged")
        else:
            save_path = f"{model_name}_merged"

        console.print(f"\n💾 Saving merged model: [bold cyan]{save_path}[/bold cyan]")
        # For merged model, save through the base bridge
        base_bridge = peft_bridge._base_bridge
        base_bridge.save_hf_pretrained(merged_model, save_path)
        console.print("   ✅ Merged model saved successfully")

        # Verify the saved model can be loaded
        console.print("\n🔎 Verifying saved model...")
        try:
            verification_bridge = AutoBridge.from_hf_pretrained(save_path)
            console.print("   ✅ Saved model loads successfully")

            # Check that it has the expected configuration
            config = verification_bridge.hf_pretrained.config
            console.print(f"   • Model type: {config.model_type}")
            console.print(f"   • Hidden size: {config.hidden_size}")
            console.print(f"   • Number of layers: {config.num_hidden_layers}")

        except Exception as e:
            console.print(f"   ❌ Failed to load saved model: {e}")
            return False

        console.print("\n✨ Round-trip merge completed successfully!")
        console.print("   The merged model is now a standard fine-tuned model")
        console.print("   It can be used without the PEFT library")

        return True

    except NotImplementedError as e:
        console.print(f"   ❌ Merge not implemented: {e}")
        console.print("   This adapter type doesn't support merging yet")
        return False
    except Exception as e:
        console.print(f"   ❌ Merge failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("--base-model-id", type=str, default=DEFAULT_BASE_MODEL, help="Base HuggingFace model ID")
    parser.add_argument("--adapter-model-id", type=str, default=DEFAULT_ADAPTER_MODEL, help="Adapter model ID")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where merged model will be saved. Defaults to current directory.",
    )
    parser.add_argument("--no-verify", action="store_true", help="Skip merge verification steps")

    args = parser.parse_args()
    success = main(args.base_model_id, args.adapter_model_id, args.output_dir, not args.no_verify)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    exit(0 if success else 1)
