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
This example demonstrates how to apply LoRA adapters from HuggingFace to a Megatron model.

The process is as follows:
1. Load a base HuggingFace model using AutoBridge (e.g., "meta-llama/Llama-3.1-8B")
2. Load pretrained LoRA adapters using AutoPEFTBridge (e.g., "username/llama-lora-math")
3. Apply the adapters to the base model to create a PEFT-enabled Megatron model
4. Optionally demonstrate adapter management (enable/disable, parameter counting)
5. Save the adapted model back to HuggingFace format for sharing or deployment

This workflow is useful for:
- Applying community-created LoRA adapters to base models
- Fine-tuning with adapters in Megatron's distributed training environment
- Converting between HuggingFace PEFT and Megatron PEFT formats
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
    show_parameters: bool = True,
    use_auto_detection: bool = True,
) -> None:
    """Apply LoRA adapters to a base model and demonstrate PEFT functionality."""
    console.print(f"🚀 Loading LoRA adapters: [bold green]{adapter_model_id}[/bold green]")

    if use_auto_detection:
        console.print("   Using auto-detected base model from adapter config...")
        peft_bridge = AutoPEFTBridge.from_hf_pretrained(adapter_model_id)
    else:
        console.print(f"   Loading base model: [bold blue]{base_model_id}[/bold blue]")
        base_bridge = AutoBridge.from_hf_pretrained(base_model_id)
        peft_bridge = AutoPEFTBridge.from_hf_pretrained(adapter_model_id, base_bridge)

    # Display adapter information
    console.print("\n📋 Adapter Configuration:")
    config = peft_bridge.peft_config
    console.print(f"  • PEFT Type: {config.peft_type}")
    console.print(f"  • Rank (r): {config.r}")
    console.print(f"  • Alpha: {config.lora_alpha}")
    console.print(f"  • Dropout: {config.lora_dropout}")
    console.print(f"  • Target Modules: {', '.join(config.target_modules)}")
    console.print(f"  • Use DoRA: {getattr(config, 'use_dora', False)}")

    console.print("\n⚙️  Creating PEFT-enabled Megatron model...")
    peft_model = peft_bridge.to_megatron_model(wrap_with_ddp=False)

    if show_parameters:
        console.print("\n📊 Parameter Statistics:")
        peft_model.print_trainable_parameters()

    console.print("\n🎛️  Demonstrating adapter control:")

    # Show adapter enable/disable functionality
    console.print("  • Disabling adapters...")
    peft_model.disable_adapters()
    console.print("    ✓ Adapters disabled - model now behaves like base model")

    console.print("  • Re-enabling adapters...")
    peft_model.enable_adapters()
    console.print("    ✓ Adapters re-enabled - model includes adapter effects")

    # Demonstrate merge functionality
    console.print("\n🔄 Demonstrating merge and unload:")
    try:
        _ = peft_model.merge_and_unload()  # Demonstrate merge functionality
        console.print("  ✓ Successfully merged adapters into base weights")
        console.print("    Model can now be saved as a standard fine-tuned model")
    except NotImplementedError as e:
        console.print(f"  ⚠️  Merge not implemented: {e}")

    # Save back to HuggingFace format
    if output_dir:
        adapter_name = adapter_model_id.split("/")[-1]
        save_path = os.path.join(output_dir, f"{adapter_name}_converted")
    else:
        save_path = "converted_adapters"

    console.print(f"\n💾 Saving adapters to HuggingFace format: [bold cyan]{save_path}[/bold cyan]")
    peft_bridge.save_hf_pretrained(peft_model, save_path)
    console.print("  ✓ Adapters saved successfully")

    console.print("\n✨ Example completed successfully!")
    console.print("   You can now load the converted adapters with:")
    console.print("   >>> from peft import AutoPeftModelForCausalLM")
    console.print(f"   >>> model = AutoPeftModelForCausalLM.from_pretrained('{save_path}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply LoRA adapters to a base model")
    parser.add_argument("--base-model-id", type=str, default=DEFAULT_BASE_MODEL, help="Base HuggingFace model ID")
    parser.add_argument("--adapter-model-id", type=str, default=DEFAULT_ADAPTER_MODEL, help="LoRA adapter model ID")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where converted adapters will be saved. Defaults to current directory.",
    )
    parser.add_argument("--no-show-parameters", action="store_true", help="Skip displaying parameter statistics")
    parser.add_argument(
        "--no-auto-detect",
        action="store_true",
        help="Disable auto-detection of base model (use explicit base-model-id instead)",
    )

    args = parser.parse_args()
    main(
        args.base_model_id,
        args.adapter_model_id,
        args.output_dir,
        not args.no_show_parameters,
        not args.no_auto_detect,
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
