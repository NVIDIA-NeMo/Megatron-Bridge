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
This example demonstrates round-trip conversion for PEFT adapters between
HuggingFace and Megatron formats.

The process is as follows:
1. Load pretrained LoRA adapters from HuggingFace format
2. Apply them to a Megatron model for distributed training
3. Extract the adapter weights and save them back to HuggingFace format
4. Verify that the round-trip conversion preserves adapter weights accurately

This workflow validates:
- Bidirectional conversion accuracy between HF PEFT and Megatron PEFT
- Adapter weight preservation during format conversions
- Parameter mapping correctness for different adapter layouts
- Distributed training compatibility with adapter weights
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


def compare_adapter_weights(original_state: dict, converted_state: dict, tolerance: float = 1e-5) -> bool:
    """Compare adapter weights between original and converted states."""
    console.print("\n🔍 Comparing adapter weights:")

    # Get common keys
    original_keys = set(original_state.keys())
    converted_keys = set(converted_state.keys())

    if original_keys != converted_keys:
        missing_in_converted = original_keys - converted_keys
        extra_in_converted = converted_keys - original_keys

        if missing_in_converted:
            console.print(f"  ❌ Missing keys in converted: {missing_in_converted}")
        if extra_in_converted:
            console.print(f"  ❌ Extra keys in converted: {extra_in_converted}")

        return False

    # Compare values
    mismatches = 0
    total_keys = len(original_keys)

    for key in original_keys:
        original_tensor = original_state[key]
        converted_tensor = converted_state[key]

        if original_tensor.shape != converted_tensor.shape:
            console.print(f"  ❌ Shape mismatch for {key}: {original_tensor.shape} vs {converted_tensor.shape}")
            mismatches += 1
            continue

        if not torch.allclose(original_tensor, converted_tensor, atol=tolerance):
            max_diff = torch.max(torch.abs(original_tensor - converted_tensor)).item()
            console.print(f"  ❌ Value mismatch for {key}: max difference = {max_diff}")
            mismatches += 1
        else:
            console.print(f"  ✅ {key}: shapes match, values within tolerance")

    success_rate = (total_keys - mismatches) / total_keys * 100
    console.print(f"\n📈 Comparison Results: {total_keys - mismatches}/{total_keys} parameters match ({success_rate:.1f}%)")

    return mismatches == 0


def main(
    base_model_id: str = DEFAULT_BASE_MODEL,
    adapter_model_id: str = DEFAULT_ADAPTER_MODEL,
    output_dir: str | None = None,
    tolerance: float = 1e-5,
    use_auto_detection: bool = True
) -> None:
    """Perform round-trip conversion and verification for PEFT adapters."""
    console.print(f"🚀 Starting adapter round-trip test")
    console.print(f"   Adapters: [bold green]{adapter_model_id}[/bold green]")

    # Step 1: Load adapters
    console.print(f"\n📥 Loading adapters...")
    peft_bridge = AutoPEFTBridge.from_hf_pretrained(adapter_model_id)

    if use_auto_detection:
        console.print(f"\n🔍 Using auto-detected base model from adapter config...")
        base_bridge = None  # Let auto-detection handle it
    else:
        console.print(f"   Base model: [bold blue]{base_model_id}[/bold blue]")
        console.print(f"\n📥 Loading base model...")
        base_bridge = AutoBridge.from_hf_pretrained(base_model_id)

    # Display original adapter info
    config = peft_bridge.peft_config
    console.print(f"   • Type: {config.peft_type}")
    console.print(f"   • Layout: {'Canonical' if 'proj' in str(config.target_modules) else 'Fused'}")
    console.print(f"   • Target modules: {len(config.target_modules)} modules")

    # Step 2: Apply to Megatron model
    console.print(f"\n⚙️  Converting to Megatron PEFT model...")
    peft_model = peft_bridge.to_megatron_model(wrap_with_ddp=False)  # Auto-detects base model

    # Extract original state for comparison
    console.print(f"\n📋 Extracting original adapter state...")
    original_adapter_state = peft_model.adapter_state_dict()
    console.print(f"   • Found {len(original_adapter_state)} adapter parameters")

    # Step 3: Save to HuggingFace format
    temp_save_dir = "temp_round_trip_adapters"
    console.print(f"\n💾 Saving adapters to temporary directory...")
    peft_bridge.save_hf_pretrained(peft_model, temp_save_dir)

    # Step 4: Load back from saved format
    console.print(f"\n📤 Loading adapters from saved format...")
    peft_bridge_reloaded = AutoPEFTBridge.from_hf_pretrained(temp_save_dir)
    # Note: Reloaded adapters may not have base_model_name_or_path, so we need the original base_bridge
    base_bridge = AutoBridge.from_hf_pretrained(base_model_id)  # Fallback for reloaded adapters
    peft_model_reloaded = peft_bridge_reloaded.to_megatron_model(base_bridge, wrap_with_ddp=False)

    # Extract reloaded state for comparison
    console.print(f"\n📋 Extracting reloaded adapter state...")
    reloaded_adapter_state = peft_model_reloaded.adapter_state_dict()
    console.print(f"   • Found {len(reloaded_adapter_state)} adapter parameters")

    # Step 5: Compare states
    weights_match = compare_adapter_weights(original_adapter_state, reloaded_adapter_state, tolerance)

    if weights_match:
        console.print("\n✅ Round-trip conversion successful!")
        console.print("   All adapter weights preserved accurately")
    else:
        console.print("\n❌ Round-trip conversion failed!")
        console.print("   Some adapter weights differ beyond tolerance")

    # Cleanup and optional final save
    console.print(f"\n🧹 Cleaning up temporary files...")
    import shutil
    shutil.rmtree(temp_save_dir, ignore_errors=True)

    if output_dir and weights_match:
        adapter_name = adapter_model_id.split("/")[-1]
        final_save_path = os.path.join(output_dir, f"{adapter_name}_verified")
        console.print(f"\n💾 Saving verified adapters to: [bold cyan]{final_save_path}[/bold cyan]")
        peft_bridge.save_hf_pretrained(peft_model, final_save_path)
        console.print("   ✓ Verified adapters saved successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test round-trip adapter conversion")
    parser.add_argument("--base-model-id", type=str, default=DEFAULT_BASE_MODEL, help="Base HuggingFace model ID")
    parser.add_argument("--adapter-model-id", type=str, default=DEFAULT_ADAPTER_MODEL, help="Adapter model ID to test")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save verified adapters (only if round-trip succeeds)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Numerical tolerance for weight comparison"
    )

    args = parser.parse_args()
    success = main(args.base_model_id, args.adapter_model_id, args.output_dir, args.tolerance)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    # Exit with status code indicating success/failure
    exit(0 if success else 1)
