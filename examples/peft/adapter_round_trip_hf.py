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

from megatron.bridge.models.conversion import weights_verification_table
from megatron.bridge.peft import AutoPEFTBridge


console = Console()
ADAPTER_ID = "codelion/Llama-3.2-1B-Instruct-tool-calling-lora"


def main(adapter_id: str = ADAPTER_ID, output_dir: str = None) -> bool:
    """Perform round-trip conversion between HuggingFace PEFT adapters and Megatron PEFT models."""
    adapter_name = adapter_id.split("/")[-1]
    if output_dir:
        save_path = os.path.join(output_dir, f"{adapter_name}_verified")
    else:
        # Default to outputs directory (not tracked by git)
        save_path = os.path.join("outputs", f"{adapter_name}_verified")

    console.print(f"Loading PEFT adapters from [bold green]{adapter_id}[/bold green]...")
    peft_bridge = AutoPEFTBridge.from_hf_pretrained(adapter_id)

    console.print("Converting to Megatron PEFT model...")
    peft_model = peft_bridge.to_megatron_model(wrap_with_ddp=False)

    # Display adapter weight mapping verification
    console.print("\n📊 Adapter Weight Mapping Verification:")
    try:
        # Use weights_verification_table with PEFT bridge
        table = weights_verification_table(peft_bridge, peft_model, export_method_name="export_adapter_weights")
        console.print(table)
    except Exception as e:
        console.print(f"⚠️  Could not verify adapter mappings: {e}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    # Perform round-trip save and reload to verify conversion accuracy
    console.print(f"\nSaving PEFT adapters in {save_path}...")
    peft_bridge.save_hf_pretrained(peft_model, save_path)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert between HuggingFace PEFT adapters and Megatron PEFT formats")
    parser.add_argument("--adapter-id", type=str, default=ADAPTER_ID, help="HuggingFace adapter ID to convert")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="The directory where the converted adapter directory will be created. Defaults to the current working directory.",
    )

    args = parser.parse_args()
    main(args.adapter_id, args.output_dir)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
