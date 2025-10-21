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
PEFT Adapter Checkpoint Conversion Example

This script demonstrates how to convert PEFT adapters between HuggingFace and Megatron
checkpoint formats using the AutoPEFTBridge import_ckpt and export_ckpt methods.

Features:
- Import HuggingFace PEFT adapters to Megatron checkpoint format
- Export Megatron PEFT checkpoints to HuggingFace adapter format
- Support for various adapter types (LoRA, DoRA, etc.)
- Configurable base model and conversion settings

Workflow:
1. Import: Load HF adapters + base model → Convert to Megatron → Save checkpoint
2. Export: Load Megatron checkpoint → Extract adapters → Save HF format

Usage examples:
  # Import HuggingFace LoRA adapters to Megatron checkpoint
  python examples/peft/convert_adapter_checkpoints.py import \\
    --adapter-id codelion/Llama-3.2-1B-Instruct-tool-calling-lora \\
    --megatron-path ./checkpoints/llama_lora

  # Import with explicit base model
  python examples/peft/convert_adapter_checkpoints.py import \\
    --adapter-id ./my_adapters \\
    --base-model meta-llama/Llama-3.2-1B \\
    --megatron-path ./checkpoints/my_lora

  # Export Megatron PEFT checkpoint to HuggingFace format
  python examples/peft/convert_adapter_checkpoints.py export \\
    --adapter-id codelion/Llama-3.2-1B-Instruct-tool-calling-lora \\
    --megatron-path ./checkpoints/llama_lora \\
    --hf-path ./exports/llama_lora_hf

  # Export without progress bar (useful for scripting)
  python examples/peft/convert_adapter_checkpoints.py export \\
    --adapter-id ./my_adapters \\
    --megatron-path ./checkpoints/my_lora \\
    --hf-path ./exports/my_lora_hf \\
    --no-progress
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from rich.console import Console

from megatron.bridge import AutoBridge
from megatron.bridge.peft import AutoPEFTBridge


console = Console()


def validate_path(path: str, must_exist: bool = False) -> Path:
    """Validate and convert string path to Path object."""
    path_obj = Path(path)
    if must_exist and not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")
    return path_obj


def import_hf_adapters_to_megatron(
    adapter_id: str,
    megatron_path: str,
    base_model: Optional[str] = None,
    trust_remote_code: bool = False,
) -> None:
    """
    Import HuggingFace PEFT adapters and save as a Megatron checkpoint.

    This function loads HuggingFace PEFT adapters (LoRA, DoRA, etc.), applies them
    to a base model, converts the combined model to Megatron format, and saves it
    as a Megatron checkpoint that can be used for distributed training.

    Args:
        adapter_id: HuggingFace adapter ID or path to adapter directory
        megatron_path: Directory path where the Megatron checkpoint will be saved
        base_model: Optional base model ID/path. If not provided, loaded from adapter config
        trust_remote_code: Allow custom model code execution
    """
    console.print("\n[bold cyan]🔄 Starting PEFT adapter import[/bold cyan]")
    console.print(f"   Adapter: [green]{adapter_id}[/green]")
    console.print(f"   Target: [green]{megatron_path}[/green]")

    # Prepare optional base bridge if base model is specified
    base_bridge = None
    if base_model:
        console.print(f"   Base model: [green]{base_model}[/green]")
        console.print("\n📥 Loading base model...")
        base_bridge = AutoBridge.from_hf_pretrained(base_model, trust_remote_code=trust_remote_code)
        console.print("✅ Base model loaded successfully")

    # Import using the convenience method
    console.print(f"\n📥 Loading HuggingFace PEFT adapters from: {adapter_id}")
    console.print("🔧 Converting to Megatron format...")
    console.print("💾 Saving checkpoint...")

    AutoPEFTBridge.import_ckpt(
        hf_adapter_path=adapter_id,
        megatron_path=megatron_path,
        base_bridge=base_bridge,
        trust_remote_code=trust_remote_code,
    )

    console.print(f"\n[bold green]✅ Successfully imported adapters to:[/bold green] {megatron_path}")

    # Verify the checkpoint was created
    checkpoint_path = Path(megatron_path)
    if checkpoint_path.exists():
        console.print("\n📁 Checkpoint structure:")
        for item in sorted(checkpoint_path.iterdir())[:10]:  # Show first 10 items
            if item.is_dir():
                console.print(f"   📂 [blue]{item.name}/[/blue]")
            else:
                size_mb = item.stat().st_size / (1024 * 1024)
                console.print(f"   📄 {item.name} [dim]({size_mb:.2f} MB)[/dim]")

    console.print("\n[bold]Next steps:[/bold]")
    console.print("   • Use this checkpoint for Megatron distributed training")
    console.print("   • Export back to HuggingFace format after training (see export command)")


def export_megatron_to_hf_adapters(
    adapter_id: str,
    megatron_path: str,
    hf_path: str,
    show_progress: bool = True,
    trust_remote_code: bool = False,
) -> None:
    """
    Export a Megatron PEFT checkpoint to HuggingFace adapter format.

    This function loads a Megatron checkpoint containing PEFT adapters, extracts
    the adapter weights, and saves them in HuggingFace PEFT format for easy
    sharing, deployment, or use with HuggingFace inference tools.

    Args:
        adapter_id: Original HuggingFace adapter ID (used to recreate bridge)
        megatron_path: Directory path where the Megatron checkpoint is stored
        hf_path: Directory path where the HuggingFace adapter will be saved
        show_progress: Display progress bar during adapter export
        trust_remote_code: Allow custom model code execution
    """
    console.print("\n[bold cyan]🔄 Starting PEFT adapter export[/bold cyan]")
    console.print(f"   Source: [green]{megatron_path}[/green]")
    console.print(f"   Target: [green]{hf_path}[/green]")

    # Validate megatron checkpoint exists
    checkpoint_path = validate_path(megatron_path, must_exist=True)
    console.print(f"\n✅ Found Megatron checkpoint: {checkpoint_path}")

    # Look for checkpoint structure
    iter_dirs = sorted([d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.startswith("iter_")])
    if iter_dirs:
        latest_iter = iter_dirs[-1]
        console.print(f"📂 Using iteration: [blue]{latest_iter.name}[/blue]")

    # Load the PEFT bridge with adapter configuration
    # This recreates the bridge structure needed to export adapters
    console.print(f"\n📥 Loading adapter configuration from: {adapter_id}")
    peft_bridge = AutoPEFTBridge.from_hf_pretrained(adapter_id, trust_remote_code=trust_remote_code)

    # Export using the convenience method
    console.print("\n📤 Exporting adapters to HuggingFace format...")
    peft_bridge.export_ckpt(
        megatron_path=megatron_path,
        hf_path=hf_path,
        show_progress=show_progress,
    )

    console.print(f"\n[bold green]✅ Successfully exported adapters to:[/bold green] {hf_path}")

    # Verify the export was created
    export_path = Path(hf_path)
    if export_path.exists():
        console.print("\n📁 Export structure:")
        for item in sorted(export_path.iterdir()):
            if item.is_dir():
                console.print(f"   📂 [blue]{item.name}/[/blue]")
            else:
                size_kb = item.stat().st_size / 1024
                console.print(f"   📄 {item.name} [dim]({size_kb:.2f} KB)[/dim]")

    console.print("\n[bold]🔍 Load this adapter with HuggingFace:[/bold]")
    console.print("   [dim]from peft import PeftModel, AutoModelForCausalLM[/dim]")
    console.print("   [dim]base_model = AutoModelForCausalLM.from_pretrained('base_model_name')[/dim]")
    console.print(f"   [dim]model = PeftModel.from_pretrained(base_model, '{hf_path}')[/dim]")


def main():
    """Main function to handle command line arguments and execute conversions."""
    parser = argparse.ArgumentParser(
        description="Convert PEFT adapters between HuggingFace and Megatron checkpoint formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Conversion direction")

    # Import subcommand (HF adapters -> Megatron checkpoint)
    import_parser = subparsers.add_parser(
        "import", help="Import HuggingFace PEFT adapters to Megatron checkpoint format"
    )
    import_parser.add_argument(
        "--adapter-id",
        required=True,
        help="HuggingFace adapter ID or path to adapter directory (e.g., 'username/llama-lora')",
    )
    import_parser.add_argument(
        "--megatron-path", required=True, help="Directory path where the Megatron checkpoint will be saved"
    )
    import_parser.add_argument(
        "--base-model",
        help="Base model ID or path (if not specified, loaded from adapter config)",
    )
    import_parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code execution")

    # Export subcommand (Megatron checkpoint -> HF adapters)
    export_parser = subparsers.add_parser(
        "export", help="Export Megatron PEFT checkpoint to HuggingFace adapter format"
    )
    export_parser.add_argument(
        "--adapter-id",
        required=True,
        help="Original HuggingFace adapter ID (used to recreate bridge structure)",
    )
    export_parser.add_argument(
        "--megatron-path", required=True, help="Directory path where the Megatron checkpoint is stored"
    )
    export_parser.add_argument(
        "--hf-path", required=True, help="Directory path where the HuggingFace adapter will be saved"
    )
    export_parser.add_argument("--no-progress", action="store_true", help="Disable progress bar during export")
    export_parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code execution")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "import":
            import_hf_adapters_to_megatron(
                adapter_id=args.adapter_id,
                megatron_path=args.megatron_path,
                base_model=args.base_model,
                trust_remote_code=args.trust_remote_code,
            )

        elif args.command == "export":
            export_megatron_to_hf_adapters(
                adapter_id=args.adapter_id,
                megatron_path=args.megatron_path,
                hf_path=args.hf_path,
                show_progress=not args.no_progress,
                trust_remote_code=args.trust_remote_code,
            )
        else:
            raise RuntimeError(f"Unknown command: {args.command}")

    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        return 1

    finally:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
