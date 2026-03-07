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
Megatron-HuggingFace Checkpoint Conversion Example

This script demonstrates how to convert models between HuggingFace and Megatron formats
using the AutoBridge import_ckpt and export_ckpt methods.

Usage examples:
  # Download the HF checkpoint locally
  huggingface-cli download black-forest-labs/FLUX.1-dev \
  --local-dir /root/.cache/huggingface/flux.1-dev \
  --local-dir-use-symlinks False

  # Import a HuggingFace model to Megatron format
  python examples/diffusion/recipes/flux/conversion/convert_checkpoints.py import \
  --hf-model /root/.cache/huggingface/flux.1-dev\
  --megatron-path /workspace/checkpoints/megatron_checkpoints/flux.1-dev

  # Export a Megatron checkpoint to HuggingFace format
  python examples/diffusion/recipes/flux/conversion/convert_checkpoints.py export \
  --hf-model /root/.cache/huggingface/flux.1-dev \
  --megatron-path /workspace/checkpoints/megatron_checkpoints/flux.1-dev/iter_0000000 \
  --hf-path /workspace/checkpoints/hf_checkpoints/flux.1-dev_hf

  NOTE: The converted checkpoint /workspace/checkpoints/hf_checkpoints/flux.1-dev_hf
  only contains the DiT model transformer weights. You still need other components in
  the diffusion pipeline (VAE, text encoders, etc) to run inference. To do so, you can
  duplicate the original HF checkpoint directory /root/.cache/huggingface/flux.1-dev (which
  contains VAE, text encoders, etc.), and replace ./transformer with
  /workspace/checkpoints/hf_checkpoints/flux.1-dev_hf/transformer.

"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Optional

import torch

from megatron.bridge.diffusion.conversion.flux.flux_bridge import FluxBridge
from megatron.bridge.diffusion.conversion.flux.flux_hf_pretrained import PreTrainedFlux
from megatron.bridge.training.model_load_save import (
    load_megatron_model,
    save_megatron_model,
    temporary_distributed_context,
)


def validate_path(path: str, must_exist: bool = False) -> Path:
    """Validate and convert string path to Path object."""
    path_obj = Path(path)
    if must_exist and not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")
    return path_obj


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def import_hf_to_megatron(
    hf_model: str,
    megatron_path: str,
    torch_dtype: Optional[str] = None,
    device_map: Optional[str] = None,
    trust_remote_code: bool = False,
) -> None:
    """
    Import a HuggingFace model and save it as a Megatron checkpoint.

    Args:
        hf_model: HuggingFace model ID or path to model directory
        megatron_path: Directory path where the Megatron checkpoint will be saved
        torch_dtype: Model precision ("float32", "float16", "bfloat16")
        device_map: Device placement strategy ("auto", "cuda:0", etc.)
        trust_remote_code: Allow custom model code execution
    """
    print(f"🔄 Starting import: {hf_model} -> {megatron_path}")

    # Prepare kwargs
    kwargs = {}
    if torch_dtype:
        kwargs["torch_dtype"] = get_torch_dtype(torch_dtype)
        print(f"   Using torch_dtype: {torch_dtype}")

    if device_map:
        kwargs["device_map"] = device_map
        print(f"   Using device_map: {device_map}")

    if trust_remote_code:
        kwargs["trust_remote_code"] = trust_remote_code
        print(f"   Trust remote code: {trust_remote_code}")

    # Import using the convenience method
    print(f"📥 Loading HuggingFace model from diffusers: {hf_model}")
    # Minimal single-rank env to satisfy provider init if needed
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + random.randint(0, 1000)))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    hf = PreTrainedFlux(hf_model)
    bridge = FluxBridge()
    provider = bridge.provider_bridge(hf)
    provider.perform_initialization = False
    megatron_models = provider.provide_distributed_model(wrap_with_ddp=False, use_cpu_initialization=True)
    bridge.load_weights_hf_to_megatron(hf, megatron_models)

    # Save all parameters in transformer with their norms to a file
    param_norms_file = "megatron_bridge_param_norms.txt"
    with open(param_norms_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Transformer Parameters and Norms\n")
        f.write("=" * 80 + "\n")
        for name, param in megatron_models[0].named_parameters():
            if param.requires_grad:
                norm = param.data.norm().item()
                f.write(f"{name:80s} | shape: {str(list(param.shape)):20s} | norm: {norm:.6f}\n")
        f.write("=" * 80 + "\n")
    print(f"Parameter norms saved to: {param_norms_file}")

    save_megatron_model(megatron_models, megatron_path, hf_tokenizer_path=None)

    print(f"✅ Successfully imported model to: {megatron_path}")

    # Verify the checkpoint was created
    checkpoint_path = Path(megatron_path)
    if checkpoint_path.exists():
        print("📁 Checkpoint structure:")
        for item in checkpoint_path.iterdir():
            if item.is_dir():
                print(f"   📂 {item.name}/")
            else:
                print(f"   📄 {item.name}")


def export_megatron_to_hf(
    hf_model: str,
    megatron_path: str,
    hf_path: str,
    show_progress: bool = True,
    strict: bool = True,
) -> None:
    """
    Export a Megatron checkpoint to HuggingFace format.

    Args:
        megatron_path: Directory path where the Megatron checkpoint is stored
        hf_path: Directory path where the HuggingFace model will be saved
        show_progress: Display progress bar during weight export
    """
    print(f"🔄 Starting export: {megatron_path} -> {hf_path}")

    # Validate megatron checkpoint exists
    checkpoint_path = validate_path(megatron_path, must_exist=True)
    print(f"📂 Found Megatron checkpoint: {checkpoint_path}")

    # Look for configuration files to determine the model type
    config_files = list(checkpoint_path.glob("**/run_config.yaml"))
    if not config_files:
        # Look in iter_ subdirectories
        iter_dirs = [d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.startswith("iter_")]
        if iter_dirs:
            # Use the latest iteration
            latest_iter = max(iter_dirs, key=lambda d: int(d.name.replace("iter_", "")))
            config_files = list(latest_iter.glob("run_config.yaml"))

    if not config_files:
        raise FileNotFoundError(
            f"Could not find run_config.yaml in {checkpoint_path}. Please ensure this is a valid Megatron checkpoint."
        )

    print(f"📋 Found configuration: {config_files[0]}")

    print("ℹ️ Starting Flux Diffusers export...")
    # Minimal single-process distributed context on CPU for loading Megatron ckpt
    with temporary_distributed_context(backend="gloo"):
        # Resolve latest iter_* directory (use the config file we found)
        checkpoint_iter_dir = config_files[0].parent
        # 1) Load Megatron model from checkpoint
        megatron_models = load_megatron_model(str(checkpoint_iter_dir), use_cpu_init=True, skip_temp_dist_context=True)
        if not isinstance(megatron_models, list):
            megatron_models = [megatron_models]

        # 2) Prepare HF Flux wrapper for state/metadata and save artifacts
        hf = PreTrainedFlux(hf_model)
        Path(hf_path).mkdir(parents=True, exist_ok=True)
        # Some diffusers configs are FrozenDict and don't support save_pretrained; skip quietly
        try:
            hf.save_artifacts(hf_path)
        except Exception:
            pass

        # 3) Stream-export weights Megatron -> HF safetensors via Flux bridge
        bridge = FluxBridge()
        generator = bridge.stream_weights_megatron_to_hf(megatron_models, hf, cpu=True, show_progress=show_progress)
        # 4) Save streamed weights into hf_path
        hf.state.source.save_generator(generator, hf_path)

    print(f"✅ Successfully exported model to: {hf_path}")

    # Verify the export was created
    export_path = Path(hf_path)
    if export_path.exists():
        print("📁 Export structure:")
        for item in export_path.iterdir():
            if item.is_dir():
                print(f"   📂 {item.name}/")
            else:
                print(f"   📄 {item.name}")

    print("🔍 You can now load this model with:")
    print("   from transformers import AutoModelForCausalLM")
    print(f"   model = AutoModelForCausalLM.from_pretrained('{hf_path}')")


def main():
    """Main function to handle command line arguments and execute conversions."""
    parser = argparse.ArgumentParser(
        description="Convert models between HuggingFace and Megatron formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Conversion direction")

    # Import subcommand (HF -> Megatron)
    import_parser = subparsers.add_parser("import", help="Import HuggingFace model to Megatron checkpoint format")
    import_parser.add_argument("--hf-model", required=True, help="HuggingFace model ID or path to model directory")
    import_parser.add_argument(
        "--megatron-path", required=True, help="Directory path where the Megatron checkpoint will be saved"
    )
    import_parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], help="Model precision")
    import_parser.add_argument("--device-map", help='Device placement strategy (e.g., "auto", "cuda:0")')
    import_parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code execution")

    # Export subcommand (Megatron -> HF)
    export_parser = subparsers.add_parser("export", help="Export Megatron checkpoint to HuggingFace format")
    export_parser.add_argument("--hf-model", required=True, help="HuggingFace model ID or path to model directory")
    export_parser.add_argument(
        "--megatron-path", required=True, help="Directory path where the Megatron checkpoint is stored"
    )
    export_parser.add_argument(
        "--hf-path", required=True, help="Directory path where the HuggingFace model will be saved"
    )
    export_parser.add_argument("--no-progress", action="store_true", help="Disable progress bar during export")
    export_parser.add_argument(
        "--not-strict", action="store_true", help="Allow source and target checkpoint to have different keys"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "import":
        import_hf_to_megatron(
            hf_model=args.hf_model,
            megatron_path=args.megatron_path,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )

    elif args.command == "export":
        export_megatron_to_hf(
            hf_model=args.hf_model,
            megatron_path=args.megatron_path,
            hf_path=args.hf_path,
            show_progress=not args.no_progress,
            strict=not args.not_strict,
        )
    else:
        raise RuntimeError(f"Unknown command: {args.command}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
