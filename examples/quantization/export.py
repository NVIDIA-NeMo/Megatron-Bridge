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
This example demonstrates how to export a Megatron-LM quantized checkpoint
to HuggingFace format using the AutoBridge on multiple GPUs.

Prerequisites:
First, you must run the quantization process to create a quantized checkpoint:
    torchrun --nproc_per_node 2 examples/quantization/quantize.py --megatron-save-path ./quantized_megatron_checkpoint --tp 2

The process is as follows:
1. An AutoBridge is initialized from a pretrained Hugging Face model
    to get the tokenizer and model structure.
2. The quantized Megatron-LM model is loaded from the checkpoint using the specified path.
3. The model is exported to HuggingFace format using ModelOpt export utilities.

Usage:
torchrun --nproc_per_node 2 examples/quantization/export.py --megatron-load-path ./quantized_megatron_checkpoint --export-dir ./hf_export --tp 2
"""

import argparse
import os
import sys
import warnings

import modelopt.torch.export as mtex
import torch
from megatron.core.utils import unwrap_model
from rich.console import Console

from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo


warnings.filterwarnings("ignore")

HF_MODEL_ID = "meta-llama/Llama-3.2-1B"
console = Console()


@torchrun_main
def main(
    hf_model_id: str = HF_MODEL_ID,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
    megatron_load_path: str = "./quantized_megatron_checkpoint",
    export_dir: str = "./hf_export",
    export_extra_modules: bool = False,
    dtype: str = "bfloat16",
    trust_remote_code: bool | None = None,
) -> None:
    """Export a quantized Megatron-LM checkpoint to HuggingFace format on multiple GPUs."""
    if os.environ.get("WORLD_SIZE") is None:
        console.print("This script must be launched with torchrun. Please run:")
        console.print(f"torchrun --nproc_per_node <gpus> {sys.argv[0]}")
        sys.exit(1)

    # Check if the checkpoint path exists
    if not os.path.exists(megatron_load_path):
        console.print(f"[red]Error: Quantized checkpoint path {megatron_load_path} does not exist![/red]")
        console.print("[yellow]Please run the quantization process first:[/yellow]")
        console.print(
            f"[yellow]torchrun --nproc_per_node {tp} examples/quantization/quantize.py --megatron-save-path {megatron_load_path} --tp {tp}[/yellow]"
        )
        sys.exit(1)

    # Initialize bridge from HF model to get tokenizer and model structure
    bridge = AutoBridge.from_hf_pretrained(
        hf_model_id,
        trust_remote_code=is_safe_repo(
            trust_remote_code=trust_remote_code,
            hf_path=hf_model_id,
        ),
    )

    # Get model provider and configure for multi-GPU execution
    model_provider = bridge.to_megatron_provider(load_weights=False)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp

    # Convert dtype string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    model_provider.pipeline_dtype = torch_dtype

    # Once all overrides are set, finalize the model provider to ensure the post initialization logic is run
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)

    # Load the quantized model
    megatron_model = bridge.load_megatron_model(
        megatron_load_path,
        mp_overrides={
            "tensor_model_parallel_size": tp,
            "pipeline_model_parallel_size": pp,
            "expert_model_parallel_size": ep,
            "expert_tensor_parallel_size": etp,
        },
        wrap_with_ddp=False,
    )

    # Now we can check for rank
    is_rank_0 = torch.distributed.get_rank() == 0

    if is_rank_0:
        console.print(f"[green]Tensor parallel size: {model_provider.tensor_model_parallel_size}[/green]")
        console.print(f"[green]Pipeline parallel size: {model_provider.pipeline_model_parallel_size}[/green]")
        console.print(f"[green]Expert parallel size: {model_provider.expert_model_parallel_size}[/green]")
        console.print(f"[green]Expert tensor parallel size: {model_provider.expert_tensor_parallel_size}[/green]")
        console.print(f"[green]Loaded quantized model from: {megatron_load_path}[/green]")

    # Get the unwrapped model for export
    unwrapped_model = unwrap_model(megatron_model)[0]

    # Decide whether we are exporting only the extra_modules (e.g. EAGLE, Medusa).
    # Only the last pp stage may have extra_modules, hence broadcast from the last rank.
    has_extra_modules = hasattr(unwrapped_model, "eagle_module") or hasattr(unwrapped_model, "medusa_heads")

    # Broadcast from the last rank in case this is pipeline parallel
    if torch.distributed.is_initialized():
        extra_modules_list = [has_extra_modules]
        torch.distributed.broadcast_object_list(
            extra_modules_list,
            src=torch.distributed.get_world_size() - 1,
        )
        export_extra_modules_flag = extra_modules_list[0] if export_extra_modules else False
    else:
        export_extra_modules_flag = has_extra_modules if export_extra_modules else False

    if is_rank_0:
        console.print("[green]Exporting to HuggingFace format...[/green]")
        console.print(f"[green]Export directory: {export_dir}[/green]")
        console.print(f"[green]Export extra modules: {export_extra_modules_flag}[/green]")

    # Export the model to HuggingFace format
    mtex.export_mcore_gpt_to_hf(
        unwrapped_model,
        hf_model_id,
        export_extra_modules=export_extra_modules_flag,
        dtype=torch_dtype,
        export_dir=export_dir,
        trust_remote_code=is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model_id),
    )

    if is_rank_0:
        console.print("[green]Export completed successfully![/green]")
        console.print(f"[green]Model exported to: {export_dir}[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a quantized Megatron-LM checkpoint to HuggingFace format on multiple GPUs"
    )
    parser.add_argument(
        "--hf-model-id", type=str, default=HF_MODEL_ID, help="HuggingFace model ID for tokenizer and model structure"
    )

    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")
    parser.add_argument(
        "--megatron-load-path",
        type=str,
        default="./quantized_megatron_checkpoint",
        help="Path to the quantized Megatron checkpoint to load (must be created first using quantize.py)",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="./hf_export",
        help="Directory to export the HuggingFace model to",
    )
    parser.add_argument(
        "--export-extra-modules",
        action="store_true",
        help="Export extra modules such as Medusa, EAGLE, or MTP",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for export",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="if trust_remote_code",
    )

    args = parser.parse_args()
    main(
        args.hf_model_id,
        args.tp,
        args.pp,
        args.ep,
        args.etp,
        args.megatron_load_path,
        args.export_dir,
        args.export_extra_modules,
        args.dtype,
        args.trust_remote_code,
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
