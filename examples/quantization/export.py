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
    torchrun --nproc_per_node 2 examples/quantization/quantize.py \
        --hf-model-id meta-llama/Llama-3.2-1B --megatron-save-path ./quantized_megatron_checkpoint --tp 2

The process is as follows:
1. An AutoBridge is initialized from a pretrained Hugging Face model
    to get the tokenizer and model structure.
2. The quantized Megatron-LM model is loaded from the checkpoint using the specified path.
3. The model is exported to HuggingFace format using ModelOpt export utilities.

Usage:
torchrun --nproc_per_node 2 examples/quantization/export.py --hf-model-id meta-llama/Llama-3.2-1B \
    --megatron-load-path ./quantized_megatron_checkpoint --export-dir ./hf_export --tp 2
"""

import argparse
import warnings

import modelopt.torch.export as mtex
import modelopt.torch.utils.distributed as dist
import torch
from megatron.core.utils import unwrap_model
from quantize_utils import (
    add_common_model_args,
    build_bridge_and_provider,
    console,
    load_quantized_megatron_model,
    print_parallelism_summary,
    require_checkpoint,
    require_torchrun,
)

from megatron.bridge.models.decorators import torchrun_main
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo


warnings.filterwarnings("ignore")


@torchrun_main
def main(
    hf_model_id: str,
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
    require_torchrun()
    require_checkpoint(
        megatron_load_path,
        quantize_command=(
            f"torchrun --nproc_per_node {tp} examples/quantization/quantize.py "
            f"--hf-model-id {hf_model_id} --megatron-save-path {megatron_load_path} --tp {tp}"
        ),
    )

    # Convert dtype string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    # Initialize bridge from HF model to get tokenizer and model structure
    bridge, model_provider = build_bridge_and_provider(
        hf_model_id,
        tp=tp,
        pp=pp,
        ep=ep,
        etp=etp,
        load_weights=False,
        pipeline_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    megatron_model = load_quantized_megatron_model(bridge, megatron_load_path, tp=tp, pp=pp, ep=ep, etp=etp)

    # Now we can check for rank
    is_rank_0 = dist.is_master()

    print_parallelism_summary(model_provider)
    if is_rank_0:
        console.print(f"[green]Loaded quantized model from: {megatron_load_path}[/green]")

    # Get the unwrapped model for export
    unwrapped_model = unwrap_model(megatron_model)[0]

    # Only the last pp stage may have extra_modules (e.g. EAGLE, Medusa), hence broadcast from the last rank.
    has_extra_modules = hasattr(unwrapped_model, "eagle_module") or hasattr(unwrapped_model, "medusa_heads")
    has_extra_modules = dist.broadcast(has_extra_modules, src=dist.size() - 1)
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
        moe_router_dtype=getattr(unwrapped_model.config, "moe_router_dtype", None),
        trust_remote_code=is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model_id),
    )

    if is_rank_0:
        console.print("[green]Export completed successfully![/green]")
        console.print(f"[green]Model exported to: {export_dir}[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a quantized Megatron-LM checkpoint to HuggingFace format on multiple GPUs"
    )
    add_common_model_args(parser)

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

    dist.cleanup()
