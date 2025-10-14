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
This example demonstrates how to use the AutoBridge to perform a round-trip
conversion between a Hugging Face model and a Megatron-LM model on multiple GPUs.

The process is as follows:
1. An AutoBridge is initialized from a pretrained Hugging Face model
    (e.g., "meta-llama/Llama-3.2-1B"). This downloads the model from the Hub and loads it.
2. The bridge's `to_megatron_provider` method is called to get a Megatron-LM compatible model provider.
3. The model provider is configured for multi-GPU execution.
4. The model provider is used to instantiate the Megatron-LM model.
5. The weights of the converted Megatron-LM model are verified against the original
    Hugging Face model.
6. The `save_hf_pretrained` method is used to save the Megatron-LM
    model back into the Hugging Face format. A new directory, named after the
    model, will be created for the converted model files. By default, this
    directory is created in the current working directory, but a different
    parent directory can be specified via the `--output-dir` argument.
7. Optionally, the `save_megatron_model` method can be used to save the model
    in Megatron's native checkpoint format by specifying the `--megatron-save-path` argument.

Usage:
torchrun --nproc_per_node 1 examples/conversion/hf_megatron_roundtrip_multi_gpu.py
torchrun --nproc_per_node 1 examples/conversion/hf_megatron_roundtrip_multi_gpu.py --megatron-save-path ./megatron_checkpoint
"""

import argparse
import os
import sys

import torch
from rich.console import Console
from rich.table import Table

from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main


HF_MODEL_ID = "meta-llama/Llama-3.2-1B"
console = Console()


@torchrun_main
def main(
    hf_model_id: str = HF_MODEL_ID,
    output_dir: str = None,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
    megatron_save_path: str | None = None,
    megatron_load_path: str | None = None,
) -> None:
    """Perform round-trip conversion between HuggingFace and Megatron-LM models on multiple GPUs."""
    if os.environ.get("WORLD_SIZE") is None:
        console.print("This script must be launched with torchrun. Please run:")
        console.print(f"torchrun --nproc_per_node <gpus> {sys.argv[0]}")
        sys.exit(1)

    model_name = hf_model_id.split("/")[-1]
    if output_dir:
        save_path = os.path.join(output_dir, model_name)
    else:
        save_path = model_name

    bridge = AutoBridge.from_hf_pretrained(hf_model_id, trust_remote_code=True)

    if megatron_load_path:
        model_provider = bridge.to_megatron_provider(load_weights=False)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.pipeline_dtype = torch.bfloat16
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp

        # Once all overrides are set, finalize the model provider to ensure the post initialization logic is run
        model_provider.finalize()
        model_provider.initialize_model_parallel(seed=0)
        megatron_model = bridge.load_megatron_model(
            megatron_load_path,
            mp_overrides={
                "tensor_model_parallel_size": tp,
                "pipeline_model_parallel_size": pp,
                "expert_model_parallel_size": ep,
                "expert_tensor_parallel_size": etp,
                "pipeline_dtype": torch.bfloat16,
            },
            wrap_with_ddp=False,
        )
        megatron_model = [m.cuda() for m in megatron_model]

    else:
        model_provider = bridge.to_megatron_provider(load_weights=True)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.pipeline_dtype = torch.bfloat16
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp

        # Once all overrides are set, finalize the model provider to ensure the post initialization logic is run
        model_provider.finalize()
        model_provider.initialize_model_parallel(seed=0)
        megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    # Now we can check for rank
    is_rank_0 = torch.distributed.get_rank() == 0

    # Formatting
    if is_rank_0:
        table = Table(title="Hugging Face Weights Verification")
        table.add_column("Weight Name", style="cyan")
        table.add_column("Shape")
        table.add_column("DType")
        table.add_column("Device")
        table.add_column("Matches Original", justify="center")

    if is_rank_0:
        console.print(f"[yellow]Tensor parallel size: {model_provider.tensor_model_parallel_size}[/yellow]")
        console.print(f"[yellow]Pipeline parallel size: {model_provider.pipeline_model_parallel_size}[/yellow]")
        console.print(f"[yellow]Expert parallel size: {model_provider.expert_model_parallel_size}[/yellow]")
        console.print(f"[yellow]Expert tensor parallel size: {model_provider.expert_tensor_parallel_size}[/yellow]")

    for name, param in bridge.export_hf_weights(megatron_model, show_progress=False):
        if is_rank_0:
            original_param = bridge.hf_pretrained.state[name]
            match = torch.allclose(
                param, original_param.to(param.device), atol=1e-1
            )  # Increased tolerance for bfloat16
            table.add_row(
                name,
                str(tuple(param.shape)),
                str(param.dtype).replace("torch.", ""),
                str(param.device),
                "✅" if match else "❌",
            )

    if is_rank_0:
        console.print(table)
        console.print(f"Saving HF-ckpt in {save_path}...")

    bridge.save_hf_pretrained(megatron_model, save_path)

    # Save in Megatron format if path is provided
    if megatron_save_path:
        if is_rank_0:
            console.print(f"Saving Megatron checkpoint in {megatron_save_path}...")
        bridge.save_megatron_model(megatron_model, megatron_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert between HuggingFace and Megatron-LM model formats on multiple GPUs"
    )
    parser.add_argument("--hf-model-id", type=str, default=HF_MODEL_ID, help="HuggingFace model ID to convert")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="The directory where the converted model directory will be created. Defaults to the current working directory.",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")

    parser.add_argument(
        "--megatron-save-path",
        type=str,
        default=None,
        help="Path to save the model in Megatron checkpoint format. If not provided, model will not be saved in Megatron format.",
    )
    parser.add_argument(
        "--megatron-load-path",
        type=str,
        default=None,
        help="Path to load the model in Megatron checkpoint format. If provided, model will not start from HF checkpoint.",
    )
    args = parser.parse_args()
    main(
        args.hf_model_id,
        args.output_dir,
        args.tp,
        args.pp,
        args.ep,
        args.etp,
        args.megatron_save_path,
        args.megatron_load_path,
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
