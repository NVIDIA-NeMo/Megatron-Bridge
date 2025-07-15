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
This example demonstrates how to use the CausalLMBridge to perform a round-trip
conversion between a Hugging Face model and a Megatron-LM model on multiple GPUs.

The process is as follows:
1. A CausalLMBridge is initialized from a pretrained Hugging Face model
    (e.g., "meta-llama/Llama-3.2-1B"). This downloads the model from the Hub and loads it.
2. The bridge's `to_megatron_provider` method is called to get a Megatron-LM compatible model provider.
3. The model provider is configured for multi-GPU execution.
4. The model provider is used to instantiate the Megatron-LM model.
5. The weights of the converted Megatron-LM model are verified against the original
    Hugging Face model.
6. Finally, the `save_hf_pretrained` method is used to save the Megatron-LM
    model back into the Hugging Face format. A new directory, named after the
    model, will be created for the converted model files. By default, this
    directory is created in the current working directory, but a different
    parent directory can be specified via the `--output-dir` argument.

Usage:
torchrun --nproc_per_node 2 examples/multi_gpu_hf.py
"""

import argparse
import os
import sys

import torch
from rich.console import Console
from rich.table import Table

from megatron.bridge import CausalLMBridge
from megatron.bridge.common.decorators import torchrun_main


HF_MODEL_ID = "meta-llama/Llama-3.1-8B"
console = Console()


@torchrun_main
def main(hf_model_id: str = HF_MODEL_ID, output_dir: str = None) -> None:
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

    bridge = CausalLMBridge.from_hf_pretrained(hf_model_id)

    model_provider = bridge.to_megatron_provider()
    model_provider.tensor_model_parallel_size = int(os.environ.get("WORLD_SIZE", "1"))
    model_provider.initialize_model_parallel(seed=0)

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

    megatron_model = model_provider(wrap_with_ddp=False)

    # Debug: Print model info
    if is_rank_0:
        console.print(f"[yellow]Tensor parallel size: {model_provider.tensor_model_parallel_size}[/yellow]")
        console.print(f"[yellow]Pipeline parallel size: {model_provider.pipeline_model_parallel_size}[/yellow]")

    for name, param in bridge(megatron_model, show_progress=False, order="safetensors"):
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

    args = parser.parse_args()
    main(args.hf_model_id, args.output_dir)
