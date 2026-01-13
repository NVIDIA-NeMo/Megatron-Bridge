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
This example demonstrates how to load a quantized Megatron-LM VLM checkpoint
and perform image+text generation using the AutoBridge on multiple GPUs.

Prerequisites:
First, you must run the quantization process to create a quantized checkpoint:
    torchrun --nproc_per_node 8 examples/quantization/quantize_vlm.py \
        --hf-model-id Qwen/Qwen3-VL-8B-Instruct \
        --export-quant-cfg fp8 \
        --megatron-save-path ./qwen3_vl_quantized \
        --tp 8

The process is as follows:
1. An AutoBridge is initialized from a pretrained Hugging Face VLM model
    to get the processor and model structure.
2. The quantized Megatron-LM model is loaded from the checkpoint using the specified path.
3. Image+text generation is performed using the loaded quantized model.

Usage:
torchrun --nproc_per_node 8 examples/quantization/ptq_generate_vlm.py \
    --hf-model-id Qwen/Qwen3-VL-8B-Instruct \
    --megatron-load-path ./qwen3_vl_quantized \
    --tp 8 \
    --image-path /path/to/image.jpg \
    --prompts "Describe this image."
"""

import argparse
import os
import sys
import warnings

import torch
from megatron.core.utils import unwrap_model
from quantize_utils import console
from quantize_vlm import _custom_prompt_forward_loop_func
from transformers import AutoProcessor

from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo


warnings.filterwarnings("ignore")

HF_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_IMAGE_PATH = "/models/demo.jpeg"


def _validate_quantized_model(model: torch.nn.Module, is_rank_0: bool) -> None:
    """Validate that the model contains quantized layers.

    This is a functional test to ensure quantized checkpoints are loaded correctly.
    If someone accidentally breaks the quantization loading logic (e.g., in
    has_modelopt_state or build_and_load_model), this check will catch it.

    We check for QuantRowParallelLinear and QuantColumnParallelLinear as these
    are present in all quantized model architectures (GPT, Llama, Qwen, Nemotron-H, etc).

    Args:
        model: The unwrapped model to validate
        is_rank_0: Whether this is rank 0 (for printing)

    Raises:
        RuntimeError: If the model doesn't contain expected quantized layers
    """
    # Check for quantized layer types that are universal across all architectures
    model_str = str(model)

    required_quant_layers = [
        "QuantRowParallelLinear",
        "QuantColumnParallelLinear",
    ]

    missing_layers = [layer for layer in required_quant_layers if layer not in model_str]

    if missing_layers:
        error_msg = (
            f"\n{'=' * 80}\n"
            f"QUANTIZATION VALIDATION FAILED!\n"
            f"{'=' * 80}\n"
            f"Expected quantized layers not found in the loaded model.\n"
            f"This indicates the quantized checkpoint was not loaded correctly.\n\n"
            f"Missing: {missing_layers}\n"
            f"Expected: {required_quant_layers}\n\n"
            f"This is likely due to a bug in the checkpoint loading logic.\n"
            f"{'=' * 80}\n"
        )
        if is_rank_0:
            console.print(f"[red]{error_msg}[/red]")
        raise RuntimeError(error_msg)

    if is_rank_0:
        console.print(
            "[green]✓ Quantization validation passed: Found QuantRowParallelLinear and QuantColumnParallelLinear[/green]"
        )


@torchrun_main
def main(
    hf_model_id: str = HF_MODEL_ID,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
    megatron_load_path: str = "./quantized_megatron_checkpoint",
    prompts: str = "Describe this image.",
    osl: int = 32,
    image_path: str = DEFAULT_IMAGE_PATH,
    trust_remote_code: bool = True,
) -> None:
    """Load a quantized Megatron-LM VLM checkpoint and perform image+text generation on multiple GPUs."""
    if os.environ.get("WORLD_SIZE") is None:
        console.print("This script must be launched with torchrun. Please run:")
        console.print(f"torchrun --nproc_per_node <gpus> {sys.argv[0]}")
        sys.exit(1)

    # Check if the checkpoint path exists
    if not os.path.exists(megatron_load_path):
        console.print(f"[red]Error: Quantized checkpoint path {megatron_load_path} does not exist![/red]")
        console.print("[yellow]Please run the quantization process first:[/yellow]")
        console.print(
            f"[yellow]torchrun --nproc_per_node {tp} examples/quantization/quantize_vlm.py "
            f"--hf-model-id {hf_model_id} --megatron-save-path {megatron_load_path} --tp {tp}[/yellow]"
        )
        sys.exit(1)

    # Check if the image path exists
    if not os.path.exists(image_path):
        console.print(f"[red]Error: Image path {image_path} does not exist![/red]")
        sys.exit(1)

    # Initialize bridge from HF model to get processor and model structure
    bridge = AutoBridge.from_hf_pretrained(
        hf_model_id,
        trust_remote_code=is_safe_repo(
            trust_remote_code=trust_remote_code,
            hf_path=hf_model_id,
        ),
    )

    # Load processor for VLM
    processor = AutoProcessor.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code)

    # Get model provider and configure for multi-GPU execution
    model_provider = bridge.to_megatron_provider(load_weights=False)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp
    model_provider.pipeline_dtype = torch.bfloat16

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
        },
        wrap_with_ddp=False,
    )
    megatron_model = [m.cuda() for m in megatron_model]

    # Now we can check for rank
    is_rank_0 = torch.distributed.get_rank() == 0

    if is_rank_0:
        console.print(f"[green]Tensor parallel size: {model_provider.tensor_model_parallel_size}[/green]")
        console.print(f"[green]Pipeline parallel size: {model_provider.pipeline_model_parallel_size}[/green]")
        console.print(f"[green]Expert parallel size: {model_provider.expert_model_parallel_size}[/green]")
        console.print(f"[green]Expert tensor parallel size: {model_provider.expert_tensor_parallel_size}[/green]")
        console.print(f"[green]Loaded quantized model from: {megatron_load_path}[/green]")

    # Get the unwrapped model for generation
    unwrapped_model = unwrap_model(megatron_model)[0]
    unwrapped_model.eval()

    # Validate that the model has quantized layers
    # _validate_quantized_model(unwrapped_model, is_rank_0)

    # Test quantized model with custom prompts
    if is_rank_0:
        console.print(f"[green]Loaded Quantized Model:\n {unwrapped_model}[/green]")
        console.print("[green]Testing quantized VLM model with image and prompt...[/green]")

    _custom_prompt_forward_loop_func(unwrapped_model, processor, is_rank_0, prompts, osl, test_image_path=image_path)

    if is_rank_0:
        console.print("[green]Generation completed successfully![/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a quantized Megatron-LM VLM checkpoint and perform image+text generation on multiple GPUs"
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default=HF_MODEL_ID,
        help="HuggingFace model ID for processor and model structure (e.g., Qwen/Qwen3-VL-8B-Instruct)",
    )

    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")
    parser.add_argument(
        "--megatron-load-path",
        type=str,
        default="./quantized_megatron_checkpoint",
        help="Path to the quantized Megatron checkpoint to load (must be created first using quantize_vlm.py)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="Describe this image.",
        help="Text prompt for testing quantized VLM model.",
    )
    parser.add_argument(
        "--osl",
        type=int,
        default=32,
        help="Output sequence length for generation.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help="Path to the image file for VLM generation.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=True, help="if trust_remote_code")

    args = parser.parse_args()
    main(
        args.hf_model_id,
        args.tp,
        args.pp,
        args.ep,
        args.etp,
        args.megatron_load_path,
        args.prompts,
        args.osl,
        args.image_path,
        args.trust_remote_code,
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

