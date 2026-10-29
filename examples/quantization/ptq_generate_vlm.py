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
from quantize_vlm import _custom_prompt_forward_loop_func
from transformers import AutoProcessor

from megatron.bridge.models.decorators import torchrun_main


warnings.filterwarnings("ignore")

DEFAULT_IMAGE_PATH = "/models/demo.jpeg"


def _validate_quantized_model(model: torch.nn.Module, is_rank_0: bool) -> None:
    """Validate that the model contains quantized layers.

    This is a functional test to ensure quantized checkpoints are loaded correctly.
    If someone accidentally breaks the quantization loading logic (e.g., in
    has_modelopt_state or build_and_load_model), this check will catch it.

    For VLM models, we only check for TE spec quantized layers since all supported
    VLM models (Qwen3-VL) use TE spec.

    Args:
        model: The unwrapped model to validate
        is_rank_0: Whether this is rank 0 (for printing)

    Raises:
        RuntimeError: If the model doesn't contain expected quantized layers
    """
    model_str = str(model)

    # TE spec quantized layers (VLM models always use TE spec)
    te_spec_layers = [
        "QuantTERowParallelLinear",
        "QuantTELayerNormColumnParallelLinear",
    ]

    # Check if model has TE spec quantized layers
    has_te_spec = all(layer in model_str for layer in te_spec_layers)

    if not has_te_spec:
        error_msg = (
            f"\n{'=' * 80}\n"
            f"QUANTIZATION VALIDATION FAILED!\n"
            f"{'=' * 80}\n"
            f"Expected quantized layers not found in the loaded model.\n"
            f"This indicates the quantized checkpoint was not loaded correctly.\n\n"
            f"Expected TE spec layers: {te_spec_layers}\n\n"
            f"This is likely due to a bug in the checkpoint loading logic.\n"
            f"{'=' * 80}\n"
        )
        if is_rank_0:
            console.print(f"[red]{error_msg}[/red]")
        raise RuntimeError(error_msg)

    if is_rank_0:
        console.print(
            "[green]✓ Quantization validation passed: Found TE spec quantized layers "
            "(QuantTERowParallelLinear, QuantTELayerNormColumnParallelLinear)[/green]"
        )


@torchrun_main
def main(
    hf_model_id: str,
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
    require_torchrun()
    require_checkpoint(
        megatron_load_path,
        quantize_command=(
            f"torchrun --nproc_per_node {tp} examples/quantization/quantize_vlm.py "
            f"--hf-model-id {hf_model_id} --megatron-save-path {megatron_load_path} --tp {tp}"
        ),
    )

    # Check if the image path exists (skip check for URLs)
    is_url = image_path.startswith("http://") or image_path.startswith("https://")
    if not is_url and not os.path.exists(image_path):
        console.print(f"[red]Error: Image path {image_path} does not exist![/red]")
        sys.exit(1)

    # Initialize bridge from HF model to get processor and model structure
    bridge, model_provider = build_bridge_and_provider(
        hf_model_id,
        tp=tp,
        pp=pp,
        ep=ep,
        etp=etp,
        load_weights=False,
        pipeline_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
    )

    # Load processor for VLM
    processor = AutoProcessor.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code)

    megatron_model = load_quantized_megatron_model(bridge, megatron_load_path, tp=tp, pp=pp, ep=ep, etp=etp)
    megatron_model = [m.cuda() for m in megatron_model]

    # Now we can check for rank
    is_rank_0 = dist.is_master()

    print_parallelism_summary(model_provider)
    if is_rank_0:
        console.print(f"[green]Loaded quantized model from: {megatron_load_path}[/green]")

    # Get the unwrapped model for generation
    unwrapped_model = unwrap_model(megatron_model)[0]

    # Validate that the model has quantized layers
    _validate_quantized_model(unwrapped_model, is_rank_0)

    # Test quantized model with custom prompts
    if is_rank_0:
        console.print(f"[green]Loaded Quantized Model:\n {unwrapped_model}[/green]")
        console.print("[green]Testing quantized VLM model with image and prompt...[/green]")

    # .eval() above disables dropout/etc but not autograd -- without no_grad(),
    # every decode step in the osl loop retains a full backward-pass graph it
    # never uses, which was enough to OOM a 27B TP=8 model at ~79GB/79.18GB
    # even after model loading and quantization validation had succeeded.
    with torch.no_grad():
        _custom_prompt_forward_loop_func(
            unwrapped_model, processor, is_rank_0, prompts, osl, test_image_path=image_path
        )

    if is_rank_0:
        console.print("[green]Generation completed successfully![/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a quantized Megatron-LM VLM checkpoint and perform image+text generation on multiple GPUs"
    )
    add_common_model_args(parser)

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

    args = parser.parse_args()
    try:
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
    finally:
        dist.cleanup()
