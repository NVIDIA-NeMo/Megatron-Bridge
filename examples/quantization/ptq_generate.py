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
This example demonstrates how to load a quantized Megatron-LM checkpoint
and perform text generation using the AutoBridge on multiple GPUs.

Prerequisites:
First, you must run the quantization process to create a quantized checkpoint:
    torchrun --nproc_per_node 2 examples/quantization/quantize.py \
        --hf-model-id meta-llama/Llama-3.2-1B --megatron-save-path ./quantized_megatron_checkpoint --tp 2

The process is as follows:
1. An AutoBridge is initialized from a pretrained Hugging Face model
    to get the tokenizer and model structure.
2. The quantized Megatron-LM model is loaded from the checkpoint using the specified path.
3. Text generation is performed using the loaded quantized model.

Usage:
torchrun --nproc_per_node 2 examples/quantization/ptq_generate.py --hf-model-id meta-llama/Llama-3.2-1B \
    --megatron-load-path ./quantized_megatron_checkpoint --tp 2
"""

import argparse
import warnings

import modelopt.torch.utils.distributed as dist
import torch
from megatron.core.utils import unwrap_model
from quantize import _custom_prompt_forward_loop_func
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


warnings.filterwarnings("ignore")


def _validate_quantized_model(model: torch.nn.Module, is_rank_0: bool) -> None:
    """Validate that the model contains quantized layers.

    This is a functional test to ensure quantized checkpoints are loaded correctly.
    If someone accidentally breaks the quantization loading logic (e.g., in
    has_modelopt_state or build_and_load_model), this check will catch it.

    We check for quantized layer types that indicate successful quantization:
    - Local spec: QuantRowParallelLinear, QuantColumnParallelLinear
    - TE spec: QuantTERowParallelLinear, QuantTELayerNormColumnParallelLinear

    Args:
        model: The unwrapped model to validate
        is_rank_0: Whether this is rank 0 (for printing)

    Raises:
        RuntimeError: If the model doesn't contain expected quantized layers
    """
    model_str = str(model)

    # Local spec quantized layers
    local_spec_layers = [
        "QuantRowParallelLinear",
        "QuantColumnParallelLinear",
    ]

    # TE spec quantized layers
    te_spec_layers = [
        "QuantTERowParallelLinear",
        "QuantTELayerNormColumnParallelLinear",
    ]

    # Check if model has local spec quantized layers
    has_local_spec = all(layer in model_str for layer in local_spec_layers)

    # Check if model has TE spec quantized layers
    has_te_spec = all(layer in model_str for layer in te_spec_layers)

    if not has_local_spec and not has_te_spec:
        error_msg = (
            f"\n{'=' * 80}\n"
            f"QUANTIZATION VALIDATION FAILED!\n"
            f"{'=' * 80}\n"
            f"Expected quantized layers not found in the loaded model.\n"
            f"This indicates the quantized checkpoint was not loaded correctly.\n\n"
            f"Expected one of:\n"
            f"  - Local spec: {local_spec_layers}\n"
            f"  - TE spec: {te_spec_layers}\n\n"
            f"This is likely due to a bug in the checkpoint loading logic.\n"
            f"{'=' * 80}\n"
        )
        if is_rank_0:
            console.print(f"[red]{error_msg}[/red]")
        raise RuntimeError(error_msg)

    if is_rank_0:
        if has_te_spec:
            console.print(
                "[green]✓ Quantization validation passed: Found TE spec quantized layers "
                "(QuantTERowParallelLinear, QuantTELayerNormColumnParallelLinear)[/green]"
            )
        else:
            console.print(
                "[green]✓ Quantization validation passed: Found local spec quantized layers "
                "(QuantRowParallelLinear, QuantColumnParallelLinear)[/green]"
            )


@torchrun_main
def main(
    hf_model_id: str,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
    megatron_load_path: str = "./quantized_megatron_checkpoint",
    prompts: str = "Hello!|Born in California, Soyer trained as a",
    osl: int = 32,
    trust_remote_code: bool | None = None,
) -> None:
    """Load a quantized Megatron-LM checkpoint and perform text generation on multiple GPUs."""
    require_torchrun()
    require_checkpoint(
        megatron_load_path,
        quantize_command=(
            f"torchrun --nproc_per_node {tp} examples/quantization/quantize.py "
            f"--hf-model-id {hf_model_id} --megatron-save-path {megatron_load_path} --tp {tp}"
        ),
    )

    # Initialize bridge from HF model to get tokenizer and model structure
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
        console.print("[green]Testing quantized model with custom prompts...[/green]")

    _custom_prompt_forward_loop_func(unwrapped_model, prompts, bridge.hf_pretrained.tokenizer, is_rank_0, osl)

    if is_rank_0:
        console.print("[green]Generation completed successfully![/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a quantized Megatron-LM checkpoint and perform text generation on multiple GPUs"
    )
    add_common_model_args(parser)

    parser.add_argument(
        "--megatron-load-path",
        type=str,
        default="./quantized_megatron_checkpoint",
        help="Path to the quantized Megatron checkpoint to load (must be created first using quantize.py)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="Hello!|Born in California, Soyer trained as a",
        help="Input texts for testing quantized model. Please use | to separate different batches.",
    )
    parser.add_argument(
        "--osl",
        type=int,
        default=32,
        help="Output sequence length for generation.",
    )

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
        args.trust_remote_code,
    )

    dist.cleanup()
