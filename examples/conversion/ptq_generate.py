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

The process is as follows:
1. An AutoBridge is initialized from a pretrained Hugging Face model
    to get the tokenizer and model structure.
2. The quantized Megatron-LM model is loaded from the checkpoint using the specified path.
3. Text generation is performed using the loaded quantized model.

Usage:
torchrun --nproc_per_node 2 examples/models/ptq_generate.py --megatron-load-path ./megatron_checkpoint --prompts "Hello!|Born in California, Soyer trained as a"
"""

import argparse
import os
import sys
import warnings

import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main
from megatron.core.transformer.moe.router import TopKRouter
from modelopt.torch.utils.plugins.megatron_generate import megatron_generate
from megatron.training.utils import unwrap_model
from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec

warnings.filterwarnings('ignore')

HF_MODEL_ID = "/models/Llama-3.2-1B"
console = Console()


def _custom_prompt_forward_loop_func(model, prompts, tokenizer, is_rank_0, osl=32):
    """Forward loop function for testing quantized model with custom prompts."""
    all_prompts = prompts.split("|")
    
    for idx, prompt in tqdm(enumerate(all_prompts), disable=torch.distributed.get_rank()):
        tokens = tokenizer(prompt, return_tensors="pt")
        generated_ids = megatron_generate(model, tokens.input_ids.cuda(0), osl=osl, enable_kv_cache=False)
        generated_texts = tokenizer.batch_decode(generated_ids)
        if is_rank_0:
            console.print(f"[green]Prompt {idx + 1}: {prompt}[/green]")
            console.print(f"[green]Generated: {generated_texts}[/green]")


@torchrun_main
def main(
    hf_model_id: str = HF_MODEL_ID,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
    megatron_load_path: str = "./megatron_checkpoint",
    prompts: str = "Hello!|Born in California, Soyer trained as a",
    osl: int = 32,
) -> None:
    """Load a quantized Megatron-LM checkpoint and perform text generation on multiple GPUs."""
    if os.environ.get("WORLD_SIZE") is None:
        console.print("This script must be launched with torchrun. Please run:")
        console.print(f"torchrun --nproc_per_node <gpus> {sys.argv[0]}")
        sys.exit(1)
    
    # Check if the checkpoint path exists
    if not os.path.exists(megatron_load_path):
        console.print(f"[red]Error: Checkpoint path {megatron_load_path} does not exist![/red]")
        sys.exit(1)
    
    # Initialize bridge from HF model to get tokenizer and model structure
    bridge = AutoBridge.from_hf_pretrained(hf_model_id)
    
    # Get model provider and configure for multi-GPU execution
    model_provider = bridge.to_megatron_provider(load_weights=False)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp
    model_provider.transformer_layer_spec = lambda config: get_gpt_modelopt_spec(
        config=config,
        local_core_attention=False,
        remap_te_layernorm=False,
        real_quant_cfg="None",
        use_arbitrary_attention_mask=True,
    )
    model_provider.initialize_model_parallel(seed=0)
    
    # Create the model using the provider (this ensures the correct transformer_layer_spec)
    megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)
    
    # Load weights from the checkpoint into the existing model
    from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint
    _load_model_weights_from_checkpoint(megatron_load_path, megatron_model)
    
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

    # Test quantized model with custom prompts
    if is_rank_0:
        console.print("[green]Testing quantized model with custom prompts...[/green]")
    
    _custom_prompt_forward_loop_func(unwrapped_model, prompts, bridge.tokenizer, is_rank_0, osl)

    if is_rank_0:
        console.print("[green]Generation completed successfully![/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a quantized Megatron-LM checkpoint and perform text generation on multiple GPUs"
    )
    parser.add_argument("--hf-model-id", type=str, default=HF_MODEL_ID, help="HuggingFace model ID for tokenizer and model structure")

    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")
    parser.add_argument(
        "--megatron-load-path",
        type=str,
        default="./megatron_checkpoint",
        help="Path to the quantized Megatron checkpoint to load",
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
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
