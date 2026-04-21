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
This example demonstrates how to use the AutoBridge to perform quantization
from a Hugging Face model to a quantized Megatron-LM model on multiple GPUs.

The process is as follows:
1. An AutoBridge is initialized from a pretrained Hugging Face model
    (e.g., "meta-llama/Llama-3.2-1B"). This downloads the model from the Hub and loads it.
2. ModelOpt quantization is applied to the Megatron-LM model using the specified configuration.
3. The quantized Megatron-LM model is saved in Megatron's native checkpoint format
    using the `--megatron-save-path` argument.

Usage:
torchrun --nproc_per_node 2 examples/models/quantize.py --export-quant-cfg fp8
torchrun --nproc_per_node 2 examples/models/quantize.py --export-quant-cfg fp8 --megatron-save-path ./megatron_checkpoint
"""

import argparse
import os
import sys
import warnings

import modelopt.torch.quantization as mtq
import torch
from datasets import load_dataset
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.utils import unwrap_model
from modelopt.torch.utils.plugins.megatron_generate import megatron_generate
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main
from megatron.bridge.models.gpt_provider import quantization_layer_spec


warnings.filterwarnings("ignore")

HF_MODEL_ID = "meta-llama/Llama-3.2-1B"
console = Console()


QUANT_CFG_CHOICES = {
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "fp8_blockwise": mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
}


def get_modelopt_torch_quantization_config(export_quant_cfg, export_kv_cache_quant=False, weight_only=False):
    """Return a quantization config based on the specified configuration."""
    mtq_config = QUANT_CFG_CHOICES[export_quant_cfg]

    fp8_config = {"enable": True, "num_bits": (4, 3), "axis": None}
    fp4_config = {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        "axis": None,
        "enable": True,
    }

    if "fp8" == export_quant_cfg:
        # Enable Medusa heads and kv-cache quantization
        mtq_config["quant_cfg"]["*medusa_heads**"] = fp8_config
    if "fp4" in export_quant_cfg:
        # Enable Medusa heads and kv-cache quantization
        mtq_config["quant_cfg"]["*medusa_heads**"] = fp4_config
    if "awq" in export_quant_cfg:
        weight_quantizer = mtq_config["quant_cfg"]["*weight_quantizer"]  # type: ignore
        if isinstance(weight_quantizer, list):
            weight_quantizer = weight_quantizer[0]
        weight_quantizer["block_sizes"][-1] = 128
    if export_kv_cache_quant:
        mtq_config["quant_cfg"]["*linear_qkv.output_quantizer"] = fp8_config
    if weight_only:
        mtq_config["quant_cfg"]["*input_quantizer"] = {"enable": False}

    return mtq_config


def get_calib_dataloader(calib_size=512, max_sequence_length=512):
    """Return a dataloader for calibration."""
    dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
    text_column = "article"

    calib_size = min(len(dataset), calib_size)
    for i in range(calib_size):
        yield dataset[i][text_column][:max_sequence_length]


def _hf_dataset_forward_loop_func(model, tokenizer, calib_size, force_all_expert_routing=False):
    """Forward loop function for calibration using HuggingFace dataset."""
    dataloader = get_calib_dataloader(calib_size)

    if force_all_expert_routing:
        for name, module in model.named_modules():
            if isinstance(module, TopKRouter):
                module.topk = module.num_experts

    for prompt in tqdm(dataloader, total=calib_size, disable=torch.distributed.get_rank()):
        tokens = tokenizer(prompt, return_tensors="pt")
        # Use megatron_generate for calibration (same as quantize.py)
        megatron_generate(model, tokens.input_ids.cuda(), osl=1)

        if force_all_expert_routing:
            for name, module in model.named_modules():
                if isinstance(module, TopKRouter):
                    module.topk = module.config.moe_router_topk


def _custom_prompt_forward_loop_func(model, prompts, tokenizer, is_rank_0, osl=32):
    """Forward loop function for testing quantized model with custom prompts."""
    all_prompts = prompts.split("|")

    for idx, prompt in tqdm(enumerate(all_prompts), disable=torch.distributed.get_rank()):
        tokens = tokenizer(prompt, return_tensors="pt")
        generated_ids = megatron_generate(model, tokens.input_ids.cuda(), osl=osl, enable_kv_cache=False)
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
    megatron_save_path: str | None = None,
    export_quant_cfg: str = "fp8",
    calib_size: int = 512,
    compress: bool = False,
    weight_only: bool = False,
    export_kv_cache_quant: bool = False,
    force_all_expert_routing: bool = False,
    prompts: str = "Hello!|Born in California, Soyer trained as a",
) -> None:
    """Perform quantization from HuggingFace model to quantized Megatron-LM model on multiple GPUs."""
    if os.environ.get("WORLD_SIZE") is None:
        console.print("This script must be launched with torchrun. Please run:")
        console.print(f"torchrun --nproc_per_node <gpus> {sys.argv[0]}")
        sys.exit(1)

    bridge = AutoBridge.from_hf_pretrained(hf_model_id)

    model_provider = bridge.to_megatron_provider(load_weights=True)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.transformer_layer_spec = quantization_layer_spec
    # Once all overrides are set, finalize the model provider to ensure the post initialization logic is run
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)
    megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    # Now we can check for rank
    is_rank_0 = torch.distributed.get_rank() == 0

    if is_rank_0:
        console.print(f"[green]Tensor parallel size: {model_provider.tensor_model_parallel_size}[/green]")
        console.print(f"[green]Pipeline parallel size: {model_provider.pipeline_model_parallel_size}[/green]")
        console.print(f"[green]Expert parallel size: {model_provider.expert_model_parallel_size}[/green]")
        console.print(f"[green]Expert tensor parallel size: {model_provider.expert_tensor_parallel_size}[/green]")

    # Formatting
    if is_rank_0:
        table = Table(title="Quantization Statistics")
        table.add_column("Parameter Name", style="cyan")
        table.add_column("Shape")
        table.add_column("Max Value", justify="right")

    # Apply quantization
    if export_quant_cfg in QUANT_CFG_CHOICES:
        if is_rank_0:
            console.print(f"[green]Quantizing the model with {export_quant_cfg} configuration...[/green]")

        # Get the unwrapped model for quantization
        unwrapped_model = unwrap_model(megatron_model)[0]

        # Get quantization configuration
        mtq_config = get_modelopt_torch_quantization_config(export_quant_cfg, export_kv_cache_quant, weight_only)

        # Define forward loop function for calibration
        ptq_forward_loop_func = lambda model: _hf_dataset_forward_loop_func(
            model, bridge.hf_pretrained.tokenizer, calib_size, force_all_expert_routing
        )

        # Apply quantization
        if weight_only:
            mtq.quantize(unwrapped_model, mtq_config)
        elif hasattr(unwrapped_model, "calibration_mode"):
            unwrapped_model.calibration_mode = True
            mtq.quantize(unwrapped_model, mtq_config, ptq_forward_loop_func)
            unwrapped_model.calibration_mode = False
        else:
            mtq.quantize(unwrapped_model, mtq_config, ptq_forward_loop_func)

        if compress:
            mtq.compress(unwrapped_model)
            if is_rank_0:
                console.print("[green]Weights are now compressed to low-bit![/green]")

        if is_rank_0:
            console.print(f"[green]Fake Quantized Model:\n {unwrapped_model}[/green]")

        if is_rank_0:
            for k, v in unwrapped_model.state_dict().items():
                if "amax" not in k and "_scale" not in k:
                    continue
                if isinstance(v, torch.Tensor):
                    table.add_row(k, str(tuple(v.shape)), f"{torch.max(torch.abs(v)):.4e}")
                else:
                    table.add_row(k, "", "")

            console.print(table)

    # Test quantized model with custom prompts
    if is_rank_0:
        console.print("[green]Testing quantized model with custom prompts...[/green]")

    _custom_prompt_forward_loop_func(unwrapped_model, prompts, bridge.hf_pretrained.tokenizer, is_rank_0)

    # Save quantized model in Megatron format
    if megatron_save_path:
        save_path = megatron_save_path
    else:
        # Create default save path using model name and quantization config
        model_name = hf_model_id.split("/")[-1]
        save_path = f"{model_name}_quantized_{export_quant_cfg}"
        if is_rank_0:
            console.print(f"[yellow]No --megatron-save-path specified. Using default path: {save_path}[/yellow]")

    if is_rank_0:
        console.print(f"Saving quantized Megatron checkpoint in {save_path}...")
    bridge.save_megatron_model(megatron_model, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantize HuggingFace model to Megatron-LM format using ModelOpt on multiple GPUs"
    )
    parser.add_argument("--hf-model-id", type=str, default=HF_MODEL_ID, help="HuggingFace model ID to quantize")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")

    parser.add_argument(
        "--megatron-save-path",
        type=str,
        default=None,
        help="Path to save the quantized model in Megatron checkpoint format. If not provided, will use default path: {model_name}_quantized_{config}",
    )
    parser.add_argument(
        "--export-quant-cfg",
        type=str,
        default="fp8",
        choices=list(QUANT_CFG_CHOICES.keys()),
        help="Quantization configuration to use.",
    )
    parser.add_argument(
        "--calib-size",
        type=int,
        default=512,
        help="Samples to use for PTQ calibration.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Enable real low-bit quantization.",
    )
    parser.add_argument(
        "--weight-only",
        action="store_true",
        help="Disable input quantization.",
    )
    parser.add_argument(
        "--export-kv-cache-quant",
        action="store_true",
        help="Enable KV cache quantization.",
    )
    parser.add_argument(
        "--force-all-expert-routing",
        action="store_true",
        help="Forcing all experts to be routed during the calibration.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="Hello!|Born in California, Soyer trained as a",
        help="Input texts for testing quantized model. Please use | to separate different batches.",
    )

    args = parser.parse_args()
    main(
        args.hf_model_id,
        args.tp,
        args.pp,
        args.ep,
        args.etp,
        args.megatron_save_path,
        args.export_quant_cfg,
        args.calib_size,
        args.compress,
        args.weight_only,
        args.export_kv_cache_quant,
        args.force_all_expert_routing,
        args.prompts,
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
