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
1. An AutoBridge is initialized from the pretrained Hugging Face model given by `--hf-model-id`.
    This downloads the model from the Hub (or loads it from a local path) and loads it.
2. ModelOpt quantization is applied to the Megatron-LM model using the specified configuration.
3. The quantized Megatron-LM model is saved in Megatron's native checkpoint format
    using the `--megatron-save-path` argument.

Usage:
torchrun --nproc_per_node 2 examples/quantization/quantize.py \
    --hf-model-id meta-llama/Llama-3.2-1B --export-quant-cfg fp8
torchrun --nproc_per_node 2 examples/quantization/quantize.py \
    --hf-model-id meta-llama/Llama-3.2-1B --export-quant-cfg fp8 --megatron-save-path ./megatron_checkpoint
"""

import argparse
import warnings

import modelopt.torch.quantization as mtq
import modelopt.torch.utils.distributed as dist
import torch
from megatron.core.utils import unwrap_model
from modelopt.torch.utils.plugins.megatron_calibration import get_megatron_calibration_forward_loop
from modelopt.torch.utils.plugins.megatron_generate import megatron_generate
from quantize_utils import (
    QUANT_CFG_CHOICES,
    add_common_quantization_args,
    build_bridge_and_provider,
    console,
    create_quantization_stats_table,
    get_modelopt_torch_quantization_config,
    print_parallelism_summary,
    require_torchrun,
)

from megatron.bridge.models.decorators import torchrun_main


warnings.filterwarnings("ignore")

# Resolved by ModelOpt's dataset registry, which pins the concrete namespaced Hub id.
DEFAULT_CALIB_DATASET = "cnn_dailymail"


def _custom_prompt_forward_loop_func(model, prompts, tokenizer, is_rank_0, osl=32):
    """Forward loop function for testing quantized model with custom prompts."""
    all_prompts = prompts.split("|")

    for idx, prompt in enumerate(all_prompts):
        tokens = tokenizer(prompt, return_tensors="pt")
        generated_ids = megatron_generate(model, tokens.input_ids.cuda(), osl=osl, enable_kv_cache=False)
        generated_texts = tokenizer.batch_decode(generated_ids)
        if is_rank_0:
            console.print(f"[green]Prompt {idx + 1}: {prompt}[/green]")
            console.print(f"[green]Generated: {generated_texts}[/green]")


@torchrun_main
def main(
    hf_model_id: str,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
    megatron_save_path: str | None = None,
    export_quant_cfg: str = "fp8",
    calib_size: int = 512,
    calib_seq_length: int = 512,
    calib_batch_size: int = 1,
    calib_dataset: str = DEFAULT_CALIB_DATASET,
    compress: bool = False,
    weight_only: bool = False,
    export_kv_cache_quant: bool = False,
    prompts: str = "Hello!|Born in California, Soyer trained as a",
    trust_remote_code: bool | None = None,
) -> None:
    """Perform quantization from HuggingFace model to quantized Megatron-LM model on multiple GPUs."""
    require_torchrun()

    bridge, model_provider = build_bridge_and_provider(
        hf_model_id,
        tp=tp,
        pp=pp,
        ep=ep,
        etp=etp,
        load_weights=True,
        pipeline_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
    )
    megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    # Now we can check for rank
    is_rank_0 = dist.is_master()

    print_parallelism_summary(model_provider)

    # Formatting
    if is_rank_0:
        table = create_quantization_stats_table()

    # Apply quantization
    if export_quant_cfg in QUANT_CFG_CHOICES:
        if is_rank_0:
            console.print(f"[green]Quantizing the model with {export_quant_cfg} configuration...[/green]")

        # Get the unwrapped model for quantization
        unwrapped_model = unwrap_model(megatron_model)[0]

        # Get quantization configuration
        mtq_config = get_modelopt_torch_quantization_config(export_quant_cfg, export_kv_cache_quant, weight_only)

        # These configs derive scales from weights alone, so skip the dataset download and forward pass.
        if weight_only or not mtq.need_calibration(mtq_config):
            if is_rank_0:
                console.print("[yellow]Weight-only or dynamic quantization: skipping calibration.[/yellow]")
            ptq_forward_loop_func = None
        else:
            ptq_forward_loop_func = get_megatron_calibration_forward_loop(
                bridge.hf_pretrained.tokenizer,
                dataset_name=calib_dataset,
                num_samples=calib_size,
                seq_length=calib_seq_length,
                batch_size=calib_batch_size,
                pack=True,
            )

        # Apply quantization
        if hasattr(unwrapped_model, "calibration_mode"):
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

    dist.barrier()

    # Save quantized model in Megatron format
    if megatron_save_path:
        save_path = megatron_save_path
    else:
        model_name = hf_model_id.split("/")[-1]
        save_path = f"{model_name}_quantized_{export_quant_cfg}"
        if is_rank_0:
            console.print(f"[yellow]No --megatron-save-path specified. Using default path: {save_path}[/yellow]")

    if is_rank_0:
        console.print(f"Saving quantized Megatron checkpoint in {save_path}...")
    bridge.save_megatron_model(megatron_model, save_path)

    dist.barrier()

    # Test quantized model with custom prompts
    if export_quant_cfg in QUANT_CFG_CHOICES:
        if is_rank_0:
            console.print("[green]Testing quantized model with custom prompts...[/green]")

        _custom_prompt_forward_loop_func(unwrapped_model, prompts, bridge.hf_pretrained.tokenizer, is_rank_0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantize HuggingFace model to Megatron-LM format using ModelOpt on multiple GPUs"
    )
    add_common_quantization_args(parser)

    # LLM-specific arguments
    parser.add_argument(
        "--prompts",
        type=str,
        default="Hello!|Born in California, Soyer trained as a",
        help="Input texts for testing quantized model. Please use | to separate different batches.",
    )
    parser.add_argument(
        "--calib-seq-length",
        type=int,
        default=512,
        help="Calibration sequence length in tokens.",
    )
    parser.add_argument(
        "--calib-batch-size",
        type=int,
        default=1,
        help="Calibration batch size. Raise it to speed up calibration when memory allows.",
    )
    parser.add_argument(
        "--calib-dataset",
        type=str,
        default=DEFAULT_CALIB_DATASET,
        help="Calibration dataset: a ModelOpt registered dataset name, a HuggingFace dataset path, "
        "a local directory, or a .jsonl file. Use a local path when running without Hub access.",
    )
    parser.add_argument(
        "--disable-hf-datasets-file-lock",
        action="store_true",
        help="Disable HF datasets file lock. This is only needed when testing with data in a read-only directory.",
    )

    args = parser.parse_args()
    if args.disable_hf_datasets_file_lock:
        from unittest.mock import MagicMock

        import datasets

        datasets.builder.FileLock = MagicMock()

    main(
        hf_model_id=args.hf_model_id,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        etp=args.etp,
        megatron_save_path=args.megatron_save_path,
        export_quant_cfg=args.export_quant_cfg,
        calib_size=args.calib_size,
        calib_seq_length=args.calib_seq_length,
        calib_batch_size=args.calib_batch_size,
        calib_dataset=args.calib_dataset,
        compress=args.compress,
        weight_only=args.weight_only,
        export_kv_cache_quant=args.export_kv_cache_quant,
        prompts=args.prompts,
        trust_remote_code=args.trust_remote_code,
    )

    dist.cleanup()
