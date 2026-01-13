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
for Vision-Language Models (VLMs) from a Hugging Face model to a quantized
Megatron-LM model on multiple GPUs.

The process is as follows:
1. An AutoBridge is initialized from a pretrained Hugging Face VLM model
    (e.g., "Qwen/Qwen3-VL-8B-Instruct"). This downloads the model from the Hub and loads it.
2. Calibration data is loaded from detection-datasets/coco (has embedded PIL images).
3. ModelOpt quantization is applied to the Megatron-LM model using the specified configuration.
4. The quantized Megatron-LM model is saved in Megatron's native checkpoint format
    using the `--megatron-save-path` argument.

Usage:
    torchrun --nproc_per_node=2 examples/quantization/quantize_vlm.py \
        --export-quant-cfg fp8 \
        --hf-model-id Qwen/Qwen3-VL-8B-Instruct \
        --megatron-save-path ./qwen3_vl_quantized \
        --tp 2
"""

import argparse
import os
import sys
import warnings
from typing import Generator, Optional

import modelopt.torch.quantization as mtq
import torch
from datasets import load_dataset
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.utils import unwrap_model
from modelopt.torch.utils.plugins.megatron_generate import megatron_generate
from quantize_utils import (
    QUANT_CFG_CHOICES,
    add_common_quantization_args,
    console,
    create_quantization_stats_table,
    get_modelopt_torch_quantization_config,
)
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor

from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo


warnings.filterwarnings("ignore")

# Calibration prompts for COCO images (dataset only has images, no text)
CALIBRATION_PROMPTS = [
    "What objects can you see in this image?",
    "Describe this image in detail.",
    "What is happening in this scene?",
    "List the main objects in this image.",
    "What colors are prominent in this image?",
    "Describe the setting of this image.",
    "How many objects are in this image?",
    "What is the main subject of this image?",
    "Describe the composition of this image.",
    "What can you tell me about this image?",
]


def get_coco_dataloader(
    dataset_name: str = "detection-datasets/coco",
    calib_size: int = 512,
) -> Generator[dict, None, None]:
    """Load calibration data from detection-datasets/coco.

    Dataset: https://huggingface.co/datasets/detection-datasets/coco

    Args:
        dataset_name: HuggingFace dataset name.
        calib_size: Number of samples to use for calibration.

    Yields:
        List of messages in OpenAI format.
    """
    # Use custom cache dir to avoid filling up root filesystem
    cache_dir = os.environ.get("HF_HOME", "/opt/Megatron-Bridge/cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set environment variables to override ALL HuggingFace cache locations
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_dir, "hub")

    dataset = load_dataset(
        dataset_name,
        split="train",
        trust_remote_code=True,
        streaming=True,
        cache_dir=cache_dir,
    ).take(calib_size)  # Only download calib_size samples

    for i, sample in enumerate(dataset):
        image = sample["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        prompt = CALIBRATION_PROMPTS[i % len(CALIBRATION_PROMPTS)]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        yield messages


def _hf_dataset_forward_loop_func(
    model,
    processor,
    calib_size: int,
    force_all_expert_routing: bool = False,
    calib_dataset: str = "detection-datasets/coco",
):
    """Forward loop function for calibration using HuggingFace dataset.

    Args:
        model: The VLM model to calibrate.
        processor: HuggingFace processor for the model.
        calib_size: Number of calibration samples.
        force_all_expert_routing: Force all experts during MoE calibration.
        calib_dataset: HuggingFace dataset name for calibration.
    """
    # Force all expert routing for MoE models during calibration
    if force_all_expert_routing:
        for name, module in model.named_modules():
            if isinstance(module, TopKRouter):
                module.topk = module.num_experts

    dataloader = get_coco_dataloader(calib_dataset, calib_size)

    for messages in tqdm(dataloader, total=calib_size, disable=torch.distributed.get_rank(), desc="Calibration"):
        image_inputs, video_inputs = process_vision_info(messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        input_ids = inputs.input_ids.cuda()
        pixel_values = getattr(inputs, "pixel_values", None)
        image_grid_thw = getattr(inputs, "image_grid_thw", None)
        image_sizes = getattr(inputs, "image_sizes", None)
        if pixel_values is not None:
            pixel_values = pixel_values.cuda()
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.cuda()
        if image_sizes is not None:
            image_sizes = image_sizes.cuda()

        megatron_generate(
            model=model,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_sizes=image_sizes,
            osl=1,
            enable_kv_cache=False,
            disable_tqdm=True,
        )

        if force_all_expert_routing:
            for name, module in model.named_modules():
                if isinstance(module, TopKRouter):
                    module.topk = module.config.moe_router_topk


def _custom_prompt_forward_loop_func(
    model,
    processor,
    is_rank_0: bool,
    prompts: str,
    osl: int = 32,
    test_image_path: str = "/models/demo.jpeg",
):
    """Test the quantized VLM model with an image and prompt."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image_path},
                {"type": "text", "text": prompts},
            ],
        }
    ]

    image_inputs, video_inputs = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    input_ids = inputs.input_ids.cuda()
    pixel_values = getattr(inputs, "pixel_values", None)
    image_grid_thw = getattr(inputs, "image_grid_thw", None)
    image_sizes = getattr(inputs, "image_sizes", None)
    if pixel_values is not None:
        pixel_values = pixel_values.cuda()
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.cuda()
    if image_sizes is not None:
        image_sizes = image_sizes.cuda()

    eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    eos_token_ids = [eos_token_id] if eos_token_id else []

    generated_ids = megatron_generate(
        model=model,
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        image_sizes=image_sizes,
        osl=osl,
        eos_token_id=eos_token_ids,
        enable_kv_cache=False,
        disable_tqdm=not is_rank_0,
    )

    if is_rank_0:
        console.print(f"[green]Image:[/green] {test_image_path}")
        console.print(f"[green]Prompt:[/green] {prompts}")
        console.print(
            f"[green]Generated:[/green] {processor.tokenizer.decode(generated_ids[0], skip_special_tokens=False)}"
        )


@torchrun_main
def main(
    hf_model_id: str,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
    megatron_save_path: Optional[str] = None,
    export_quant_cfg: str = "fp8",
    calib_size: int = 512,
    compress: bool = False,
    weight_only: bool = False,
    export_kv_cache_quant: bool = False,
    force_all_expert_routing: bool = False,
    trust_remote_code: bool = True,
    prompts: str = "Describe this image.",
    skip_quantization: bool = False,
) -> None:
    """Perform quantization from HuggingFace VLM model to quantized Megatron-LM model on multiple GPUs."""
    if os.environ.get("WORLD_SIZE") is None:
        console.print("This script must be launched with torchrun. Please run:")
        console.print(f"torchrun --nproc_per_node <gpus> {sys.argv[0]}")
        sys.exit(1)

    bridge = AutoBridge.from_hf_pretrained(
        hf_model_id,
        trust_remote_code=is_safe_repo(
            trust_remote_code=trust_remote_code,
            hf_path=hf_model_id,
        ),
    )

    processor = AutoProcessor.from_pretrained(hf_model_id, trust_remote_code=trust_remote_code)

    model_provider = bridge.to_megatron_provider(load_weights=True)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp
    model_provider.pipeline_dtype = torch.bfloat16

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
        table = create_quantization_stats_table()

    if export_quant_cfg in QUANT_CFG_CHOICES and not skip_quantization:
        if is_rank_0:
            console.print(f"[green]Quantizing the model with {export_quant_cfg} configuration...[/green]")

        # Get the unwrapped model for quantization
        unwrapped_model = unwrap_model(megatron_model)[0]

        # Get quantization configuration
        mtq_config = get_modelopt_torch_quantization_config(export_quant_cfg, export_kv_cache_quant, weight_only)

        # Disable quantization for entire vision_model (all HuggingFace vision components)
        mtq_config["quant_cfg"]["*vision_model*"] = {"enable": False}

        # Define forward loop function for calibration
        def ptq_forward_loop_func(model):
            _hf_dataset_forward_loop_func(
                model,
                processor,
                calib_size,
                force_all_expert_routing,
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
    elif skip_quantization:
        unwrapped_model = unwrap_model(megatron_model)[0]
        if is_rank_0:
            console.print(f"[green]Not Quantized Model:\n {unwrapped_model}[/green]")
            console.print("[yellow]⚠ Skipping quantization (--skip-quantization flag set)[/yellow]")

    # Save quantized model
    if megatron_save_path is None:
        model_name = hf_model_id.replace("/", "_")
        megatron_save_path = f"./{model_name}_quantized_{export_quant_cfg}"

    if is_rank_0:
        console.print("[green]Testing model AFTER quantization...[/green]")

    _custom_prompt_forward_loop_func(unwrapped_model, processor, is_rank_0, prompts)

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
    parser = argparse.ArgumentParser(description="Post-Training Quantization for Vision-Language Models (VLMs)")
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model ID for the VLM model (e.g., Qwen/Qwen3-VL-8B-Instruct)",
    )

    # Add common quantization arguments
    add_common_quantization_args(parser)

    # VLM-specific arguments
    parser.add_argument(
        "--prompts",
        type=str,
        default="Describe this image.",
        help="Text prompt for testing quantized model.",
    )
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        default=False,
        help="Skip quantization (use default layer spec, useful for debugging)",
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
        args.trust_remote_code,
        args.prompts,
        args.skip_quantization,
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
