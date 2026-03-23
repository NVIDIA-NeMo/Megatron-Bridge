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
Kimi VL generation script — converts HF checkpoint to Megatron and runs
greedy auto-regressive generation.

Example:
  torchrun --nproc_per_node=2 examples/models/vlm/kimi_vl/generate.py \
      --hf_model_path ../kimi/kimi_toy \
      --trust_remote_code \
      --image_path "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
      --prompt "Describe this image." \
      --tp 1 --ep 2 --pp 1
"""

import argparse
from typing import Optional

import requests
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.common_utils import get_last_rank, print_rank_0


def _patch_kimi_vision_processor(hf_model_path: str):
    """Monkey-patch KimiK25VisionProcessor.from_dict to avoid duplicate keyword errors.

    The upstream from_dict passes both **config and **kwargs to cls(), which causes
    'got multiple values for keyword argument' when AutoProcessor injects '_from_auto'.
    """
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        klass = get_class_from_dynamic_module(
            "kimi_k25_vision_processing.KimiK25VisionProcessor",
            hf_model_path,
        )
        if klass is None or getattr(klass, "_from_dict_patched", False):
            return

        @classmethod  # type: ignore[misc]
        def _patched_from_dict(cls, config_dict, **kwargs):
            config = config_dict.copy()
            # Remove keys already present in kwargs to prevent duplicates
            for key in list(kwargs.keys()):
                config.pop(key, None)
            media_proc_cfg = config.pop("media_proc_cfg", {})
            return cls(media_proc_cfg=media_proc_cfg, **config, **kwargs)

        klass.from_dict = _patched_from_dict
        klass._from_dict_patched = True
    except Exception:
        pass


def pad_input_ids_to_tp_multiple(input_ids, tp_size: int, pad_token_id: int = 0):
    """Pad input_ids so sequence length is divisible by tp_size.

    This is needed for sequence parallel, which is required for MoE models
    when using tensor parallel and expert parallel together.
    """
    seq_len = input_ids.shape[1]
    remainder = seq_len % tp_size
    if remainder != 0:
        pad_len = tp_size - remainder
        padding = torch.full(
            (input_ids.shape[0], pad_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, padding], dim=1)
    return input_ids


class SingleBatchIterator:
    """Iterator that yields a single batch of data for text generation.
    Required by the forward_backward_func function.
    """

    def __init__(self, input_ids, position_ids, attention_mask, pixel_values=None, grid_thws=None):
        self.batch = dict(
            tokens=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        if pixel_values is not None:
            self.batch["pixel_values"] = pixel_values
        if grid_thws is not None:
            self.batch["image_grid_thw"] = grid_thws

        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def vlm_forward_step(data_iterator, model, **kwargs) -> torch.Tensor:
    """Forward step function for vision-language generation."""
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }

    if "pixel_values" in batch:
        forward_args["pixel_values"] = batch["pixel_values"]
    if "image_grid_thw" in batch:
        forward_args["image_grid_thw"] = batch["image_grid_thw"]

    def loss_func(x, **kwargs):
        return x

    model_output = model(**forward_args)
    if isinstance(model_output, tuple):
        output_tensor, _ = model_output
    else:
        output_tensor = model_output

    return output_tensor, loss_func


def load_image(image_path: str) -> Image.Image:
    """Load an image from URL or file path."""
    if image_path.startswith(("http://", "https://")):
        return Image.open(requests.get(image_path, stream=True).raw)
    else:
        return Image.open(image_path)


def process_image_inputs(processor, tokenizer, image_path: Optional[str], prompt: str, tp_size: int = 1):
    """Process image inputs for Kimi VL model.

    Uses the KimiK25Processor directly with messages format, and pads
    input_ids to be divisible by tp_size for sequence parallel.

    Returns:
        Tuple of (input_ids, pixel_values, grid_thws)
    """
    if image_path:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image_url": load_image(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor(messages=messages)
        input_ids = pad_input_ids_to_tp_multiple(inputs.input_ids, tp_size, tokenizer.pad_token_id or 0)
        grid_thws = getattr(inputs, "grid_thws", None)
        return input_ids, inputs.pixel_values, grid_thws
    else:
        inputs = processor(text=[prompt], return_tensors="pt")
        input_ids = pad_input_ids_to_tp_multiple(inputs.input_ids, tp_size, tokenizer.pad_token_id or 0)
        return input_ids, None, None


def main(args) -> None:
    """Main function for Kimi VL generation."""
    tp = args.tp
    pp = args.pp
    ep = args.ep
    etp = args.etp

    trust_remote = is_safe_repo(
        trust_remote_code=args.trust_remote_code,
        hf_path=args.hf_model_path,
    )

    if args.megatron_model_path:
        print_rank_0(f"Loading Megatron model from: {args.megatron_model_path}")
        bridge = AutoBridge.from_hf_pretrained(args.hf_model_path, trust_remote_code=trust_remote)
        model_provider = bridge.to_megatron_provider(load_weights=False)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp
        model_provider.pipeline_dtype = torch.bfloat16
        model_provider.finalize()
        model_provider.initialize_model_parallel(seed=0)
        model = bridge.load_megatron_model(
            args.megatron_model_path,
            mp_overrides={
                "tensor_model_parallel_size": tp,
                "pipeline_model_parallel_size": pp,
                "expert_model_parallel_size": ep,
                "expert_tensor_parallel_size": etp,
                "pipeline_dtype": torch.bfloat16,
            },
            wrap_with_ddp=False,
        )
    else:
        print_rank_0(f"Loading HuggingFace model from: {args.hf_model_path}")
        bridge = AutoBridge.from_hf_pretrained(args.hf_model_path, trust_remote_code=trust_remote)
        model_provider = bridge.to_megatron_provider(load_weights=True)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp
        model_provider.pipeline_dtype = torch.bfloat16
        model_provider.finalize()
        model_provider.initialize_model_parallel(seed=0)
        model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    # Disable MTP for inference
    def _disable_mtp(m):
        m.config.mtp_num_layers = None
        inner = m.module if hasattr(m, "module") else m
        lang = getattr(inner, "language_model", inner)
        if hasattr(lang, "mtp_process"):
            lang.mtp_process = False

    model = [m.cuda() for m in model]
    for m in model:
        m.eval()
        _disable_mtp(m)
        if hasattr(m, "config"):
            m.config.grad_scale_func = None

    # Initialize tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=trust_remote)
    _patch_kimi_vision_processor(args.hf_model_path)
    processor = AutoProcessor.from_pretrained(args.hf_model_path, trust_remote_code=trust_remote)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Process inputs
    input_ids, pixel_values, grid_thws = process_image_inputs(
        processor, tokenizer, args.image_path, args.prompt, tp_size=tp
    )

    # Move to GPU
    input_ids = input_ids.cuda()
    if pixel_values is not None:
        if isinstance(pixel_values, (list, tuple)):
            pixel_values = [pv.cuda() for pv in pixel_values]
        else:
            pixel_values = pixel_values.cuda()
    if grid_thws is not None:
        if isinstance(grid_thws, (list, tuple)):
            grid_thws = [g.cuda() for g in grid_thws]
        else:
            grid_thws = grid_thws.cuda()

    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    )
    # Megatron-Core convention: True in attention_mask means "mask OUT this token"
    # (opposite of HuggingFace). Passing None lets Megatron auto-generate the correct causal mask.
    attention_mask = None
    generated_ids = input_ids.clone()

    stop_tokens = [tokenizer.eos_token_id]

    # Greedy generation loop
    for step in range(args.max_new_tokens):
        with torch.no_grad():
            print_rank_0(f"Generation step {step}")

            fwd_bwd_function = get_forward_backward_func()
            iterator = SingleBatchIterator(input_ids, position_ids, attention_mask, pixel_values, grid_thws)

            output = fwd_bwd_function(
                forward_step_func=vlm_forward_step,
                data_iterator=iterator,
                model=model,
                num_microbatches=1,
                forward_only=True,
                seq_length=input_ids.size(1),
                micro_batch_size=1,
                collect_non_loss_data=True,
            )
            if isinstance(output, list) and len(output) > 0:
                output = output[0]

            if parallel_state.is_pipeline_last_stage():
                world_size = parallel_state.get_tensor_model_parallel_world_size()
                gathered_tensors = [torch.zeros_like(output) for _ in range(world_size)]
                dist.all_gather(gathered_tensors, output, group=parallel_state.get_tensor_model_parallel_group())
                output = torch.cat(gathered_tensors, dim=2)
                next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)

                if step < 5:
                    print_rank_0(f"Step {step}: output shape={output.shape}, var={output.var():.4f}")
                    logits = output[0, -1, :]
                    top5_vals, top5_ids = torch.topk(logits, 5)
                    top5_tokens = [tokenizer.decode([idx]) for idx in top5_ids]
                    print_rank_0(f"Top 5: {list(zip(top5_tokens, top5_vals.tolist()))}")
                    print_rank_0(
                        f"Selected: '{tokenizer.decode([next_token_ids.item()])}' (id={next_token_ids.item()})"
                    )
            else:
                next_token_ids = torch.ones((1, 1), device=generated_ids.device, dtype=generated_ids.dtype)

            torch.distributed.broadcast(next_token_ids, get_last_rank())
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            input_ids = pad_input_ids_to_tp_multiple(generated_ids, tp, tokenizer.pad_token_id or 0)
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )

            if next_token_ids.item() in stop_tokens:
                break

    generated_text = tokenizer.decode(list(generated_ids[0]))
    print_rank_0("======== GENERATED TEXT OUTPUT ========")
    if args.image_path:
        print_rank_0(f"Image: {args.image_path}")
    print_rank_0(f"Prompt: {args.prompt}")
    print_rank_0(f"Generated: {generated_text}")
    print_rank_0("=======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kimi VL Generation")
    parser.add_argument("--hf_model_path", type=str, required=True, help="Path to the HuggingFace Kimi VL model.")
    parser.add_argument("--prompt", type=str, default="Describe this image.", help="Input prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Maximum number of new tokens to generate.")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")
    parser.add_argument("--megatron_model_path", type=str, default=None, help="Path to Megatron model checkpoint")
    parser.add_argument("--image_path", type=str, default=None, help="Path or URL to image (optional).")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for HF model loading")
    args = parser.parse_args()

    main(args)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
