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

"""ERNIE 4.5 VL: HF-to-Megatron generate (VLM).

ERNIE 4.5 VL uses a custom processor API that differs from Qwen-style models:
  - ``processor.tokenizer.apply_chat_template()`` instead of ``processor.apply_chat_template()``
  - ``processor.process_vision_info()`` for image pre-processing
  - Output keys: "images" (pixel_values), "grid_thw" (image_grid_thw)
  - mm_token_type_ids must be constructed from image_token_id positions

This script mirrors ``hf_to_megatron_generate_vlm.py`` but handles ERNIE-specific
processor differences.

Example:
  # Single GPU:
  torchrun --nproc_per_node=1 examples/models/vlm/ernie_vl/hf_to_megatron_generate_ernie_vl.py \
      --hf_model_path baidu/ERNIE-4.5-VL-28B-A3B-Instruct \
      --image_path /path/to/image.png \
      --prompt "Describe this image."

  # Multi-GPU (TP=2, EP=4):
  torchrun --nproc_per_node=8 examples/models/vlm/ernie_vl/hf_to_megatron_generate_ernie_vl.py \
      --hf_model_path baidu/ERNIE-4.5-VL-28B-A3B-Instruct \
      --image_path /path/to/image.png \
      --prompt "Describe this image." \
      --tp 2 --ep 4
"""

import argparse
import json
import os
import sys
import types


os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# Fake 'decord' module -- ERNIE VL processor imports it but only uses it for video.
if "decord" not in sys.modules:

    class _FakeVideoReader:
        def __init__(self, *a, **kw):
            raise RuntimeError("decord not installed; video processing unavailable")

    _decord_fake = types.ModuleType("decord")
    _decord_fake.VideoReader = _FakeVideoReader
    _decord_fake.cpu = lambda x=0: x
    _bridge = types.ModuleType("decord.bridge")
    _bridge.set_bridge = lambda *a, **kw: None
    sys.modules["decord"] = _decord_fake
    sys.modules["decord.bridge"] = _bridge

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from transformers import AutoProcessor

from megatron.bridge import AutoBridge
from megatron.bridge.utils.common_utils import get_last_rank, print_rank_0, print_rank_last


# ---------------------------------------------------------------------------
# Forward step
# ---------------------------------------------------------------------------


class SingleBatchIterator:
    """Iterator that yields a single batch then stops.  Required by
    ``get_forward_backward_func``."""

    def __init__(
        self, input_ids, position_ids, attention_mask, pixel_values=None, image_grid_thw=None, mm_token_type_ids=None
    ):
        self.batch = dict(
            tokens=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        if pixel_values is not None:
            self.batch["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            self.batch["image_grid_thw"] = image_grid_thw
        if mm_token_type_ids is not None:
            self.batch["mm_token_type_ids"] = mm_token_type_ids
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def ernie_vl_forward_step(data_iterator, model, **kwargs) -> torch.Tensor:
    """Forward step for ERNIE VL generation."""
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "mm_token_type_ids": batch.get("mm_token_type_ids"),
        "moe_mm_token_type_ids": batch.get("mm_token_type_ids"),
        "attention_mask": batch.get("attention_mask"),
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


# ---------------------------------------------------------------------------
# Input processing
# ---------------------------------------------------------------------------


def process_ernie_vl_inputs(processor, hf_model_path, image_path, prompt):
    """Process inputs using ERNIE 4.5 VL processor API.

    Returns:
        (input_ids, pixel_values, image_grid_thw, mm_token_type_ids)
        All tensors are on CPU; caller moves to GPU.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_path}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = processor.process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("images")  # ERNIE uses "images" key
    image_grid_thw = inputs.get("grid_thw")  # ERNIE uses "grid_thw" key

    # Build mm_token_type_ids from image_token_id positions.
    # The HF processor's token_type_ids marks IMAGE_START/END as type 1,
    # but Megatron expects only actual image placeholder tokens to be type 1.
    with open(os.path.join(hf_model_path, "config.json")) as f:
        hf_cfg = json.load(f)
    image_token_id = hf_cfg.get("image_token_id", 100295)

    mm_token_type_ids = torch.zeros(1, input_ids.shape[1], dtype=torch.int32)
    image_placeholder_mask = input_ids == image_token_id
    mm_token_type_ids[image_placeholder_mask] = 1

    return input_ids, pixel_values, image_grid_thw, mm_token_type_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def to_cuda(x):
    """Move tensor to CUDA if not None."""
    if x is None:
        return None
    return x.cuda()


def main(args):
    """Main generation function."""
    print_rank_0("=" * 60)
    print_rank_0("ERNIE 4.5 VL -- HF to Megatron Generate")
    print_rank_0("=" * 60)

    trust_remote = args.trust_remote_code
    tp, pp, ep = args.tp, args.pp, args.ep

    # ------------------------------------------------------------------
    # Load model via AutoBridge
    # ------------------------------------------------------------------
    print_rank_0(f"Loading model: {args.hf_model_path}")
    bridge = AutoBridge.from_hf_pretrained(
        args.hf_model_path, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote
    )
    model_provider = bridge.to_megatron_provider(load_weights=True)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.params_dtype = torch.bfloat16
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)
    model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    model = [m.cuda() for m in model]
    for m in model:
        m.eval()
        if hasattr(m, "config"):
            m.config.grad_scale_func = None
            m.config.deallocate_pipeline_outputs = False

    # ------------------------------------------------------------------
    # Processor
    # ------------------------------------------------------------------
    processor = AutoProcessor.from_pretrained(args.hf_model_path, trust_remote_code=trust_remote)
    eos_token_id = processor.tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # Process inputs
    # ------------------------------------------------------------------
    input_ids_raw, pixel_values, image_grid_thw, mm_token_type_ids = process_ernie_vl_inputs(
        processor, args.hf_model_path, args.image_path, args.prompt
    )

    input_ids_raw = input_ids_raw.cuda()
    pixel_values = to_cuda(pixel_values)
    image_grid_thw = to_cuda(image_grid_thw)
    mm_token_type_ids = to_cuda(mm_token_type_ids)

    print_rank_0(f"Input tokens: {input_ids_raw.shape[1]}, Image tokens: {(mm_token_type_ids == 1).sum().item()}")

    # ------------------------------------------------------------------
    # Greedy generation loop
    # ------------------------------------------------------------------
    generated_ids = input_ids_raw.clone()

    for step in range(args.max_new_tokens):
        with torch.no_grad():
            print_rank_0(f"Generation step {step}")

            real_seq_len = generated_ids.size(1)
            input_ids = generated_ids

            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )

            fwd_bwd_function = get_forward_backward_func()
            iterator = SingleBatchIterator(
                input_ids,
                position_ids,
                None,
                pixel_values,
                image_grid_thw,
                mm_token_type_ids,
            )

            output = fwd_bwd_function(
                forward_step_func=ernie_vl_forward_step,
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
                dist.all_gather(
                    gathered_tensors,
                    output,
                    group=parallel_state.get_tensor_model_parallel_group(),
                )
                output = torch.cat(gathered_tensors, dim=2)

                last_pos = real_seq_len - 1
                next_token_ids = torch.argmax(output[:, last_pos], dim=-1, keepdim=True)

                if step < 5:
                    logits = output[0, last_pos, :]
                    top5_vals, top5_ids = torch.topk(logits, 5)
                    top5_tokens = [processor.tokenizer.decode([idx]) for idx in top5_ids]
                    print_rank_last(f"Top 5: {list(zip(top5_tokens, top5_vals.tolist()))}")
                    print_rank_last(
                        f"Selected: '{processor.tokenizer.decode([next_token_ids.item()])}' "
                        f"(id={next_token_ids.item()})"
                    )
            else:
                next_token_ids = torch.ones((1, 1), device=generated_ids.device, dtype=generated_ids.dtype)

            torch.distributed.broadcast(next_token_ids, get_last_rank())
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            if mm_token_type_ids is not None:
                mm_token_type_ids = torch.cat(
                    [mm_token_type_ids, torch.zeros_like(next_token_ids, dtype=mm_token_type_ids.dtype)],
                    dim=-1,
                )

            if next_token_ids.item() == eos_token_id:
                break

    generated_text = processor.tokenizer.decode(list(generated_ids[0, input_ids_raw.shape[1] :]))
    print_rank_0("======== GENERATED TEXT OUTPUT ========")
    if args.image_path:
        print_rank_0(f"Image: {args.image_path}")
    print_rank_0(f"Prompt: {args.prompt}")
    print_rank_0(f"Generated: {generated_text}")
    print_rank_0("=======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ERNIE 4.5 VL: HF to Megatron Generate")
    parser.add_argument("--hf_model_path", type=str, required=True, help="Path to the HuggingFace ERNIE 4.5 VL model.")
    parser.add_argument("--prompt", type=str, default="Describe this image.", help="Input prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate.")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--image_path", type=str, default=None, help="Path or URL to image (optional).")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for HF model loading")
    args = parser.parse_args()

    main(args)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
