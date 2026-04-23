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
Inference script for Gemma 4 VLM (google/gemma-4-26B-A4B and variants).

Uses the HF AutoProcessor with apply_chat_template to prepare inputs
compatible with Gemma4VLModel's forward signature:
  - input_ids (with image token placeholders at 258880)
  - pixel_values
  - image_position_ids (optional, from processor)

Example (text-only):
  uv run python examples/conversion/hf_to_megatron_generate_gemma4.py \\
    --hf_model_path="google/gemma-4-26B-A4B" \\
    --prompt="What is the capital of France?" \\
    --max_new_tokens=30

Example (VLM with image URL):
  uv run python examples/conversion/hf_to_megatron_generate_gemma4.py \\
    --hf_model_path="google/gemma-4-26B-A4B-it" \\
    --image_path="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg" \\
    --prompt="What is shown in this image?" \\
    --max_new_tokens=50
"""

import argparse
import io
from typing import Optional

import requests
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.utils.common_utils import get_last_rank, print_rank_0


class SingleBatchIterator:
    """Iterator yielding one batch for the forward_backward_func."""

    def __init__(self, input_ids, position_ids, attention_mask, pixel_values=None, image_position_ids=None):
        self.batch = dict(
            tokens=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        if pixel_values is not None:
            self.batch["pixel_values"] = pixel_values
        if image_position_ids is not None:
            self.batch["image_position_ids"] = image_position_ids
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def gemma4_forward_step(data_iterator, model, **kwargs) -> torch.Tensor:
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }
    if "pixel_values" in batch:
        forward_args["pixel_values"] = batch["pixel_values"]
    if "image_position_ids" in batch:
        forward_args["image_position_ids"] = batch["image_position_ids"]

    def loss_func(x, **kwargs):
        return x

    model_output = model(**forward_args)
    if isinstance(model_output, tuple):
        output_tensor, _ = model_output
    else:
        output_tensor = model_output
    return output_tensor, loss_func


def load_image(image_path: str) -> Image.Image:
    if image_path.startswith(("http://", "https://")):
        response = requests.get(image_path, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    return Image.open(image_path).convert("RGB")


def prepare_inputs(processor, image_path: Optional[str], prompt: str):
    """Build input_ids + pixel_values for Gemma4.

    Uses apply_chat_template if available (instruction-tuned models),
    falls back to raw processor/tokenizer for base models.
    """
    if image_path:
        image = load_image(image_path)
        # Try chat template first (works for -it models)
        has_chat_template = (
            hasattr(processor, "apply_chat_template")
            and getattr(processor, "chat_template", None) is not None
        )
        if has_chat_template:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        else:
            # Base model: use processor directly with image+text.
            # Gemma4 processor requires the image token in the text to align
            # image embeddings with text tokens.
            image_token = getattr(processor.tokenizer, "image_token", "<|image|>")
            if image_token not in prompt:
                prompt = image_token + "\n" + prompt
            inputs = processor(
                text=[prompt],
                images=[image],
                return_tensors="pt",
            )
    else:
        # Text-only: use tokenizer directly (no image processing needed)
        inputs = processor.tokenizer(text=[prompt], return_tensors="pt")

    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values", None)
    image_position_ids = inputs.get("image_position_ids", None)
    return input_ids, pixel_values, image_position_ids


def main(args) -> None:
    tp = args.tp
    pp = args.pp
    ep = args.ep
    etp = args.etp

    bridge = AutoBridge.from_hf_pretrained(args.hf_model_path)

    if args.megatron_model_path:
        print_rank_0(f"Loading Megatron model from: {args.megatron_model_path}")
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
        model_provider = bridge.to_megatron_provider(load_weights=True)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp
        model_provider.pipeline_dtype = torch.bfloat16
        model_provider.finalize()
        model_provider.initialize_model_parallel(seed=0)
        model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    model = [m.cuda() for m in model]
    for m in model:
        m.eval()
        # Disable MTP (training-only feature)
        m.config.mtp_num_layers = None
        inner = m.module if hasattr(m, "module") else m
        lang = getattr(inner, "language_model", inner)
        if hasattr(lang, "mtp_process"):
            lang.mtp_process = False

    for m in model:
        if hasattr(m, "config"):
            m.config.grad_scale_func = None

    processor = AutoProcessor.from_pretrained(args.hf_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids, pixel_values, image_position_ids = prepare_inputs(processor, args.image_path, args.prompt)

    input_ids = input_ids.cuda()
    if pixel_values is not None:
        pixel_values = pixel_values.cuda().to(torch.bfloat16)
    if image_position_ids is not None:
        image_position_ids = image_position_ids.cuda()

    print_rank_0(f"Input shape: {input_ids.shape}, has image: {pixel_values is not None}")

    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        .unsqueeze(0)
        .expand_as(input_ids)
    )
    # Pass attention_mask=None so Megatron uses its built-in causal masking.
    # Passing a 2D bool mask breaks causal masking for Gemma4.
    attention_mask = None
    original_prompt_len = input_ids.size(1)
    generated_ids = input_ids.clone()

    # Build stop token set: eos + all model-specific end-of-turn tokens.
    # Gemma4 uses <turn|> (id 106) to signal end-of-turn; Llama uses <|eot_id|>.
    stop_tokens = {tokenizer.eos_token_id}
    for name in ["<turn|>", "<end_of_turn>", "<|end_of_turn|>", "<|eot_id|>"]:
        tid = tokenizer.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            stop_tokens.add(tid)
    for step in range(args.max_new_tokens):
        with torch.no_grad():
            print_rank_0(f"Generation step {step}")

            # Without KV-cache, the full sequence is re-processed at every step.
            # pixel_values must be passed every step so image features are re-inserted
            # at the image token positions on each forward pass.
            pv = pixel_values
            ipids = image_position_ids

            iterator = SingleBatchIterator(input_ids, position_ids, attention_mask, pv, ipids)
            fwd_bwd_function = get_forward_backward_func()

            output = fwd_bwd_function(
                forward_step_func=gemma4_forward_step,
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
                gathered = [torch.zeros_like(output) for _ in range(world_size)]
                dist.all_gather(gathered, output, group=parallel_state.get_tensor_model_parallel_group())
                output = torch.cat(gathered, dim=2)
                next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)

                if step < 5:
                    logits = output[0, -1, :]
                    top5_vals, top5_ids = torch.topk(logits, 5)
                    top5_tokens = [tokenizer.decode([idx]) for idx in top5_ids]
                    print_rank_0(f"Step {step}: shape={output.shape}, var={output.var():.4f}")
                    print_rank_0(f"Top 5: {list(zip(top5_tokens, top5_vals.tolist()))}")
                    print_rank_0(f"Selected: '{tokenizer.decode([next_token_ids.item()])}' (id={next_token_ids.item()})")
            else:
                next_token_ids = torch.ones((1, 1), device=generated_ids.device, dtype=generated_ids.dtype)

            torch.distributed.broadcast(next_token_ids, get_last_rank())
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            input_ids = generated_ids
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )
            attention_mask = None

            if next_token_ids.item() in stop_tokens:
                break

    # Decode only the newly generated tokens (excluding the original prompt)
    generated_text = tokenizer.decode(list(generated_ids[0]), skip_special_tokens=False)
    new_tokens_text = tokenizer.decode(
        list(generated_ids[0][original_prompt_len:]),
        skip_special_tokens=True,
    )

    print_rank_0("======== GENERATED TEXT OUTPUT ========")
    if args.image_path:
        print_rank_0(f"Image: {args.image_path}")
    print_rank_0(f"Prompt: {args.prompt}")
    print_rank_0(f"New tokens: {new_tokens_text}")
    print_rank_0("=======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma 4 VLM Inference via Megatron Bridge")
    parser.add_argument("--hf_model_path", type=str, required=True)
    parser.add_argument("--megatron_model_path", type=str, default=None, help="Path to imported Megatron checkpoint (optional)")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?")
    parser.add_argument("--image_path", type=str, default=None, help="URL or local path to image (optional)")
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--etp", type=int, default=1)
    args = parser.parse_args()

    main(args)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
