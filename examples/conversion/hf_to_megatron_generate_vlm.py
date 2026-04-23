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
Example:
  # Vision-Language generation with image from URL:
  uv run python examples/conversion/hf_to_megatron_generate_vlm.py --hf_model_path="Qwen/Qwen2.5-VL-3B-Instruct" --image_path="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" --prompt="Describe this image."

  # Vision-Language generation with local image:
  uv run python examples/conversion/hf_to_megatron_generate_vlm.py --hf_model_path="Qwen/Qwen2.5-VL-3B-Instruct" --image_path="/path/to/image.jpg" --prompt="What do you see in this image?"

  # Text-only generation (no image):
  uv run python examples/conversion/hf_to_megatron_generate_vlm.py --hf_model_path="Qwen/Qwen2.5-VL-3B-Instruct" --prompt="Hello, how are you?"

  # Load from Megatron checkpoint:
  uv run python examples/conversion/hf_to_megatron_generate_vlm.py --hf_model_path="Qwen/Qwen2.5-VL-3B-Instruct" --megatron_model_path="/path/to/megatron/checkpoint" --image_path="/path/to/image.jpg" --prompt="Describe this image."
"""

import argparse

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from vlm_generate_utils import (
    pad_input_ids_to_tp_multiple,
    patch_kimi_vision_processor,
    process_image_inputs,
    to_cuda,
)

from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.common_utils import get_last_rank, print_rank_0, print_rank_last


# ---------------------------------------------------------------------------
# Forward step
# ---------------------------------------------------------------------------


class SingleBatchIterator:
    """Iterator that yields a single batch then stops.  Required by
    ``get_forward_backward_func``."""

    def __init__(
        self,
        input_ids,
        position_ids,
        attention_mask,
        pixel_values=None,
        image_grid_thw=None,
        image_sizes=None,
        mm_token_type_ids=None,
        image_position_ids=None,
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
        if image_sizes is not None:
            self.batch["image_sizes"] = image_sizes
        if mm_token_type_ids is not None:
            self.batch["mm_token_type_ids"] = mm_token_type_ids
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


def vlm_forward_step(data_iterator, model, **kwargs) -> torch.Tensor:
    """Forward step for VLM generation."""
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask"),
    }
    for key in ("pixel_values", "image_grid_thw", "image_sizes", "mm_token_type_ids"):
        if key in batch:
            forward_args[key] = batch[key]

    def loss_func(x, **kwargs):
        return x

    model_output = model(**forward_args)
    if isinstance(model_output, tuple):
        output_tensor, _ = model_output
    else:
        output_tensor = model_output
    return output_tensor, loss_func


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args) -> None:
    """Run VLM inference with HuggingFace or Megatron checkpoints."""
    tp = args.tp
    pp = args.pp
    ep = args.ep
    etp = args.etp

    trust_remote = is_safe_repo(
        trust_remote_code=args.trust_remote_code,
        hf_path=args.hf_model_path,
    )

    # Detect model family for processor-specific handling
    config = AutoConfig.from_pretrained(args.hf_model_path, trust_remote_code=trust_remote)
    model_type = getattr(config, "model_type", "")
    is_kimi = "kimi" in model_type
    image_token_id = getattr(config, "image_token_id", None)
    if is_kimi and image_token_id is None:
        image_token_id = 163605

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    bridge = AutoBridge.from_hf_pretrained(args.hf_model_path, trust_remote_code=trust_remote)

    if args.megatron_model_path:
        print_rank_0(f"Loading Megatron model from: {args.megatron_model_path}")
        model_provider = bridge.to_megatron_provider(load_weights=False)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp
        model_provider.pipeline_dtype = torch.bfloat16
        model_provider.init_model_with_meta_device = True
        if args.pp_layout:
            model_provider.pipeline_model_parallel_layout = args.pp_layout
        model_provider.finalize()
        model_provider.initialize_model_parallel(seed=0)

        mp_overrides = {
            "tensor_model_parallel_size": tp,
            "pipeline_model_parallel_size": pp,
            "expert_model_parallel_size": ep,
            "expert_tensor_parallel_size": etp,
            "pipeline_dtype": torch.bfloat16,
        }
        if args.pp_layout:
            mp_overrides["pipeline_model_parallel_layout"] = args.pp_layout
        model = bridge.load_megatron_model(
            args.megatron_model_path,
            mp_overrides=mp_overrides,
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

    # ------------------------------------------------------------------
    # Tokenizer & processor
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=trust_remote)
    if is_kimi:
        patch_kimi_vision_processor(args.hf_model_path)
    processor = AutoProcessor.from_pretrained(args.hf_model_path, trust_remote_code=trust_remote)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id or 0

    # ------------------------------------------------------------------
    # Process inputs
    # ------------------------------------------------------------------
    input_ids_raw, pixel_values, image_grid_thw, image_sizes, mm_token_type_ids = process_image_inputs(
        processor,
        args.image_path,
        args.prompt,
        is_kimi=is_kimi,
        image_token_id=image_token_id,
    )

    input_ids_raw = input_ids_raw.cuda()
    pixel_values = to_cuda(pixel_values)
    image_grid_thw = to_cuda(image_grid_thw)
    image_sizes = to_cuda(image_sizes)
    mm_token_type_ids = to_cuda(mm_token_type_ids)

    # ------------------------------------------------------------------
    # Greedy generation loop
    # ------------------------------------------------------------------
    generated_ids = input_ids_raw.clone()
    stop_tokens = [tokenizer.eos_token_id]

    for step in range(args.max_new_tokens):
        with torch.no_grad():
            print_rank_0(f"Generation step {step}")

            real_seq_len = generated_ids.size(1)
            input_ids = pad_input_ids_to_tp_multiple(generated_ids, tp, pad_token_id)

            mm_ids_padded = None
            if mm_token_type_ids is not None:
                mm_ids_padded = pad_input_ids_to_tp_multiple(mm_token_type_ids, tp, 0)

            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )

            fwd_bwd_function = get_forward_backward_func()
            iterator = SingleBatchIterator(
                input_ids, position_ids, None, pixel_values, image_grid_thw, image_sizes, mm_ids_padded
            )

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

                last_pos = real_seq_len - 1
                next_token_ids = torch.argmax(output[:, last_pos], dim=-1, keepdim=True)

                if step < 5:
                    print_rank_last(
                        f"Step {step}: output shape={output.shape}, "
                        f"real_seq_len={real_seq_len}, var={output.var():.4f}"
                    )
                    logits = output[0, last_pos, :]
                    top5_vals, top5_ids = torch.topk(logits, 5)
                    top5_tokens = [tokenizer.decode([idx]) for idx in top5_ids]
                    print_rank_last(f"Top 5: {list(zip(top5_tokens, top5_vals.tolist()))}")
                    print_rank_last(
                        f"Selected: '{tokenizer.decode([next_token_ids.item()])}' (id={next_token_ids.item()})"
                    )
            else:
                next_token_ids = torch.ones((1, 1), device=generated_ids.device, dtype=generated_ids.dtype)

            torch.distributed.broadcast(next_token_ids, get_last_rank())
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            if mm_token_type_ids is not None:
                mm_token_type_ids = torch.cat(
                    [mm_token_type_ids, torch.zeros_like(next_token_ids, dtype=mm_token_type_ids.dtype)], dim=-1
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
    parser = argparse.ArgumentParser(description="VLM Generation from HuggingFace Models")
    parser.add_argument("--hf_model_path", type=str, required=True, help="Path to the HuggingFace VL model.")
    parser.add_argument("--prompt", type=str, default="Describe this image.", help="Input prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Maximum number of new tokens to generate.")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")
    parser.add_argument(
        "--pp_layout", type=str, default=None, help="Pipeline model parallel layout (e.g. 'Et*15|t*15|t*16|t*15L')"
    )
    parser.add_argument("--megatron_model_path", type=str, default=None, help="Path to Megatron model checkpoint")
    parser.add_argument("--image_path", type=str, default=None, help="Path or URL to image (optional).")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for HF model loading")
    args = parser.parse_args()

    main(args)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
