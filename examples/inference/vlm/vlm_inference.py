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
  uv run python examples/inference/vlm/vlm_inference.py --hf_model_path="Qwen/Qwen2.5-VL-3B-Instruct" --image_path="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" --prompt="Describe this image."

  # Vision-Language generation with local image:
  uv run python examples/inference/vlm/vlm_inference.py --hf_model_path="Qwen/Qwen2.5-VL-3B-Instruct" --image_path="/path/to/image.jpg" --prompt="What do you see in this image?"

  # Text-only generation (no image):
  uv run python examples/inference/vlm/vlm_inference.py --hf_model_path="Qwen/Qwen2.5-VL-3B-Instruct" --prompt="Hello, how are you?"

  # Load from Megatron checkpoint:
  uv run python examples/inference/vlm/vlm_inference.py --hf_model_path="Qwen/Qwen2.5-VL-3B-Instruct" --megatron_model_path="/path/to/megatron/checkpoint" --image_path="/path/to/image.jpg" --prompt="Describe this image."
"""

import argparse
from typing import Optional

import requests
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.common_utils import get_last_rank, print_rank_0

from megatron.bridge.inference.vlm.base import generate, setup_inference_wrapper

from megatron.core.inference.common_inference_params import CommonInferenceParams

def process_image_inputs(processor, image_path: Optional[str], prompt: str):
    """Process image inputs for vision-language model.

    Args:
        processor: AutoProcessor for the VL model
        image_path: Path or URL to the image (optional)
        prompt: Text prompt

    Returns:
        Tuple of (input_ids, image_inputs, video_inputs)
    """
    if image_path:
        # Create messages with image and text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        return (
            text,
            image_inputs,
            video_inputs,
        )
    else:
        # Text-only processing
        return prompt, None, None


def main(args) -> None:
    """Main function for vision-language generation from HuggingFace VL models.

    Loads a VL model either from HuggingFace (with optional conversion to Megatron)
    or directly from a Megatron checkpoint, then performs greedy generation
    using the provided prompt and optional image input.

    Args:
        args: Parsed command line arguments containing model paths, prompt,
              image path, parallelism settings, and generation parameters
    """
    # pylint: disable=C0115,C0116
    tp = args.tp
    ep = args.ep
    etp = args.etp

    # Choose loading method based on arguments
    if args.megatron_model_path:
        # Load from Megatron checkpoint
        print_rank_0(f"Loading Megatron model from: {args.megatron_model_path}")

        # We still need HF config for tokenizer, but we'll load the model from Megatron checkpoint
        # Create bridge from HF config only (no weights)
        bridge = AutoBridge.from_hf_pretrained(args.hf_model_path)

        # Initialize model parallel before loading
        model_provider = bridge.to_megatron_provider(load_weights=False)
        model_provider.tensor_model_parallel_size = tp
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp
        model_provider.pipeline_dtype = torch.bfloat16
        model_provider.parallel_output = False
        model_provider.finalize()
        model_provider.initialize_model_parallel(seed=0)
        # Load the Megatron model directly
        model = bridge.load_megatron_model(
            args.megatron_model_path,
            mp_overrides={
                "tensor_model_parallel_size": tp,
                "expert_model_parallel_size": ep,
                "expert_tensor_parallel_size": etp,
                "pipeline_dtype": torch.bfloat16,
            },
            wrap_with_ddp=False,
        )

    else:
        # Load from HuggingFace and convert to Megatron
        print_rank_0(f"Loading HuggingFace model from: {args.hf_model_path}")
        bridge = AutoBridge.from_hf_pretrained(args.hf_model_path)
        model_provider = bridge.to_megatron_provider(load_weights=True)
        model_provider.tensor_model_parallel_size = tp
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp
        model_provider.pipeline_dtype = torch.bfloat16
        model_provider.parallel_output = False
        model_provider.finalize()
        model_provider.initialize_model_parallel(seed=0)
        model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    model = [m.cuda() for m in model]
    for m in model:
        m.eval()

    # Set grad_scale_func to None on the model's config for inference
    for m in model:
        if hasattr(m, "config"):
            m.config.grad_scale_func = None

    # Initialize tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_path,
        trust_remote_code=is_safe_repo(
            trust_remote_code=args.trust_remote_code,
            hf_path=args.hf_model_path,
        ),
    )
    processor = AutoProcessor.from_pretrained(
        args.hf_model_path,
        trust_remote_code=is_safe_repo(
            trust_remote_code=args.trust_remote_code,
            hf_path=args.hf_model_path,
        ),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Process inputs (text and image if provided)
    prompt = args.prompt
    text, image_inputs, video_inputs = process_image_inputs(
        processor, args.image_path, prompt
    )

    # Setup inference wrapper
    inference_wrapped_model = setup_inference_wrapper(
        model[0],
        tokenizer,
        params_dtype=torch.bfloat16,
        inference_batch_times_seqlen_threshold=1000,
    )

    # Setup inference parameters
    inference_params = CommonInferenceParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_tokens_to_generate=args.max_new_tokens,
    )

    # Generate text
    results = generate(
        wrapped_model=inference_wrapped_model,
        tokenizer=tokenizer,
        image_processor=processor.image_processor,
        prompts=[text],
        images=[image_inputs] if image_inputs is not None else None,
        processor=processor,
        max_batch_size=1,
        random_seed=0,
        inference_params=inference_params,
    )

    # Print results
    print_rank_0("======== GENERATED TEXT OUTPUT ========")
    print_rank_0(f"Prompt: {prompt}")
    print_rank_0(f"Generated: {results[0].text}")
    print_rank_0("=======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision-Language Generation from HuggingFace VL Models")
    parser.add_argument(
        "--hf_model_path",
        type=str,
        required=True,
        help="Path to the HuggingFace VL model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image.",
        help="Input prompt for vision-language generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.0,
        help="Top-p for sampling.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top-k for sampling.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")
    parser.add_argument("--megatron_model_path", type=str, default=None, help="Path to the Megatron model checkpoint")
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path or URL to the image for vision-language generation (optional).",
    )
    parser.add_argument("--trust_remote_code", action="store_true", help="if trust_remote_code")
    args = parser.parse_args()

    main(args)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
