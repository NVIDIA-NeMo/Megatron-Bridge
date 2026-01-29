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
  # Load from Megatron checkpoint, with image from URL:
  uv run python examples/inference/vlm/vlm_inference.py --megatron_model_path="/path/to/megatron/checkpoint" --image_path="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" --prompt="Describe this image."
"""

import argparse
from typing import Optional

import torch
from megatron.core.inference.common_inference_params import CommonInferenceParams
from qwen_vl_utils import process_vision_info

from megatron.bridge.inference.vlm.base import generate, setup_model_and_tokenizer
from megatron.bridge.utils.common_utils import print_rank_0


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

    # Setup model and processor
    inference_wrapped_model, processor = setup_model_and_tokenizer(
        megatron_model_path=args.megatron_model_path,
        tp=args.tp,
        pp=args.pp,
        inference_batch_times_seqlen_threshold=1000,
        inference_max_seq_length=args.max_seq_length,
    )

    # Process inputs (text and image if provided)
    prompt = args.prompt
    text, image_inputs, video_inputs = process_image_inputs(processor, args.image_path, prompt)

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
        tokenizer=processor.tokenizer,
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
    parser.add_argument("--megatron_model_path", type=str, default=None, help="Path to the Megatron model checkpoint")
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
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=8192,
        help="Maximum sequence length for inference (prompt + generated tokens).",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path or URL to the image for vision-language generation (optional).",
    )
    args = parser.parse_args()

    main(args)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
