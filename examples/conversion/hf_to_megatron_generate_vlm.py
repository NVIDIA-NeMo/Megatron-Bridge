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
  python examples/conversion/hf_to_megatron_generate_vlm.py --hf_model_path="Qwen/Qwen2.5-VL-3B-Instruct" --image_path="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" --prompt="Describe this image."

  # Vision-Language generation with local image:
  python examples/conversion/hf_to_megatron_generate_vlm.py --hf_model_path="Qwen/Qwen2.5-VL-3B-Instruct" --image_path="/path/to/image.jpg" --prompt="What do you see in this image?"

  # Text-only generation (no image):
  python examples/conversion/hf_to_megatron_generate_vlm.py --hf_model_path="Qwen/Qwen2.5-VL-3B-Instruct" --prompt="Hello, how are you?"

  # Load from Megatron checkpoint:
  python examples/conversion/hf_to_megatron_generate_vlm.py --hf_model_path="Qwen/Qwen2.5-VL-3B-Instruct" --megatron_model_path="/path/to/megatron/checkpoint" --image_path="/path/to/image.jpg" --prompt="Describe this image."
"""

import argparse
from typing import Optional, List
import requests
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer

from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import adjust_image_tokens

# === NEW IMPORTS FOR MEGATRON PREPROCESSING ===
try:
    from megatron.training.tokenizer.multimodal_tokenizer import MultimodalTokenizer
    from examples.multimodal.image_processing import ImageTransform, find_closest_area_weighted_aspect_ratio, process_images
except Exception:
    # Megatron might not be available in lightweight envs – fail gracefully when the user
    # chooses the default HF preprocessing path.
    MultimodalTokenizer = None  # type: ignore

from megatron.bridge import AutoBridge
from megatron.bridge.utils.common_utils import get_last_rank, print_rank_0


class SingleBatchIterator:
    """Iterator that yields a single batch of data for text generation.
    Required by the forward_backward_func function.

    This class creates an iterator that yields exactly one batch containing
    input tokens, position IDs, attention mask, and optional vision inputs,
    then raises StopIteration. Used for single-step inference in the forward pass.
    """

    def __init__(self, input_ids, position_ids, attention_mask, **kwargs):
        self.batch = dict(
            tokens=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        # Add vision inputs if provided
        if kwargs.get("images", None) is not None:
            self.batch["images"] = kwargs.get("images", None)
        elif kwargs.get("pixel_values", None) is not None:
            self.batch["pixel_values"] = kwargs.get("pixel_values", None)
        if kwargs.get("image_grid_thw", None) is not None:
            self.batch["image_grid_thw"] = kwargs.get("image_grid_thw", None)

        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def vlm_forward_step(data_iterator, model, **kwargs) -> torch.Tensor:
    """Forward step function for vision-language generation.
    Required by the forward_backward_func function.

    Extracts a batch from the data iterator and runs the model forward pass
    with the provided input tokens, position IDs, attention mask, and vision inputs.

    Args:
        data_iterator: Iterator providing batches of input data
        model: The Megatron model to run forward pass on
        **kwargs: Additional keyword arguments (unused)

    Returns:
        Tuple of (model_output, loss_function)
    """
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }

    if "images" in batch:
        forward_args["images"] = batch["images"]
    elif "pixel_values" in batch:
        forward_args["pixel_values"] = batch["pixel_values"]
    if "image_grid_thw" in batch:
        forward_args["image_grid_thw"] = batch["image_grid_thw"]

    def loss_func(x, **kwargs):
        return x

    return model(**forward_args), loss_func


def load_image(image_path: str) -> Image.Image:
    """Load an image from URL or file path.

    Args:
        image_path: URL or local file path to the image

    Returns:
        PIL Image object
    """
    if image_path.startswith(("http://", "https://")):
        response = requests.get(image_path)
        response.raise_for_status()
        return Image.open(requests.get(image_path, stream=True).raw)
    else:
        return Image.open(image_path)


def process_image_inputs(processor, image_path: Optional[str], prompt: str):
    """Process image inputs for vision-language model.

    Args:
        processor: AutoProcessor for the VL model
        image_path: Path or URL to the image (optional)
        prompt: Text prompt

    Returns:
        Tuple of (input_ids, pixel_values, image_grid_thw, messages)
    """
    if image_path:
        if "," in image_path:
            image_paths = image_path.split(",")
            content = []
            for i, path in enumerate(image_paths):
                content.append({"type": "text", "text": f"{'\n' if i > 0 else ''}Image-{i+1}: "})
                content.append({"type": "image", "image": path})
            content.append({"type": "text", "text": '\n' + prompt})
        else:
            content = [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ]
        # Create messages with image and text
        messages = [
            {
                "role": "system",
                "content": "/no_think",
            },
            {
                "role": "user",
                "content": content,
            }
        ]

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Process inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=processor.tokenizer.pad_token is not None,
            return_tensors="pt",
        )
        try: 
            image_grid_thw = inputs.image_grid_thw
        except AttributeError:
            image_grid_thw = None
        return inputs.input_ids, inputs.pixel_values, image_grid_thw, inputs.num_patches
    else:
        # Text-only processing
        inputs = processor(text=[prompt], return_tensors="pt")
        return inputs.input_ids, None, None, None

def process_video_inputs(processor, video_path: Optional[str], prompt: str):
    """Process video inputs for vision-language model.
    """
    from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import maybe_path_or_url_to_data_urls, pil_image_from_base64

    video_fps = -1
    video_nframe = 10
    video_nframe_max = -1

    # Get frames and metadata
    image_urls, metadata = maybe_path_or_url_to_data_urls(
        video_path,
        fps=max(0, int(video_fps)),
        nframe=max(0, int(video_nframe)),
        nframe_max=int(video_nframe_max),
    )
    frames = [pil_image_from_base64(image_url) for image_url in image_urls]

    print(f"Video Metadata: {metadata}")

    messages = [
        {
            "role": "system",
            "content": "/no_think"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_path}",
                },
                {
                    "type": "text",
                    "text": "\n" + prompt,
                },
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process with FPS metadata
    if metadata:
        inputs = processor(
            text=[prompt],
            videos=frames,
            videos_kwargs={'video_metadata': metadata},
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=[prompt],
            videos=frames,
            return_tensors="pt",
        )
    return inputs.input_ids, inputs.pixel_values_videos, None, inputs.num_patches

def process_inputs_megatron(image_path: Optional[str], prompt: str, hf_model_path: str):
    """Pre-process using Megatron Energon TaskEncoder utilities.

    Returns (input_ids, images, image_grid_thw, messages) matching the format
    expected by `vlm_forward_step`.
    """
    if MultimodalTokenizer is None:
        raise RuntimeError("Megatron libraries not available – cannot use preprocess_mode='megatron'.")

    # Build an underlying HuggingFace tokenizer first, then wrap with MultimodalTokenizer.
    base_tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    tokenizer = MultimodalTokenizer(
        tokenizer=base_tokenizer,
        prompt_format="nemotron-h-5p5-reasoning",  # adjust if needed
        special_tokens=[],  # special tokens are already added
        image_tag_type="internvl",
        force_system_message=False,
    )

    # Basic system prompt – keep simple.
    system_prompt = "/no_think"

    messages = []
    if image_path:
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": "<image>" + prompt})
    else:
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

    # Tokenize conversation (wrapper expects list[dict])
    input_ids = tokenizer.tokenize_conversation(messages, return_target=False, add_generation_prompt=True)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    images = None
    if image_path:
        pil_img = load_image(image_path).convert("RGB")
        # Image preprocessing aligned with Megatron training (encode_llava_pretrain)
        img_h = 512  # default tile size used during training; adjust via CLI if necessary
        img_w = 512
        vision_model_type = "radio"  # change if your model uses a different vision backbone
        patch_dim = 16
        transform_img = ImageTransform(
            img_h,
            vision_model_type,
            dynamic_resolution=False,
            res_step=patch_dim,
            max_num_patches=img_h * img_w // (patch_dim * patch_dim),
            pixel_shuffle=True,
        )

        img_tensors = transform_img(
            pil_img,
            img_h,
            img_w,
            use_tiling=True,
            max_num_tiles=12,
            use_thumbnail=True,
            augment=False,
            find_closest_aspect_ratio_fn=find_closest_area_weighted_aspect_ratio,
        )

        images, _, _, _ = process_images(img_tensors, patch_dim=patch_dim, dynamic_resolution=False, batch_mode=False)
        images = images.bfloat16()  # [num_tiles, C, H, W]

        # ------------------------------------------------------------------
        # Ensure the text stream contains one <image> token per tile.
        # Megatron's training pipeline can optionally work with a single token
        # and expand it internally, but some checkpoints expect the count to
        # match exactly.  We replicate the IMAGE token here if needed.
        # ------------------------------------------------------------------
        num_tiles = images.shape[0]
        img_start_token_id = tokenizer.convert_tokens_to_ids("<img>")
        img_end_token_id = tokenizer.convert_tokens_to_ids("</img>")
        input_ids = adjust_image_tokens(input_ids, num_tiles, img_start_token_id, img_end_token_id)

    return input_ids, images, messages, tokenizer._tokenizer


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
    pp = args.pp
    ep = args.ep
    etp = args.etp

    # Choose loading method based on arguments
    if args.megatron_model_path:
        # Load from Megatron checkpoint
        print_rank_0(f"Loading Megatron model from: {args.megatron_model_path}")

        # We still need HF config for tokenizer, but we'll load the model from Megatron checkpoint
        # Create bridge from HF config only (no weights)
        bridge = AutoBridge.from_hf_pretrained(args.hf_model_path, trust_remote_code=True)

        # Initialize model parallel before loading
        model_provider = bridge.to_megatron_provider(load_weights=False)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp
        model_provider.pipeline_dtype = torch.bfloat16
        model_provider.initialize_model_parallel(seed=0)

        # Load the Megatron model directly
        model = bridge.load_megatron_model(
            args.megatron_model_path,
            mp_overrides={
                "tensor_model_parallel_size": tp,
                "pipeline_model_parallel_size": pp,
                "expert_model_parallel_size": ep,
                "expert_tensor_parallel_size": etp,
            },
            wrap_with_ddp=False,
        )

    else:
        # Load from HuggingFace and convert to Megatron
        print_rank_0(f"Loading HuggingFace model from: {args.hf_model_path}")
        bridge = AutoBridge.from_hf_pretrained(args.hf_model_path, trust_remote_code=True)
        model_provider = bridge.to_megatron_provider(load_weights=True)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp
        model_provider.pipeline_dtype = torch.bfloat16
        model_provider.initialize_model_parallel(seed=0)
        model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    model = [m.cuda() for m in model]
    for m in model:
        m.eval()

    # Initialize tokenizer and processor
    if args.preprocess_mode == "megatron":
        input_ids, images, _, tokenizer = process_inputs_megatron(args.image_path, args.prompt, args.hf_model_path)
        pixel_values = None
        image_grid_thw = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(args.hf_model_path, trust_remote_code=True)
        img_start_token_id = tokenizer.convert_tokens_to_ids("<img>")
        img_end_token_id = tokenizer.convert_tokens_to_ids("</img>")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if args.video_path:
            input_ids, pixel_values, image_grid_thw, num_patches = process_video_inputs(processor, args.video_path, args.prompt)
        else:
            input_ids, pixel_values, image_grid_thw, num_patches = process_image_inputs(processor, args.image_path, args.prompt)

        images = None
        if args.use_llava_model:
            images = pixel_values.bfloat16()
            input_ids = adjust_image_tokens(input_ids, num_patches, img_start_token_id,
                                             img_end_token_id)
            if args.video_path:
                video_token_id = tokenizer.convert_tokens_to_ids("<video>")
                image_token_id = tokenizer.convert_tokens_to_ids("<image>")
                input_ids = torch.where(input_ids == video_token_id, image_token_id, input_ids)
            pixel_values = None


    # Move to GPU
    input_ids = input_ids.cuda()
    if pixel_values is not None:
        pixel_values = pixel_values.cuda()
    if images is not None:
        images = images.cuda()

    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    generated_ids = input_ids.clone()

    stop_tokens = [tokenizer.eos_token_id]

    # Greedy generation loop
    for step in range(args.max_new_tokens):
        with torch.no_grad():
            print_rank_0(f"Generation step {step}")

            fwd_bwd_function = get_forward_backward_func()
            # Keep passing vision inputs for all steps to ensure image features are available
            # The Megatron VL model only processes vision features when pixel_values is not None,
            # so we need to provide them throughout the generation process
            iterator = SingleBatchIterator(
                input_ids, 
                position_ids, 
                attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                images=images,
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
                if isinstance(output, tuple):
                    # for LlavaModel
                    output = output[0]

            if parallel_state.is_pipeline_last_stage():
                world_size = parallel_state.get_tensor_model_parallel_world_size()
                gathered_tensors = [torch.zeros_like(output) for _ in range(world_size)]
                # All-gather operation
                dist.all_gather(gathered_tensors, output, group=parallel_state.get_tensor_model_parallel_group())
                # Concatenate along last dimension (dim=2)
                output = torch.cat(gathered_tensors, dim=2)
                next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)

                # Debug: print token information
                if step < 5:  # Only for first few iterations
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

            input_ids = generated_ids
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

            # If the generated token is the end of sequence token, stop generating
            if next_token_ids.item() in stop_tokens:
                break

    # Decode the generated sequence
    generated_text = tokenizer.decode(list(generated_ids[0]))
    print_rank_0("======== GENERATED TEXT OUTPUT ========")
    if args.image_path:
        print_rank_0(f"Image: {args.image_path}")
    print_rank_0(f"Prompt: {args.prompt}")
    print_rank_0(f"Generated: {generated_text}")
    print_rank_0("=======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision-Language Generation from HuggingFace VL Models")
    parser.add_argument(
        "--hf_model_path",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Path to the HuggingFace VL model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image.",
        help="Input prompt for vision-language generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")
    parser.add_argument("--megatron_model_path", type=str, default=None, help="Path to the Megatron model checkpoint")
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path or URL to the image for vision-language generation (optional). Multiple images paths can be separated"
             "with commas.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path or URL to the video for vision-language generation (optional).",
    )
    parser.add_argument(
        "--preprocess_mode",
        type=str,
        choices=["hf", "megatron"],
        default="hf",
        help="Choose preprocessing pipeline: 'hf' (default) uses HuggingFace AutoProcessor/Tokenizer, 'megatron' uses "
        "MultimodalTokenizer and image utilities from Megatron Energon TaskEncoder.",
    )
    parser.add_argument(
        "--use_llava_model",
        action="store_true",
        default=False,
        help="Specify whether model uses Megatron vision model (i.e. LLaVAModel)",
    )
    args = parser.parse_args()

    main(args)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
