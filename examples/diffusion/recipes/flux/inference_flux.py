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
FLUX Inference Script for text-to-image generation.

Runs the FLUX diffusion model to generate images from text prompts. Requires a Megatron-format
FLUX checkpoint (e.g. from convert_checkpoints.py import), a VAE checkpoint, and optionally
T5/CLIP text encoder IDs (downloaded from Hugging Face by default).

Examples:
    Single prompt, default resolution (1024x1024) and 10 steps:
        $ uv run python inference_flux.py --flux_ckpt /path/to/flux/iter_0000000 \\
            --vae_ckpt /path/to/vae --prompts "a dog holding a sign said hello world" \\
            --output_path ./flux_output

    Multiple prompts:
        $ uv run python inference_flux.py --flux_ckpt /path/to/flux/iter_0000000 \\
            --vae_ckpt /path/to/vae --prompts "prompt one" --prompts "prompt two" \\
            --output_path ./flux_output

    Custom resolution and inference steps:
        $ uv run python inference_flux.py --flux_ckpt /path/to/flux/iter_0000000 \\
            --vae_ckpt /path/to/vae --prompts "a cat on a mat" \\
            --height 512 --width 512 --num_inference_steps 20 --output_path ./flux_output

    From repository root:
        $ uv run python examples/diffusion/recipes/flux/inference_flux.py \\
            --flux_ckpt /aot/checkpoints/dfm/flux.1-dev/iter_0000000 \\
            --vae_ckpt /aot/checkpoints/flux.1-dev/vae \\
            --prompts "a dog holding a sign said hello world" --output_path ./flux_output
"""

import argparse
import os

import torch
import torch.distributed as dist


def parse_args():  # noqa: D103
    parser = argparse.ArgumentParser(description="FLUX inference")
    parser.add_argument("--flux_ckpt", type=str, required=True, help="Path to FLUX checkpoint")
    parser.add_argument("--vae_ckpt", type=str, default=None, help="Path to VAE")
    parser.add_argument("--t5_version", type=str, default="google/t5-v1_1-xxl")
    parser.add_argument("--clip_version", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument(
        "--prompts",
        type=str,
        action="append",
        help="Prompt(s) to generate images from. Can be specified multiple times for multiple prompts.",
        required=True,
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--output_path", type=str, default="/tmp/flux_output")
    parser.add_argument("--base_seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main():  # noqa: D103
    args = parse_args()

    # Initialize megatron parallel state
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]
    # Import FLUX

    from megatron.bridge.diffusion.models.flux import FluxInferencePipeline

    # Create pipeline
    pipeline = FluxInferencePipeline(
        flux_checkpoint_dir=args.flux_ckpt,
        t5_checkpoint_dir=args.t5_version,
        clip_checkpoint_dir=args.clip_version,
        vae_checkpoint_dir=args.vae_ckpt,
    )

    # Generate
    images = pipeline(
        prompt=args.prompts,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        output_path=args.output_path,
    )
    print(f"Generated {len(images)} images to {args.output_path}")


if __name__ == "__main__":
    main()
