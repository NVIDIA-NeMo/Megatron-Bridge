#!/usr/bin/env python3
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

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

from easydict import EasyDict


warnings.filterwarnings("ignore")

import random

import torch
import torch.distributed as dist

from megatron.bridge.diffusion.models.wan.flow_matching.flow_inference_pipeline import FlowInferencePipeline
from megatron.bridge.diffusion.models.wan.inference import SIZE_CONFIGS, SUPPORTED_SIZES
from megatron.bridge.diffusion.models.wan.inference.utils import cache_video, str2bool


EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
}


def _validate_args(args):
    # Basic check
    assert args.task in SUPPORTED_SIZES, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 50

    if args.sample_shift is None:
        args.sample_shift = 5.0

    # Frames default handled later; no single frame arg anymore

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    # Size check: only validate provided --sizes; default handled later
    if args.sizes is not None and len(args.sizes) > 0:
        for s in args.sizes:
            assert s in SUPPORTED_SIZES[args.task], (
                f"Unsupport size {s} for task {args.task}, supported sizes are: "
                f"{', '.join(SUPPORTED_SIZES[args.task])}"
            )


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate a image or video from a text prompt or image using Wan")
    parser.add_argument(
        "--task", type=str, default="t2v-14B", choices=list(SUPPORTED_SIZES.keys()), help="The task to run."
    )
    parser.add_argument(
        "--sizes",
        type=str,
        nargs="+",
        default=None,
        choices=list(SIZE_CONFIGS.keys()),
        help="A list of sizes to generate multiple images or videos (WIDTH*HEIGHT). Example: --sizes 1280*720 1920*1080",
    )
    parser.add_argument(
        "--frame_nums",
        type=int,
        nargs="+",
        default=None,
        help="List of frame counts (each should be 4n+1). Broadcasts if single value.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="The path to the main WAN checkpoint directory.",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=None,
        help=(
            "Optional training step to load, e.g. 1800 -> iter_0001800. "
            "If not provided, the latest (largest) step in --checkpoint_dir is used.",
        ),
    )
    parser.add_argument(
        "--t5_checkpoint_dir", type=str, default=None, help="Optional directory containing T5 checkpoint/tokenizer"
    )
    parser.add_argument(
        "--vae_checkpoint_dir", type=str, default=None, help="Optional directory containing VAE checkpoint"
    )
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.",
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.",
    )
    parser.add_argument(
        "--save_file", type=str, default=None, help="The file to save the generated image or video to."
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="A list of prompts to generate multiple images or videos. Example: --prompts 'a cat' 'a dog'",
    )
    parser.add_argument("--base_seed", type=int, default=-1, help="The seed to use for generating the image or video.")
    parser.add_argument("--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift", type=float, default=None, help="Sampling shift factor for flow matching schedulers."
    )
    parser.add_argument("--sample_guide_scale", type=float, default=5.0, help="Classifier free guidance scale.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--context_parallel_size", type=int, default=1, help="Context parallel size.")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1, help="Pipeline parallel size.")
    parser.add_argument("--sequence_parallel", type=str2bool, default=False, help="Sequence parallel.")

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):  # noqa: D103
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)
    videos = []

    if args.offload_model is None:
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    inference_cfg = EasyDict(
        {
            # t5
            "t5_dtype": torch.bfloat16,
            "text_len": 512,
            # vae
            "vae_stride": (4, 8, 8),
            # transformer
            "param_dtype": torch.bfloat16,
            "patch_size": (1, 2, 2),
            # others
            "num_train_timesteps": 1000,
            "sample_fps": 16,
            "chinese_sample_neg_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            "english_sample_neg_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        }
    )

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {inference_cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if "t2v" in args.task:
        # Resolve prompts list (default to example prompt)
        if args.prompts is not None and len(args.prompts) > 0:
            prompts = args.prompts
        else:
            prompts = [EXAMPLE_PROMPT[args.task]["prompt"]]

        # Resolve sizes list (default to first supported size for task)
        if args.sizes is not None and len(args.sizes) > 0:
            size_keys = args.sizes
        else:
            size_keys = [SUPPORTED_SIZES[args.task][0]]

        # Resolve frame counts list (default 81)
        if args.frame_nums is not None and len(args.frame_nums) > 0:
            frame_nums = args.frame_nums
        else:
            frame_nums = [81]

        # Enforce 1:1 pairing across lists
        assert len(prompts) == len(size_keys) == len(frame_nums), (
            f"prompts ({len(prompts)}), sizes ({len(size_keys)}), and frame_nums ({len(frame_nums)}) "
            f"must have the same length"
        )

        logging.info("Creating flow inference pipeline.")
        pipeline = FlowInferencePipeline(
            inference_cfg=inference_cfg,
            checkpoint_dir=args.checkpoint_dir,
            model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
            checkpoint_step=args.checkpoint_step,
            t5_checkpoint_dir=args.t5_checkpoint_dir,
            vae_checkpoint_dir=args.vae_checkpoint_dir,
            device_id=device,
            rank=rank,
            t5_cpu=args.t5_cpu,
            tensor_parallel_size=args.tensor_parallel_size,
            context_parallel_size=args.context_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            sequence_parallel=args.sequence_parallel,
            pipeline_dtype=torch.float32,
        )

        rank = dist.get_rank()
        if rank == 0:
            print("Running inference with tensor_parallel_size:", args.tensor_parallel_size)
            print("Running inference with context_parallel_size:", args.context_parallel_size)
            print("Running inference with pipeline_parallel_size:", args.pipeline_parallel_size)
            print("Running inference with sequence_parallel:", args.sequence_parallel)
            print("\n\n\n")

        logging.info("Generating videos ...")
        videos = pipeline.generate(
            prompts=prompts,
            sizes=[SIZE_CONFIGS[size] for size in size_keys],
            frame_nums=frame_nums,
            shift=args.sample_shift,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
        )

        if rank == 0:
            for i, video in enumerate(videos):
                formatted_experiment_name = (args.save_file) if args.save_file is not None else "DefaultExp"
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = prompts[i].replace(" ", "_").replace("/", "_")[:50]
                suffix = ".mp4"
                formatted_save_file = (
                    f"{args.task}_{formatted_experiment_name}_videoindex{int(i)}_size{size_keys[i].replace('*', 'x') if sys.platform == 'win32' else size_keys[i]}_{formatted_prompt}_{formatted_time}"
                    + suffix
                )

                logging.info(f"Saving generated video to {formatted_save_file}")
                cache_video(
                    tensor=video[None],
                    save_file=formatted_save_file,
                    fps=inference_cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
