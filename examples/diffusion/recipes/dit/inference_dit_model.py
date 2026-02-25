# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from einops import rearrange
from megatron.core import parallel_state as ps
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import T5EncoderModel, T5TokenizerFast

from megatron.bridge.diffusion.common.tokenizers.cosmos.cosmos1.causal_video_tokenizer import CausalVideoTokenizer
from megatron.bridge.diffusion.common.utils.save_video import save_video
from megatron.bridge.diffusion.models.dit.edm.edm_pipeline import EDMPipeline


EXAMPLE_PROMPT = (
    "The teal robot is cooking food in a kitchen. Steam rises from a simmering pot "
    "as the robot chops vegetables on a worn wooden cutting board. Copper pans hang "
    "from an overhead rack, catching glints of afternoon light, while a well-loved "
    "cast iron skillet sits on the stovetop next to scattered measuring spoons and "
    "a half-empty bottle of olive oil."
)


def parse_args():  # noqa: D103
    parser = argparse.ArgumentParser(description="Video foundation model inference")
    parser.add_argument(
        "--prompt",
        type=str,
        default=EXAMPLE_PROMPT,
        help="Prompt which the sampled video condition on",
    )
    # We turn on negative prompt by default. set to "" to turn it off.
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt which the sampled video condition on",
    )
    parser.add_argument("--subject_name", type=str, default="", help="Name of fine-tuned subject")
    parser.add_argument("--guidance", type=float, default=7, help="Classifier-free guidance scale")
    parser.add_argument("--sampler", type=str, default="RES", help="Currently only supports RES sampler.")
    parser.add_argument("--video_save_path", type=str, default="outputs", help="Path to save the video")
    parser.add_argument("--fps", type=int, default=24, help="FPS of the sampled video")
    parser.add_argument("--height", type=int, default=704, help="Height of image to sample")
    parser.add_argument("--width", type=int, default=1280, help="Width of image to sample")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--num_devices", type=int, default=1, help="Number of devices for inference")
    parser.add_argument("--cp_size", type=int, default=1, help="Number of cp ranks for multi-gpu inference.")
    parser.add_argument("--num_steps", type=float, default=35, help="Number of diffusion sampling steps")
    parser.add_argument("--num_video_frames", type=int, default=121, help="Number of video frames to sample")
    parser.add_argument(
        "--tokenizer_model", type=str, default="Cosmos-0.1-Tokenizer-CV4x8x8", help="Mode of video tokenizer"
    )
    parser.add_argument("--tokenizer_cache_dir", type=str, default=None, help="Directory for video tokenizer cache")
    parser.add_argument("--guardrail_dir", type=str, default="", help="Guardrails weights directory")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Video diffusion model checkpoint path")
    parser.add_argument("--t5_cache_dir", type=str, default=None, help="Path to T5 model")
    args = parser.parse_args()
    return args


def print_rank_0(string: str):  # noqa: D103
    rank = torch.distributed.get_rank()
    if rank == 0:
        print(string)


@torch.no_grad()
def encode_for_batch(tokenizer: T5TokenizerFast, encoder: T5EncoderModel, prompts: list[str], max_length: int = 512):
    """
    Encode a batch of text prompts to a batch of T5 embeddings.
    Parameters:
        tokenizer: T5 embedding tokenizer.
        encoder: T5 embedding text encoder.
        prompts: A batch of text prompts.
        max_length: Sequence length of text embedding (defaults to 512).
    """

    batch_encoding = tokenizer.batch_encode_plus(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_length=True,
        return_offsets_mapping=False,
    )

    # We expect all the processing is done on GPU.
    input_ids = batch_encoding.input_ids.cuda()
    attn_mask = batch_encoding.attention_mask.cuda()

    outputs = encoder(input_ids=input_ids, attention_mask=attn_mask)
    encoded_text = outputs.last_hidden_state

    lengths = attn_mask.sum(dim=1).cpu()
    for batch_id in range(encoded_text.shape[0]):
        encoded_text[batch_id][lengths[batch_id] :] = 0

    return encoded_text


class PosID3D:  # noqa: D101
    def __init__(self, *, max_t=32, max_h=128, max_w=128):
        self.max_t = max_t
        self.max_h = max_h
        self.max_w = max_w
        self.generate_pos_id()

    def generate_pos_id(self):
        self.grid = torch.stack(
            torch.meshgrid(
                torch.arange(self.max_t, device="cpu"),
                torch.arange(self.max_h, device="cpu"),
                torch.arange(self.max_w, device="cpu"),
            ),
            dim=-1,
        )

    def get_pos_id_3d(self, *, t, h, w):
        if t > self.max_t or h > self.max_h or w > self.max_w:
            self.max_t = max(self.max_t, t)
            self.max_h = max(self.max_h, h)
            self.max_w = max(self.max_w, w)
            self.generate_pos_id()
        return self.grid[:t, :h, :w]


def prepare_data_batch(args, tokenizer, text_encoder, t5_embeding_max_length=512):  # noqa: D103
    print("[args.prompt]: ", args.prompt)
    # Encode text to T5 embedding
    out = encode_for_batch(tokenizer, text_encoder, [args.prompt])
    encoded_text = torch.tensor(out, dtype=torch.bfloat16)
    B, L, C = encoded_text.shape
    t5_embed = torch.zeros(B, t5_embeding_max_length, C, dtype=torch.bfloat16)
    t5_embed[:, :L, :] = encoded_text
    t, h, w = args.num_video_frames, args.height, args.width
    pt, ph, pw = 1, 2, 2
    state_shape = [
        B,  # batch dimension
        ((h // 8) // ph)
        * ((w // 8) // pw)
        * t,  # number of tokens: (h //8) * (w // 8) * 1 -> ((h // 8) // ph) * ((w // 8) // pw) * 1
        16 * (ph * pw * pt),  # token hidden size (channel * patch_spatial * patch_spatial * patch_temporal)
    ]
    # prepare pos_emb
    pos_id_3d = PosID3D()
    pt, ph, pw = 1, 2, 2
    pos_ids = rearrange(
        # pos_id_3d.get_pos_id_3d(t=t // 4, h=h // 8, w=w // 8),
        pos_id_3d.get_pos_id_3d(t=t // pt, h=(h // 8) // ph, w=(w // 8) // pw),
        "T H W d -> (T H W) d",
    )
    data_batch = {
        "video": torch.zeros((B, 3, t, h, w), dtype=torch.uint8).cuda(),
        "context_embeddings": t5_embed,
        "context_mask": torch.ones(B, t5_embeding_max_length, dtype=torch.bfloat16).cuda(),
        "image_size": torch.tensor(
            [[args.height, args.width, args.height, args.width]] * B, dtype=torch.bfloat16
        ).cuda(),
        "fps": torch.tensor([[args.fps]] * B, dtype=torch.bfloat16).cuda(),
        "num_frames": torch.tensor([[args.num_video_frames]] * B, dtype=torch.bfloat16).cuda(),
        "padding_mask": torch.zeros((B, 1, args.height, args.width), dtype=torch.bfloat16).cuda(),
        "pos_ids": pos_ids.unsqueeze(0).expand(B, -1, -1),
        "latent_shape": [16, t // pt, h // 8 // ph, w // 8 // pw],
    }
    return data_batch, state_shape


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint iteration in a checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Path to the latest iter_* folder, or the original path if no iter folders found
    """
    checkpoint_path = Path(checkpoint_dir)
    iter_folders = [f for f in checkpoint_path.iterdir() if f.is_dir() and f.name.startswith("iter_")]

    if iter_folders:
        # Find the folder with the largest iteration number
        def get_iter_number(folder_name):
            try:
                return int(folder_name.replace("iter_", ""))
            except ValueError:
                return -1

        latest_iter = max(iter_folders, key=lambda f: get_iter_number(f.name))
        return checkpoint_path / latest_iter.name
    else:
        return checkpoint_path


def load_model_from_checkpoint(args):
    """
    Load DiT model from a Megatron checkpoint using Megatron-Bridge utilities.

    Args:
        args: Command line arguments containing checkpoint_path

    Returns:
        model: Loaded model
        diffusion_pipeline: EDM pipeline with loaded model
        model_config: Model configuration
    """
    from megatron.bridge.training.model_load_save import build_and_load_model, load_model_config

    checkpoint_path = find_latest_checkpoint(args.checkpoint_path)
    print_rank_0(f"Loading model from checkpoint: {checkpoint_path}")

    # Load the model configuration from checkpoint
    model_config, _ = load_model_config(str(checkpoint_path))

    # Override parallelism settings for inference if needed
    # Keep context parallel size from args if specified
    if hasattr(args, "cp_size") and args.cp_size:
        model_config.context_parallel_size = args.cp_size

    # Build and load the model with weights from checkpoint
    model = build_and_load_model(
        checkpoint_path=str(checkpoint_path),
        model_cfg=model_config,
        skip_temp_dist_context=True,  # Already initialized distributed
        use_cpu_init=False,
    )

    # If model is returned as a list, extract the first element
    if isinstance(model, list):
        model = model[0]

    model = model.cuda().to(torch.bfloat16).eval()

    print_rank_0(f"✅ Model loaded successfully from {checkpoint_path}")

    # Setup diffusion pipeline
    diffusion_pipeline = EDMPipeline(seed=args.seed)
    return model, diffusion_pipeline, model_config


def data_preprocess(data_batch, state_shape):  # noqa: D103
    from megatron.bridge.diffusion.models.dit.dit_data_process import encode_seq_length

    data_batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in data_batch.items()}
    data_batch["inference_fwd"] = True

    data_batch["seq_len_q"] = torch.tensor([state_shape[1]] * state_shape[0]).cuda()
    data_batch["seq_len_q_padded"] = torch.tensor([state_shape[1]] * state_shape[0]).cuda()
    data_batch["seq_len_kv"] = torch.tensor([data_batch["context_embeddings"].shape[1]] * state_shape[0]).cuda()
    data_batch["seq_len_kv_padded"] = torch.tensor([data_batch["context_embeddings"].shape[1]] * state_shape[0]).cuda()
    data_batch = encode_seq_length(data_batch, format="thd")
    return data_batch


def initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1):  # noqa: D103
    ps.destroy_model_parallel()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    ps.initialize_model_parallel(
        tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size=context_parallel_size
    )


def main(args):  # noqa: D103
    # Initialize distributed environment and model parallel groups
    initialize_distributed(1, 1, context_parallel_size=args.cp_size)
    model_parallel_cuda_manual_seed(args.seed)

    # Setup model / diffusion pipeline
    print_rank_0("setting up diffusion pipeline...")
    import random

    model_parallel_cuda_manual_seed(args.seed)

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # For deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(42)

    rank = torch.distributed.get_rank()
    if rank == 0:
        gather_list = [None for _ in range(ps.get_data_parallel_world_size())]
        wandb.init(project="dit-inference-video", name="inference_generation")
    else:
        gather_list = None

    # Load model from checkpoint or initialize from scratch
    print_rank_0("Loading model from checkpoint...")
    model, diffusion_pipeline, model_config = load_model_from_checkpoint(args)

    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-11b", cache_dir=args.t5_cache_dir, dtype=torch.bfloat16)
    text_encoder = T5EncoderModel.from_pretrained(
        "google-t5/t5-11b", cache_dir=args.t5_cache_dir, dtype=torch.bfloat16
    )
    text_encoder.to("cuda").eval()

    print_rank_0("preparing data batch...")
    data_batch, state_shape = prepare_data_batch(args, tokenizer, text_encoder)
    vae = CausalVideoTokenizer.from_pretrained(args.tokenizer_model, cache_dir=args.tokenizer_cache_dir)
    vae.to("cuda").eval()

    print_rank_0("generating video...")
    data_batch = data_preprocess(data_batch, state_shape)
    C, T, H, W = data_batch["latent_shape"]
    latent = diffusion_pipeline.generate_samples_from_batch(
        data_batch=data_batch,
        model=model,
        guidance=args.guidance,
        state_shape=state_shape,
        num_steps=args.num_steps,
        is_negative_prompt=True if "neg_t5_text_embeddings" in data_batch else False,
    )
    rank = torch.distributed.get_rank()
    latent = latent[0, None, : state_shape[1]]
    latent = rearrange(
        latent,
        "b (T H W) (ph pw pt c) -> b c (T pt) (H ph) (W pw)",
        ph=model_config.patch_spatial,
        pw=model_config.patch_spatial,
        pt=model_config.patch_temporal,
        c=C,
        T=T,
        H=H,
        W=W,
    )
    decoded_video = (1.0 + vae.decode(latent / model_config.sigma_data)).clamp(0, 2) / 2
    decoded_video = (decoded_video * 255).to(torch.uint8).permute(0, 2, 3, 4, 1).cpu().numpy()
    save_video(
        grid=decoded_video[0],
        fps=args.fps,
        H=args.height,
        W=args.width,
        video_save_quality=5,
        video_save_path=f"rank={rank}_" + args.video_save_path,
    )
    print_rank_0(f"saved video to rank={rank}_{args.video_save_path}")

    torch.distributed.gather_object(
        obj=(decoded_video[0], args.prompt),
        object_gather_list=gather_list,
        dst=0,
        group=ps.get_data_parallel_group(),
    )

    if rank == 0 and wandb.run is not None:
        videos = []
        for video, caption in gather_list:
            video_data_transposed = video.transpose(0, 3, 1, 2)
            videos.append(wandb.Video(video_data_transposed, fps=args.fps, format="mp4", caption=caption))
        wandb.log({"generated_videos": videos})
        wandb.finish()
        print_rank_0("✅ All videos gathered and logged to wandb with captions")


if __name__ == "__main__":
    args = parse_args()
    main(args)
