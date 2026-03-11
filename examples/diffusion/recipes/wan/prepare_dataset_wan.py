# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import html
import json
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
import webdataset as wds
from diffusers import AutoencoderKLWan
from diffusers.utils import is_ftfy_available
from transformers import AutoTokenizer, UMT5EncoderModel


if is_ftfy_available():
    import ftfy


def basic_clean(text):
    """Fix text encoding issues and unescape HTML entities."""
    if is_ftfy_available():
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    """Normalize whitespace by replacing multiple spaces with single space."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    """Clean prompt text exactly as done in WanPipeline."""
    text = whitespace_clean(basic_clean(text))
    return text


def _map_interpolation(resize_mode: str) -> int:
    interpolation_map = {
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    if resize_mode not in interpolation_map:
        raise ValueError(f"Invalid resize_mode '{resize_mode}'. Choose from: {list(interpolation_map.keys())}")
    return interpolation_map[resize_mode]


def _calculate_resize_dimensions(
    original_height: int,
    original_width: int,
    target_size: Optional[Tuple[int, int]],
    maintain_aspect_ratio: bool,
    center_crop: bool = False,
) -> Tuple[int, int]:
    if target_size is None:
        return original_height, original_width

    target_height, target_width = target_size
    if not maintain_aspect_ratio:
        return target_height, target_width

    original_aspect = original_width / max(1, original_height)
    target_aspect = target_width / max(1, target_height)

    if center_crop:
        # Resize so the smaller dimension matches target, then crop
        if original_aspect > target_aspect:
            # Original is wider, match height
            resize_height = target_height
            resize_width = int(target_height * original_aspect)
        else:
            # Original is taller, match width
            resize_width = target_width
            resize_height = int(target_width / original_aspect)
        return resize_height, resize_width
    else:
        # Resize so the larger dimension matches target (fit inside)
        if original_aspect > target_aspect:
            # Original is wider, match width
            resize_width = target_width
            resize_height = int(target_width / original_aspect)
        else:
            # Original is taller, match height
            resize_height = target_height
            resize_width = int(target_height * original_aspect)
        return resize_height, resize_width


def _resize_frame(
    frame: np.ndarray,
    target_size: Optional[Tuple[int, int]],
    resize_mode: str,
    maintain_aspect_ratio: bool,
    center_crop: bool,
) -> np.ndarray:
    if target_size is None:
        return frame

    original_height, original_width = frame.shape[:2]
    resize_height, resize_width = _calculate_resize_dimensions(
        original_height, original_width, target_size, maintain_aspect_ratio, center_crop
    )

    interpolation = _map_interpolation(resize_mode)
    resized_frame = cv2.resize(frame, (resize_width, resize_height), interpolation=interpolation)

    if maintain_aspect_ratio and center_crop:
        target_height, target_width = target_size

        # Calculate crop coordinates
        if resized_frame.shape[0] > target_height or resized_frame.shape[1] > target_width:
            y_start = max(0, (resized_frame.shape[0] - target_height) // 2)
            x_start = max(0, (resized_frame.shape[1] - target_width) // 2)
            y_end = y_start + target_height
            x_end = x_start + target_width
            resized_frame = resized_frame[y_start:y_end, x_start:x_end]

        # Pad if necessary
        if resized_frame.shape[0] < target_height or resized_frame.shape[1] < target_width:
            pad_height = max(0, target_height - resized_frame.shape[0])
            pad_width = max(0, target_width - resized_frame.shape[1])
            resized_frame = np.pad(
                resized_frame, ((0, pad_height), (0, pad_width), (0, 0)), mode="constant", constant_values=0
            )

    return resized_frame


def _read_sidecar_caption(json_path: Path) -> str:
    """Read caption from a JSON sidecar file."""
    if not json_path.exists():
        raise FileNotFoundError(f"Sidecar JSON not found: {json_path}")

    with open(json_path, "r") as f:
        # Strict JSON loading - fail if invalid JSON
        obj = json.load(f)

        if "caption" in obj and isinstance(obj["caption"], str):
            return obj["caption"]

    raise ValueError(f"No valid 'caption' field found in {json_path}")


def _get_video_info(video_path: str) -> Tuple[int, int, int]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return max(0, total), width, height


def _load_metadata(video_folder: Path) -> List[Dict]:
    # Always scan for .mp4 files with sidecar .json; use full frame range
    items: List[Dict] = []
    for entry in sorted(video_folder.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() != ".mp4":
            continue
        video_name = entry.name
        video_path = str(entry)
        total_frames, width, height = _get_video_info(video_path)
        start_frame = 0
        end_frame = max(0, total_frames - 1)
        sidecar_json = entry.with_suffix(".json")
        caption = _read_sidecar_caption(sidecar_json)
        items.append(
            {
                "file_name": video_name,
                "width": width,
                "height": height,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "caption": caption,
            }
        )
    if not items:
        raise FileNotFoundError(f"No .mp4 files found in {video_folder}")
    return items


def _extract_first_frame(
    video_path: str,
    start_frame: int,
    target_size: Optional[Tuple[int, int]],
    resize_mode: str,
    maintain_aspect_ratio: bool,
    center_crop: bool,
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Could not read frame {start_frame} from {video_path}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = _resize_frame(frame, target_size, resize_mode, maintain_aspect_ratio, center_crop)
    return frame


def _load_frames_cv2(
    video_path: str,
    start_frame: int,
    end_frame: int,
    target_size: Optional[Tuple[int, int]],
    resize_mode: str,
    maintain_aspect_ratio: bool,
    center_crop: bool,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = _resize_frame(frame, target_size, resize_mode, maintain_aspect_ratio, center_crop)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames loaded from {video_path}")

    video_array = np.array(frames)  # T, H, W, C in [0,1]
    video_tensor = torch.from_numpy(video_array)  # T, H, W, C
    video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # 1, C, T, H, W
    video_tensor = video_tensor.to(dtype=target_dtype)
    return video_tensor


def _extract_evenly_spaced_frames(
    video_path: str,
    start_frame: int,
    end_frame: int,
    num_frames: int,
    target_size: Optional[Tuple[int, int]],
    resize_mode: str,
    maintain_aspect_ratio: bool,
    center_crop: bool,
) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    total_frames = max(0, end_frame - start_frame + 1)
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Invalid frame range [{start_frame}, {end_frame}] for {video_path}")

    if num_frames <= 1:
        frame_indices = [start_frame]
    else:
        frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int).tolist()

    extracted_frames: List[np.ndarray] = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = _resize_frame(frame, target_size, resize_mode, maintain_aspect_ratio, center_crop)
        extracted_frames.append(frame)

    cap.release()
    if not extracted_frames:
        raise ValueError(f"Could not extract any frames from {video_path}")
    return extracted_frames


def _frame_to_video_tensor(frame: np.ndarray, target_dtype: torch.dtype) -> torch.Tensor:
    # frame: RGB numpy array (H, W, C), uint8 or float
    if frame.dtype == np.uint8:
        frame = frame.astype(np.float32) / 255.0
    else:
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)
        if frame.max() > 1.0:
            frame = frame / 255.0

    tensor = torch.from_numpy(frame)  # H, W, C
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # 1, C, 1, H, W
    tensor = tensor.to(dtype=target_dtype)
    return tensor


@torch.no_grad()
def _init_hf_models(
    model_id: str,
    device: str,
    enable_memory_optimization: bool,
):
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=dtype,
    )
    text_encoder.to(device)
    text_encoder.eval()

    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=dtype,
    )
    vae.to(device)
    vae.eval()
    if enable_memory_optimization:
        vae.enable_slicing()
        vae.enable_tiling()

    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    return vae, text_encoder, tokenizer, dtype


@torch.no_grad()
def _encode_text(
    tokenizer: AutoTokenizer,
    text_encoder: UMT5EncoderModel,
    device: str,
    caption: str,
    max_sequence_length: int = 226,
) -> torch.Tensor:
    caption = prompt_clean(caption)
    inputs = tokenizer(
        caption,
        max_length=max_sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Calculate actual sequence length (excluding padding)
    seq_lens = inputs["attention_mask"].gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    ).last_hidden_state

    # CRITICAL: Trim to actual length and re-pad with zeros to match WanPipeline/preprocess_resize.py
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds],
        dim=0,
    )

    return prompt_embeds


@torch.no_grad()
def _encode_video_latents(
    vae: AutoencoderKLWan,
    device: str,
    video_tensor: torch.Tensor,
    deterministic_latents: bool,
) -> torch.Tensor:
    video_tensor = video_tensor.to(device=device, dtype=vae.dtype)
    video_tensor = video_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]

    latent_dist = vae.encode(video_tensor)
    if deterministic_latents:
        video_latents = latent_dist.latent_dist.mean
    else:
        video_latents = latent_dist.latent_dist.sample()

    if not hasattr(vae.config, "latents_mean") or not hasattr(vae.config, "latents_std"):
        raise ValueError("Wan2.1 VAE requires latents_mean and latents_std in config")

    latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=vae.dtype).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=device, dtype=vae.dtype).view(1, -1, 1, 1, 1)

    final_latents = (video_latents - latents_mean) / latents_std

    return final_latents


def main():
    parser = argparse.ArgumentParser(
        description="Prepare WAN WebDataset shards using HF automodel encoders and resizing"
    )
    parser.add_argument("--video_folder", type=str, required=True, help="Folder containing videos and meta.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write webdataset shards")
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        help="Wan2.1 model ID (e.g., Wan-AI/Wan2.1-T2V-14B-Diffusers or Wan-AI/Wan2.1-T2V-1.3B-Diffusers)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic encoding (sampling) instead of deterministic posterior mean",
    )
    parser.add_argument("--no-memory-optimization", action="store_true", help="Disable VAE slicing/tiling")
    parser.add_argument("--shard_maxcount", type=int, default=10000, help="Max samples per shard")
    parser.add_argument(
        "--mode",
        default="video",
        choices=["video", "frames"],
        help="Processing mode: 'video' for full videos, 'frames' to extract frames and treat each as a 1-frame video",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="Number of evenly-spaced frames to extract per video when using --mode frames",
    )

    # Resize arguments (match automodel)
    parser.add_argument("--height", type=int, default=None, help="Target height for video frames")
    parser.add_argument("--width", type=int, default=None, help="Target width for video frames")
    parser.add_argument(
        "--resize_mode",
        default="bilinear",
        choices=["bilinear", "bicubic", "nearest", "area", "lanczos"],
        help="Interpolation mode for resizing",
    )
    parser.add_argument("--no-aspect-ratio", action="store_true", help="Disable aspect ratio preservation")
    parser.add_argument("--center-crop", action="store_true", help="Center crop to exact target size after resize")
    parser.add_argument(
        "--output_format",
        default="energon",
        choices=["energon", "automodel"],
        help="Output format: 'energon' (WebDataset shards) or 'automodel' (individual .meta pickle files)",
    )

    args = parser.parse_args()

    # Initialize distributed
    rank = 0
    world_size = 1
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            args.device = f"cuda:{local_rank}"

    video_folder = Path(args.video_folder)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    if world_size > 1:
        dist.barrier()

    shard_pattern = str(output_dir / f"shard-rank{rank:02d}-%06d.tar")

    # Target size
    target_size = None
    if args.height is not None and args.width is not None:
        target_size = (args.height, args.width)
    elif (args.height is None) ^ (args.width is None):
        parser.error("Both --height and --width must be specified together")

    # Init HF models
    vae, text_encoder, tokenizer, model_dtype = _init_hf_models(
        model_id=args.model,
        device=args.device,
        enable_memory_optimization=not args.no_memory_optimization,
    )

    # Load metadata list
    metadata_list = _load_metadata(video_folder)
    metadata_list = metadata_list[rank::world_size]

    if args.output_format == "energon":
        context = wds.ShardWriter(shard_pattern, maxcount=args.shard_maxcount)
    else:
        from contextlib import nullcontext

        context = nullcontext()

    with context as sink:
        written = 0
        for local_index, meta in enumerate(metadata_list):
            global_index = local_index * world_size + rank
            video_name = meta["file_name"]
            start_frame = int(meta["start_frame"])  # inclusive
            end_frame = int(meta["end_frame"])  # inclusive
            caption_text = meta.get("caption", "")

            video_path = str(video_folder / video_name)
            if args.mode == "video":
                # Load frames using the same OpenCV + resize path as automodel
                video_tensor = _load_frames_cv2(
                    video_path=video_path,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    target_size=target_size,
                    resize_mode=args.resize_mode,
                    maintain_aspect_ratio=not args.no_aspect_ratio,
                    center_crop=args.center_crop,
                    target_dtype=model_dtype,
                )

                # Encode text and video with HF models exactly like automodel
                text_embed = _encode_text(tokenizer, text_encoder, args.device, caption_text)
                latents = _encode_video_latents(
                    vae, args.device, video_tensor, deterministic_latents=not args.stochastic
                )

                # Move to CPU without changing dtype; keep exact values to match automodel outputs
                text_embed_cpu = text_embed.detach().to(device="cpu")
                latents_cpu = latents.detach().to(device="cpu")

                # Reshape to match Mcore's Wan input format
                text_embed_cpu = text_embed_cpu[0]
                latents_cpu = latents_cpu[0]

                # Get dimensions from video tensor (1, C, T, H, W)
                _, T, H, W = video_tensor.shape[1:]

                if args.output_format == "automodel":
                    # Extract first frame
                    first_frame_numpy = _extract_first_frame(
                        video_path=video_path,
                        start_frame=start_frame,
                        target_size=target_size,
                        resize_mode=args.resize_mode,
                        maintain_aspect_ratio=not args.no_aspect_ratio,
                        center_crop=args.center_crop,
                    )

                    text_embed_cpu = text_embed_cpu.unsqueeze(0)
                    latents_cpu = latents_cpu.unsqueeze(0)

                    processed_data = {
                        "text_embeddings": text_embed_cpu,
                        "video_latents": latents_cpu,
                        "first_frame": first_frame_numpy,
                        "metadata": meta,
                        "num_frames": int(T),
                        "original_filename": video_name,
                        "original_video_path": video_path,
                        "deterministic_latents": bool(not args.stochastic),
                        "memory_optimization": bool(not args.no_memory_optimization),
                        "model_version": "wan2.1",
                        "processing_mode": "video",
                        "resize_settings": {
                            "target_size": target_size,
                            "resize_mode": args.resize_mode,
                            "maintain_aspect_ratio": bool(not args.no_aspect_ratio),
                            "center_crop": bool(args.center_crop),
                        },
                    }
                    out_path = output_dir / f"{Path(video_name).stem}.meta"
                    with open(out_path, "wb") as f:
                        pickle.dump(processed_data, f)
                    written += 1
                elif args.output_format == "energon":
                    # Build JSON side-info similar to prepare_energon script
                    json_data = {
                        "video_path": video_path,
                        "processed_frames": int(T),
                        "processed_height": int(H),
                        "processed_width": int(W),
                        "caption": caption_text,
                        "deterministic_latents": bool(not args.stochastic),
                        "memory_optimization": bool(not args.no_memory_optimization),
                        "model_version": "wan2.1",
                        "processing_mode": "video",
                        "resize_settings": {
                            "target_size": target_size,
                            "resize_mode": args.resize_mode,
                            "maintain_aspect_ratio": bool(not args.no_aspect_ratio),
                            "center_crop": bool(args.center_crop),
                        },
                    }

                    sample = {
                        "__key__": f"{global_index:06}",
                        "pth": latents_cpu,
                        "pickle": pickle.dumps(text_embed_cpu),
                        "json": json_data,
                    }
                    sink.write(sample)
                    written += 1
                else:
                    raise ValueError(f"Invalid output format: {args.output_format}")
            else:
                # Frames mode: extract evenly-spaced frames, treat each as a 1-frame video
                frames = _extract_evenly_spaced_frames(
                    video_path=video_path,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    num_frames=max(1, int(args.num_frames)),
                    target_size=target_size,
                    resize_mode=args.resize_mode,
                    maintain_aspect_ratio=not args.no_aspect_ratio,
                    center_crop=args.center_crop,
                )

                # Encode text once and reuse for all frames of this video
                text_embed = _encode_text(tokenizer, text_encoder, args.device, caption_text)
                text_embed_cpu = text_embed.detach().to(device="cpu")[0]

                total_extracted = len(frames)
                for frame_idx, frame in enumerate(frames, start=1):
                    video_tensor = _frame_to_video_tensor(frame, target_dtype=model_dtype)
                    latents = _encode_video_latents(
                        vae, args.device, video_tensor, deterministic_latents=not args.stochastic
                    )
                    latents_cpu = latents.detach().to(device="cpu")[0]

                    # Frame shape after resize
                    H, W = frame.shape[:2]

                    if args.output_format == "automodel":
                        text_embed_cpu_unsqueezed = text_embed_cpu.unsqueeze(0)
                        latents_cpu_unsqueezed = latents_cpu.unsqueeze(0)

                        processed_data = {
                            "text_embeddings": text_embed_cpu_unsqueezed,
                            "video_latents": latents_cpu_unsqueezed,
                            # no "first_frame" in --mode frames
                            "metadata": meta,
                            "frame_index": int(frame_idx),
                            "total_frames_in_video": int(total_extracted),
                            "num_frames": 1,
                            "original_filename": video_name,
                            "original_video_path": video_path,
                            "deterministic_latents": bool(not args.stochastic),
                            "memory_optimization": bool(not args.no_memory_optimization),
                            "model_version": "wan2.1",
                            "processing_mode": "frames",
                            "resize_settings": {
                                "target_size": target_size,
                                "resize_mode": args.resize_mode,
                                "maintain_aspect_ratio": bool(not args.no_aspect_ratio),
                                "center_crop": bool(args.center_crop),
                            },
                        }
                        out_path = output_dir / f"{Path(video_name).stem}_{frame_idx}.meta"
                        with open(out_path, "wb") as f:
                            pickle.dump(processed_data, f)
                        written += 1
                    elif args.output_format == "energon":
                        json_data = {
                            "video_path": video_path,
                            "processed_frames": 1,
                            "processed_height": int(H),
                            "processed_width": int(W),
                            "caption": caption_text,
                            "deterministic_latents": bool(not args.stochastic),
                            "memory_optimization": bool(not args.no_memory_optimization),
                            "model_version": "wan2.1",
                            "processing_mode": "frames",
                            "frame_index": int(frame_idx),
                            "total_frames_in_video": int(total_extracted),
                            "resize_settings": {
                                "target_size": target_size,
                                "resize_mode": args.resize_mode,
                                "maintain_aspect_ratio": bool(not args.no_aspect_ratio),
                                "center_crop": bool(args.center_crop),
                            },
                        }

                        sample = {
                            "__key__": f"{global_index:06}_{frame_idx:02}",
                            "pth": latents_cpu,
                            "pickle": pickle.dumps(text_embed_cpu),
                            "json": json_data,
                        }
                        sink.write(sample)
                        written += 1
                    else:
                        raise ValueError(f"Invalid output format: {args.output_format}")

    print("Done writing shards using HF automodel encoders.")


if __name__ == "__main__":
    main()