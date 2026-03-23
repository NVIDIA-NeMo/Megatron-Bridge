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

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
import webdataset as wds
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast


def _map_interpolation(resize_mode: str) -> int:
    """Map resize mode string to OpenCV interpolation constant."""
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
    """Calculate target dimensions for resizing."""
    if target_size is None:
        return original_height, original_width

    target_height, target_width = target_size
    if not maintain_aspect_ratio:
        return target_height, target_width

    original_aspect = original_width / max(1, original_height)
    target_aspect = target_width / max(1, target_height)

    if center_crop:
        # For center crop: resize so BOTH dimensions are >= target (resize on shorter edge)
        # This ensures we can crop to exact size without padding
        if original_aspect > target_aspect:
            # Image is wider: match height, width will be larger
            new_height = target_height
            new_width = int(round(target_height * original_aspect))
        else:
            # Image is taller: match width, height will be larger
            new_width = target_width
            new_height = int(round(target_width / max(1e-6, original_aspect)))
    else:
        # For no center crop: resize so image fits within target (resize on longer edge)
        if original_aspect > target_aspect:
            new_width = target_width
            new_height = int(round(target_width / max(1e-6, original_aspect)))
        else:
            new_height = target_height
            new_width = int(round(target_height * original_aspect))

    return new_height, new_width


def _resize_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]],
    resize_mode: str,
    maintain_aspect_ratio: bool,
    center_crop: bool,
) -> np.ndarray:
    """Resize and optionally center crop an image."""
    if target_size is None:
        return image

    original_height, original_width = image.shape[:2]
    resize_height, resize_width = _calculate_resize_dimensions(
        original_height, original_width, target_size, maintain_aspect_ratio, center_crop
    )

    interpolation = _map_interpolation(resize_mode)
    resized_image = cv2.resize(image, (resize_width, resize_height), interpolation=interpolation)

    if maintain_aspect_ratio and center_crop:
        target_height, target_width = target_size
        if resize_height != target_height or resize_width != target_width:
            y_start = max(0, (resize_height - target_height) // 2)
            x_start = max(0, (resize_width - target_width) // 2)
            y_end = min(resize_height, y_start + target_height)
            x_end = min(resize_width, x_start + target_width)
            resized_image = resized_image[y_start:y_end, x_start:x_end]

            if resized_image.shape[0] < target_height or resized_image.shape[1] < target_width:
                pad_height = max(0, target_height - resized_image.shape[0])
                pad_width = max(0, target_width - resized_image.shape[1])
                resized_image = np.pad(
                    resized_image, ((0, pad_height), (0, pad_width), (0, 0)), mode="constant", constant_values=0
                )

    return resized_image


def _load_image(image_path: str) -> np.ndarray:
    """Load an image from file."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _load_metadata(data_folder: Path, image_extensions: List[str] = None) -> List[Dict]:
    """
    Load metadata from meta.json or scan directory for images.

    Expected meta.json format (JSON array):
    [
        {
            "file_name": "image1.jpg",
            "caption": "A description of the image"
        },
        ...
    ]

    Or JSON Lines format (one JSON object per line):
    {"file_name": "image1.jpg", "caption": "A description"}
    {"file_name": "image2.jpg", "caption": "Another description"}
    """
    if image_extensions is None:
        image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

    meta_path = data_folder / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            content = f.read().strip()

            # Try to parse as JSON array first
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # If that fails, try parsing as JSON Lines (one JSON object per line)
                items = []
                for line in content.split("\n"):
                    line = line.strip()
                    if line:
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse line in meta.json: {line[:100]}... Error: {e}")
                            continue
                if items:
                    return items
                raise ValueError("Failed to parse meta.json as either JSON array or JSON Lines format")

    # Fallback: scan for image files with sidecar captions
    items: List[Dict] = []
    for entry in sorted(data_folder.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in image_extensions:
            continue

        image_name = entry.name
        # Look for caption in .txt file
        caption_file = entry.with_suffix(".txt")
        caption = ""
        if caption_file.exists():
            with open(caption_file, "r") as f:
                caption = f.read().strip()

        items.append(
            {
                "file_name": image_name,
                "caption": caption,
            }
        )

    if not items:
        raise FileNotFoundError(f"No meta.json and no image files found in {data_folder}")
    return items


@torch.no_grad()
def _init_flux_vae(
    model_id: str,
    device: str,
    enable_memory_optimization: bool,
):
    """Initialize FLUX VAE from pretrained model."""
    try:
        from diffusers import AutoencoderKL
    except ImportError:
        raise ImportError("Please install diffusers: pip install diffusers")

    # Use float32 for all devices to avoid dtype mismatch issues
    # The FLUX VAE appears to have internal operations that require float32
    dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=dtype,
    )
    # Ensure all parameters and buffers are on the correct device and dtype
    vae = vae.to(device=device, dtype=dtype)
    vae.eval()

    # Verify dtype consistency for all parameters
    for name, param in vae.named_parameters():
        if param.dtype != dtype:
            param.data = param.data.to(dtype)

    # Also convert buffers (like running means in batch norm)
    for name, buffer in vae.named_buffers():
        if buffer.dtype not in [torch.int32, torch.int64, torch.long]:  # Skip integer buffers
            if buffer.dtype != dtype:
                buffer.data = buffer.data.to(dtype)

    if enable_memory_optimization and hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    if enable_memory_optimization and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()

    return vae, dtype


@torch.no_grad()
def _init_text_encoders(
    t5_model_id: str,
    clip_model_id: str,
    device: str,
):
    """Initialize T5 and CLIP text encoders."""
    # Use float32 to avoid dtype mismatch with Apex fused layer norm
    dtype = torch.float32

    # T5 encoder
    t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model_id)
    t5_encoder = T5EncoderModel.from_pretrained(t5_model_id, torch_dtype=dtype)
    t5_encoder.to(device=device, dtype=dtype)
    t5_encoder.eval()

    # CLIP encoder
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_id)
    clip_encoder = CLIPTextModel.from_pretrained(clip_model_id, torch_dtype=dtype)
    clip_encoder.to(device=device, dtype=dtype)
    clip_encoder.eval()

    return t5_tokenizer, t5_encoder, clip_tokenizer, clip_encoder, dtype


@torch.no_grad()
def _encode_text_flux(
    t5_tokenizer: T5TokenizerFast,
    t5_encoder: T5EncoderModel,
    clip_tokenizer: CLIPTokenizer,
    clip_encoder: CLIPTextModel,
    device: str,
    caption: str,
    max_sequence_length: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode text with both T5 and CLIP encoders.

    Returns:
        Tuple of (t5_embeds [seq_len, hidden_dim], clip_pooled_embeds [hidden_dim])
    """
    caption = caption.strip()

    # T5 encoding
    t5_inputs = t5_tokenizer(
        caption,
        max_length=max_sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    t5_inputs = {k: v.to(device) for k, v in t5_inputs.items()}
    t5_outputs = t5_encoder(input_ids=t5_inputs["input_ids"], attention_mask=t5_inputs["attention_mask"])
    t5_embeds = t5_outputs.last_hidden_state[0]  # [seq_len, hidden_dim]

    # CLIP encoding
    clip_inputs = clip_tokenizer(
        caption,
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}
    clip_outputs = clip_encoder(input_ids=clip_inputs["input_ids"])
    clip_pooled_embeds = clip_outputs.pooler_output[0]  # [hidden_dim]

    return t5_embeds, clip_pooled_embeds


@torch.no_grad()
def _encode_image_latents(
    vae,
    device: str,
    image: np.ndarray,
    deterministic_latents: bool,
) -> torch.Tensor:
    """
    Encode image to latents using FLUX VAE.

    Args:
        vae: FLUX VAE model
        device: Device to use
        image: RGB numpy array [H, W, C] in range [0, 255]
        deterministic_latents: If True, use mean; if False, sample

    Returns:
        Latents tensor [C, H_latent, W_latent]
    """
    # Normalize to [0, 1] then to [-1, 1] (standard for diffusion VAEs)
    image = image.astype(np.float32) / 255.0
    image = image * 2.0 - 1.0

    # Convert to tensor [1, C, H, W]
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    # Match VAE's dtype to avoid dtype mismatch with internal weights
    image_tensor = image_tensor.to(device=device, dtype=vae.dtype)

    # FLUX VAE expects inputs in [-1, 1] range
    latent_dist = vae.encode(image_tensor)

    if deterministic_latents:
        latents = latent_dist.latent_dist.mode()
    else:
        latents = latent_dist.latent_dist.sample()

    # Remove batch dimension: [1, C, H, W] -> [C, H, W]
    latents = latents[0]

    return latents


def get_start_end_idx_for_this_rank(dataset_size: int, rank: int, world_size: int) -> Tuple[int, int]:
    """Calculate start and end indices for distributed processing."""
    split_size = dataset_size // world_size
    start_idx = rank * split_size
    # The last rank takes the remainder
    end_idx = start_idx + split_size if rank != world_size - 1 else dataset_size
    return start_idx, end_idx


def _save_individual_sample(
    output_dir: Path,
    sample_key: str,
    latents: torch.Tensor,
    text_embeddings: Dict,
    json_data: Dict,
    processed_image: Optional[np.ndarray] = None,
) -> None:
    """
    Save individual files for a sample.

    Args:
        output_dir: Base output directory
        sample_key: Unique key for this sample (e.g., "000001")
        latents: Latent tensor [C, H, W]
        text_embeddings: Dict with prompt_embeds and pooled_prompt_embeds
        json_data: Metadata dict
        processed_image: Optional processed image [H, W, C] in RGB format
    """
    sample_dir = output_dir / "individual_samples" / sample_key
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Save latents
    torch.save(latents, sample_dir / "latents.pt")

    # Save text embeddings
    with open(sample_dir / "text_embeddings.pkl", "wb") as f:
        pickle.dump(text_embeddings, f)

    # Save metadata
    with open(sample_dir / "metadata.json", "w") as f:
        json.dump(json_data, f, indent=2)

    # Optionally save processed image
    if processed_image is not None:
        # Convert RGB back to BGR for OpenCV
        image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(sample_dir / "processed_image.jpg"), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main():  # noqa: D103
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare FLUX WebDataset shards with VAE latents, T5, and CLIP embeddings"
    )
    parser.add_argument("--data_folder", type=str, required=True, help="Folder containing images and meta.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write webdataset shards")
    parser.add_argument(
        "--vae_model",
        default="black-forest-labs/FLUX.1-schnell",
        help="FLUX model ID for VAE (e.g., black-forest-labs/FLUX.1-schnell or FLUX.1-dev)",
    )
    parser.add_argument(
        "--t5_model",
        default="google/t5-v1_1-xxl",
        help="T5 model ID (e.g., google/t5-v1_1-xxl)",
    )
    parser.add_argument(
        "--clip_model",
        default="openai/clip-vit-large-patch14",
        help="CLIP model ID (e.g., openai/clip-vit-large-patch14)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic encoding (sampling) instead of deterministic mode",
    )
    parser.add_argument("--no-memory-optimization", action="store_true", help="Disable VAE slicing/tiling")
    parser.add_argument("--shard_maxcount", type=int, default=10000, help="Max samples per shard")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Max sequence length for T5 encoding")

    # Resize arguments
    parser.add_argument("--height", type=int, default=1024, help="Target height for images")
    parser.add_argument("--width", type=int, default=1024, help="Target width for images")
    parser.add_argument(
        "--resize_mode",
        default="bilinear",
        choices=["bilinear", "bicubic", "nearest", "area", "lanczos"],
        help="Interpolation mode for resizing",
    )
    parser.add_argument("--no-aspect-ratio", action="store_true", help="Disable aspect ratio preservation")
    parser.add_argument("--center-crop", action="store_true", help="Center crop to exact target size after resize")

    # Distributed processing
    parser.add_argument("--distributed", action="store_true", help="Use distributed processing")

    # Individual file saving
    parser.add_argument(
        "--save_individual_files",
        action="store_true",
        help="Save individual files (latents, embeddings, metadata) in addition to webdataset tars",
    )
    parser.add_argument(
        "--save_processed_images",
        action="store_true",
        help="Also save processed images when --save_individual_files is enabled",
    )

    args = parser.parse_args()

    data_folder = Path(args.data_folder)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup distributed processing if requested
    if args.distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank = 0
        world_size = 1
        device = args.device

    # Output shard pattern
    if world_size > 1:
        shard_pattern = str(output_dir / f"rank{rank}-%06d.tar")
    else:
        shard_pattern = str(output_dir / "shard-%06d.tar")

    # Target size
    target_size = (args.height, args.width)

    # Initialize models
    print(f"Rank {rank}: Initializing models...")
    vae, vae_dtype = _init_flux_vae(
        model_id=args.vae_model,
        device=device,
        enable_memory_optimization=not args.no_memory_optimization,
    )

    t5_tokenizer, t5_encoder, clip_tokenizer, clip_encoder, text_dtype = _init_text_encoders(
        t5_model_id=args.t5_model,
        clip_model_id=args.clip_model,
        device=device,
    )

    # Load metadata
    metadata_list = _load_metadata(data_folder)
    print(f"Total samples in dataset: {len(metadata_list)}")

    # Distribute work across ranks
    start_idx, end_idx = get_start_end_idx_for_this_rank(len(metadata_list), rank, world_size)
    print(f"Rank {rank} of {world_size} processing {end_idx - start_idx} samples, from {start_idx} to {end_idx}")

    if args.save_individual_files:
        print(f"Individual files will be saved to: {output_dir / 'individual_samples'}")
        if args.save_processed_images:
            print("Processed images will also be saved")

    with wds.ShardWriter(shard_pattern, maxcount=args.shard_maxcount) as sink:
        written = 0
        for index in tqdm(range(start_idx, end_idx), desc=f"Rank {rank}"):
            meta = metadata_list[index]
            image_name = meta["file_name"]
            caption = meta.get("caption", "")

            image_path = str(data_folder / image_name)

            try:
                # Load and resize image
                image = _load_image(image_path)
                image = _resize_image(
                    image=image,
                    target_size=target_size,
                    resize_mode=args.resize_mode,
                    maintain_aspect_ratio=not args.no_aspect_ratio,
                    center_crop=args.center_crop,
                )

                H, W = image.shape[:2]

                # Encode image to latents
                latents = _encode_image_latents(vae, device, image, deterministic_latents=not args.stochastic)

                # Encode text with T5 and CLIP
                t5_embeds, clip_pooled_embeds = _encode_text_flux(
                    t5_tokenizer,
                    t5_encoder,
                    clip_tokenizer,
                    clip_encoder,
                    device,
                    caption,
                    max_sequence_length=args.max_sequence_length,
                )

                # Move to CPU
                latents_cpu = latents.detach().to(device="cpu")
                t5_embeds_cpu = t5_embeds.detach().to(device="cpu")
                clip_pooled_embeds_cpu = clip_pooled_embeds.detach().to(device="cpu")

                # Create text embeddings dict
                text_embeddings = {
                    "prompt_embeds": t5_embeds_cpu,
                    "pooled_prompt_embeds": clip_pooled_embeds_cpu,
                }

                # Build JSON metadata
                json_data = {
                    "image_path": image_path,
                    "processed_height": int(H),
                    "processed_width": int(W),
                    "caption": caption,
                    "deterministic_latents": bool(not args.stochastic),
                    "memory_optimization": bool(not args.no_memory_optimization),
                    "model_version": "flux",
                    "vae_normalization": "[-1, 1]",
                    "resize_settings": {
                        "target_size": target_size,
                        "resize_mode": args.resize_mode,
                        "maintain_aspect_ratio": bool(not args.no_aspect_ratio),
                        "center_crop": bool(args.center_crop),
                    },
                }

                # Write to webdataset
                sample = {
                    "__key__": f"{index:06}",
                    "pth": latents_cpu,
                    "pickle": pickle.dumps(text_embeddings),
                    "json": json_data,
                }
                sink.write(sample)
                written += 1

                # Optionally save individual files
                if args.save_individual_files:
                    _save_individual_sample(
                        output_dir=output_dir,
                        sample_key=f"{index:06}",
                        latents=latents_cpu,
                        text_embeddings=text_embeddings,
                        json_data=json_data,
                        processed_image=image if args.save_processed_images else None,
                    )

            except Exception as e:
                print(f"Rank {rank}: Error processing {image_path}: {e}")
                continue

    print(f"Rank {rank}: Done! Wrote {written} samples.")

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
