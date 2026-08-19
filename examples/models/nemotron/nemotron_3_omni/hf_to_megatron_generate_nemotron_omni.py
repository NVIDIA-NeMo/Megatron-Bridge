# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
Nemotron Omni VL Generation Script.

This script demonstrates inference with Nemotron Omni model
using Megatron-Bridge. Unlike the InternVL-based Nemotron VL models that rely on
qwen_vl_utils, this script uses the model's native HF processor with <image> tokens.

Nemotron Omni always uses dynamic-resolution RADIO inputs. Temporal vision
settings are modality-dependent:
  * Image: temporal_patch_dim=1,
    separate_video_embedder=False. Each HF-processor tile is pre-patchified
    into [1, total_patches, 3*P*P] and passed through RADIO's packed
    dynamic-resolution path. The
    ``imgs_sizes`` / ``vision_packed_seq_params`` tensors are built from the
    per-tile shapes, and the prompt is expanded to one image placeholder per
    projected RADIO feature.
  * Audio / text-only: temporal_patch_dim=1 (the vision encoder is unused).
  * Video (and video+audio): temporal_patch_dim=2,
    separate_video_embedder=True, temporal_ckpt_compat=True so RADIO ViT
    exercises the trained `video_embedder`. The video preprocessing mirrors
    the shared SFT collator (used by `NemotronOmniTaskEncoder` with
    `use_temporal_video_embedder=True`): frames are grouped in pairs, all frames
    are pre-patchified into [1, total_patches, 3*P*P], and `imgs_sizes`
    / `num_frames` / `vision_packed_seq_params` are plumbed through to the
    canonical ``NemotronOmniModel``.

Examples:
  # Single image:
  uv run python examples/models/nemotron/nemotron_3_omni/hf_to_megatron_generate_nemotron_omni.py \
    --hf_model_path="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16" \
    --image_path="https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
    --prompt="Describe this image." \
    --max_new_tokens 300

  # Multiple images:
  uv run python examples/models/nemotron/nemotron_3_omni/hf_to_megatron_generate_nemotron_omni.py \
    --hf_model_path="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16" \
    --image_path="https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/example1a.jpeg,https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/example1b.jpeg" \
    --prompt="Describe the two images in detail." \
    --max_new_tokens 300

  # Video description:
  uv run python examples/models/nemotron/nemotron_3_omni/hf_to_megatron_generate_nemotron_omni.py \
    --hf_model_path="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16" \
    --video_path="https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/demo.mp4" \
    --prompt="Describe what you see." \
    --max_new_tokens 300

  # Audio transcription:
  uv run python examples/models/nemotron/nemotron_3_omni/hf_to_megatron_generate_nemotron_omni.py \
    --hf_model_path="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16" \
    --audio_path="/path/to/audio.wav" \
    --prompt="Transcribe the audio." \
    --max_new_tokens 300
"""

import argparse
import math
from typing import Optional

import requests
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.models.nemotron_omni.nemotron_omni_utils import (
    COMPACT_IMAGE_PLACEHOLDER,
    inference_expanded_image_token_counts,
    inference_num_image_tiles,
    patchify_temporal_frame,
    select_inference_next_token,
    temporal_model_frames,
    valid_audio_feature_lengths,
)
from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import adjust_image_tokens
from megatron.bridge.utils.common_utils import get_last_rank, print_rank_0


# Must stay in sync with nemotron_omni_provider.py and the temporal SFT recipe
# `nemotron_omni_valor32k_sft_config` (patch_dim=16, 512x512 frames,
# tps=2). Reused across video / video+audio preprocessing for temporal inference.
_VIDEO_TEMPORAL_PATCH_SIZE = 2
_VIDEO_FRAME_H = 512
_VIDEO_FRAME_W = 512
_VISION_PATCH_DIM = 16
_VIDEO_FPS = 1
_VIDEO_NFRAMES = 8


def _build_vision_packed_seq_params(imgs_sizes: Optional[torch.Tensor]) -> Optional[PackedSeqParams]:
    """Build vision PackedSeqParams from pre-grouping per-frame (H, W).

    RADIO's dynamic-resolution + class-token path reads ``packed_seq_params.cu_seqlens_q``
    to insert class tokens at per-image boundaries, and ``_apply_temporal_grouping``
    rebuilds cu_seqlens after tubelet fusion. Copied from
    ``megatron.bridge.training.nemotron_omni_step._build_vision_packed_seq_params``.
    """
    if imgs_sizes is None or imgs_sizes.numel() == 0:
        return None
    sizes = imgs_sizes.tolist() if torch.is_tensor(imgs_sizes) else list(imgs_sizes)
    seq_lens = [(int(h) // _VISION_PATCH_DIM) * (int(w) // _VISION_PATCH_DIM) for h, w in sizes]
    cu = [0]
    for sl in seq_lens:
        cu.append(cu[-1] + sl)
    device = imgs_sizes.device if torch.is_tensor(imgs_sizes) else torch.device("cpu")
    cu_tensor = torch.tensor(cu, dtype=torch.int32, device=device)
    max_len = torch.tensor(max(seq_lens) if seq_lens else 0, dtype=torch.int32, device=device)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_tensor,
        cu_seqlens_kv=cu_tensor,
        max_seqlen_q=max_len,
        max_seqlen_kv=max_len,
    )


def _fastconformer_output_length(mel_length: int, subsampling_factor: int = 8) -> int:
    """Mirror BridgeSoundEncoder._compute_output_lengths for a scalar input.

    The FastConformer subsampling stack does Conv2D(kernel=3, stride=2, padding=1,
    floor mode) repeated ``log2(subsampling_factor)`` times. The recurrence
    reduces to ``L -> (L - 1) // 2 + 1`` per layer.
    """
    num_layers = int(math.log2(subsampling_factor))
    length = int(mel_length)
    for _ in range(num_layers):
        length = (length - 1) // 2 + 1
    return length


def _align_sound_tokens(input_ids: torch.Tensor, sound_token_id: int, desired_count: int) -> torch.Tensor:
    """Rewrite ``input_ids`` so the contiguous run of sound tokens has ``desired_count`` entries.

    The HF processor estimates the sound-token count from the waveform length. Bridge
    recomputes it from Parakeet's semantic feature-mask length so padding and boundary
    frames remain invalid, then defensively realigns any differing processor token run.
    """
    assert input_ids.dim() == 2 and input_ids.size(0) == 1, "Expected a single-sample batch"
    mask = input_ids[0] == sound_token_id
    positions = torch.where(mask)[0]
    if positions.numel() == 0:
        raise ValueError(f"No sound tokens (id={sound_token_id}) found in input_ids")
    current_count = int(positions.numel())
    if current_count == desired_count:
        return input_ids
    start = int(positions[0].item())
    end = int(positions[-1].item()) + 1
    if (end - start) != current_count:
        raise ValueError("Sound tokens are not contiguous; cannot safely re-align")
    prefix = input_ids[:, :start]
    suffix = input_ids[:, end:]
    middle = torch.full((1, desired_count), sound_token_id, dtype=input_ids.dtype, device=input_ids.device)
    return torch.cat([prefix, middle, suffix], dim=1)


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

        if kwargs.get("images", None) is not None:
            self.batch["images"] = kwargs.get("images", None)
        elif kwargs.get("pixel_values", None) is not None:
            self.batch["pixel_values"] = kwargs.get("pixel_values", None)

        # Sound inputs
        if kwargs.get("sound_clips", None) is not None:
            self.batch["sound_clips"] = kwargs["sound_clips"]
        if kwargs.get("sound_length", None) is not None:
            self.batch["sound_length"] = kwargs["sound_length"]

        # Temporal video embedder inputs (dynamic-resolution pre-patchified path)
        if kwargs.get("imgs_sizes", None) is not None:
            self.batch["imgs_sizes"] = kwargs["imgs_sizes"]
        if kwargs.get("num_frames", None) is not None:
            self.batch["num_frames"] = kwargs["num_frames"]
        if kwargs.get("vision_packed_seq_params", None) is not None:
            self.batch["vision_packed_seq_params"] = kwargs["vision_packed_seq_params"]

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

    # Keep the empty-image sentinel used by the training step for audio/text.
    if "images" not in forward_args and "pixel_values" not in forward_args:
        forward_args["images"] = torch.tensor([], dtype=torch.bfloat16, device=batch["tokens"].device).reshape(0, 0, 0)

    if "sound_clips" in batch:
        forward_args["sound_clips"] = batch["sound_clips"]
    if "sound_length" in batch:
        forward_args["sound_length"] = batch["sound_length"]

    # Temporal video embedder plumbing (RADIO dynamic-resolution + separate
    # video embedder path). Matches nemotron_omni_step.forward_step kwargs.
    if "imgs_sizes" in batch:
        forward_args["imgs_sizes"] = batch["imgs_sizes"]
    if "num_frames" in batch:
        forward_args["num_frames"] = batch["num_frames"]
    if "vision_packed_seq_params" in batch:
        forward_args["vision_packed_seq_params"] = batch["vision_packed_seq_params"]

    def loss_func(x, **kwargs):
        return x

    output = model(**forward_args)
    # CP training can return (logits, loss_mask); generation needs only logits.
    if isinstance(output, tuple):
        output = output[0]
    return output, loss_func


def load_image(image_path: str) -> Image.Image:
    """Load an image from URL or file path.

    Args:
        image_path: URL or local file path to the image

    Returns:
        PIL Image object
    """
    if image_path.startswith(("http://", "https://")):
        response = requests.get(image_path, stream=True)
        response.raise_for_status()
        return Image.open(response.raw)
    else:
        return Image.open(image_path)


def _patchify_pixel_values(
    pv: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    patch_dim: int = _VISION_PATCH_DIM,
):
    """Pack image tiles into [1, total_patches, 3*P*P] patches.

    Dynamic-resolution processors return either one stacked ``[N, 3, H, W]``
    tensor when all tiles have the same size, or a list of ``[3, H, W]`` /
    ``[N, 3, H, W]`` tensors when tile sizes differ. Normalize both forms and
    concatenate their patches so RADIO sees one packed sequence matching
    ``imgs_sizes``.
    """
    if isinstance(pv, torch.Tensor):
        pixel_groups = [pv]
    elif isinstance(pv, (list, tuple)):
        pixel_groups = list(pv)
    else:
        raise TypeError(f"pixel_values must be a tensor or sequence of tensors, got {type(pv).__name__}")

    tiles = []
    for group in pixel_groups:
        if not isinstance(group, torch.Tensor):
            raise TypeError(f"Each pixel_values entry must be a tensor, got {type(group).__name__}")
        if group.ndim == 3:
            tiles.append(group)
        elif group.ndim == 4:
            tiles.extend(group.unbind(0))
        else:
            raise ValueError(f"Each pixel_values tensor must be 3D or 4D, got shape {tuple(group.shape)}")
    if not tiles:
        raise ValueError("pixel_values must contain at least one image tile")

    P = patch_dim
    patches_list = []
    sizes = []
    for tile in tiles:
        _, H, W = tile.shape
        if H % P != 0 or W % P != 0:
            raise ValueError(f"Image tile shape {(H, W)} is not divisible by patch_dim={P}")
        py, px = H // P, W // P
        p = tile.unsqueeze(0).reshape(1, 3, py, P, px, P).permute(0, 2, 4, 1, 3, 5).reshape(1, py * px, 3 * P * P)
        patches_list.append(p)
        sizes.append([H, W])
    packed = torch.cat(patches_list, dim=1)
    return packed, sizes


def process_image_inputs(
    tokenizer, processor, image_path: Optional[str], prompt: str, system_prompt: Optional[str] = None
):
    """Process image inputs using the model's native processor.

    Uses <image> token directly in the text and the model's own tokenizer/processor.
    Each HF-processor tile is pre-patchified into the packed dynamic-resolution
    format expected by Nemotron Omni's RADIO encoder (temporal_patch_dim=1).

    Returns:
        Tuple of (input_ids, packed_pixel_values, num_patches, imgs_sizes).
        ``packed_pixel_values`` has shape [1, total_patches, 3*P*P]; ``imgs_sizes``
        has shape [N_tiles, 2] with per-tile (H, W). Both are None for text-only.
    """
    if image_path:
        image_paths = image_path.split(",") if "," in image_path else [image_path]
        images = [load_image(p.strip()) for p in image_paths]

        image_placeholders = "\n".join(["<image>"] * len(images))
        text_content = f"{image_placeholders}\n{prompt}"

        messages = [{"role": "user", "content": text_content}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt")

        pixel_values = inputs.pixel_values
        # Pre-patchify for the packed dynamic-resolution RADIO path before
        # validating ownership because heterogeneous tiles may be a list.
        packed_pv, sizes = _patchify_pixel_values(pixel_values)
        tile_count = len(sizes)
        if hasattr(inputs, "num_patches") and inputs.num_patches is not None:
            num_patches = torch.as_tensor(inputs.num_patches, dtype=torch.long).reshape(-1)
            if num_patches.numel() != len(images) or int(num_patches.sum().item()) != tile_count:
                raise ValueError(
                    "num_patches must provide one ownership count per source image and sum to the number of tiles"
                )
        else:
            if tile_count != len(images):
                raise ValueError(
                    "The image processor returned multiple tiles per source image without num_patches ownership "
                    "metadata."
                )
            num_patches = torch.ones(len(images), dtype=torch.long)

        imgs_sizes = torch.tensor(sizes, dtype=torch.long)

        print_rank_0(
            f"Image: {image_path}, tiles={tile_count}, "
            f"packed_shape={tuple(packed_pv.shape)}, num_patches={num_patches.tolist()}"
        )
        return inputs.input_ids, packed_pv, num_patches, imgs_sizes
    else:
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        return inputs.input_ids, None, 0, None


def process_video_inputs(tokenizer, video_path: Optional[str], prompt: str, system_prompt: Optional[str] = None):
    """Process video inputs for the temporal video embedder inference path.

    Mirrors the shared ``nemotron_omni_collate_fn`` SFT data pipeline so the
    model sees the same tensor shapes it was trained on:

    - Frames are grouped by ``temporal_patch_size`` (pair of consecutive frames
      per image placeholder) and the prompt uses the training-time timestamp
      format with one compact ``<img><image></img>`` wrapper per group.
    - All sampled video frames are patchified to a [1, total_patches, 3*P*P]
      tensor. A single frame is repeated only in this model tensor/metadata so
      RADIO selects the temporal embedder; the prompt still has one placeholder.
    - ``imgs_sizes`` and ``num_frames`` describe those model frames so RADIO's
      ``_apply_temporal_grouping`` can pad incomplete groups, fuse tubelets, and
      route them through the trained ``video_embedder``.

    Returns:
        Tuple of (input_ids, pixel_values, num_patches, imgs_sizes, num_frames).
    """
    from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import (
        maybe_path_or_url_to_data_urls,
        pil_image_from_base64,
    )

    tps = _VIDEO_TEMPORAL_PATCH_SIZE

    # 1. Extract frames from the video
    image_urls, metadata = maybe_path_or_url_to_data_urls(
        video_path,
        fps=max(0, int(_VIDEO_FPS)),
        nframe=max(0, int(_VIDEO_NFRAMES)),
        nframe_max=-1,
    )
    frames = [pil_image_from_base64(url) for url in image_urls]
    print_rank_0(f"Video: extracted {len(frames)} frames, metadata: {metadata}")

    if not frames:
        raise ValueError("Temporal video embedder inference requires at least one sampled frame.")
    model_frames = temporal_model_frames(frames, tps)

    # 2. Build the training-style prompt with one compact wrapper per group.
    fps_for_ts = float(metadata.fps) if (metadata and metadata.fps) else float(_VIDEO_FPS)
    video_prompt_lines = ["This is a video:"]
    for i in range(0, len(frames), tps):
        group = frames[i : i + tps]
        ts_parts = [
            f"{'Frame' if j == 0 else 'frame'} {i + j + 1} sampled at {(i + j) / fps_for_ts:.2f} seconds"
            for j in range(len(group))
        ]
        video_prompt_lines.append(" and ".join(ts_parts) + f": {COMPACT_IMAGE_PLACEHOLDER}")

    content = "\n".join(video_prompt_lines) + "\n" + prompt
    messages = [{"role": "user", "content": content}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 3. The compact visual wrappers are already model-ready; no image
    #    processor prepass is needed because its pixels would be discarded.
    input_ids = tokenizer([text], return_tensors="pt").input_ids
    num_patches = torch.ones(len(video_prompt_lines) - 1, dtype=torch.long)

    # 4. Replace the HF processor's pixels with the model-frame patch tensor.
    all_patches = [
        patchify_temporal_frame(
            frame,
            height=_VIDEO_FRAME_H,
            width=_VIDEO_FRAME_W,
            patch_dim=_VISION_PATCH_DIM,
        )
        for frame in model_frames
    ]
    packed_pixel_values = torch.cat(all_patches, dim=0).unsqueeze(0)

    imgs_sizes = torch.tensor([[_VIDEO_FRAME_H, _VIDEO_FRAME_W]] * len(model_frames), dtype=torch.long)
    num_frames = torch.tensor([len(model_frames)], dtype=torch.long)

    return input_ids, packed_pixel_values, num_patches, imgs_sizes, num_frames


def process_audio_inputs(tokenizer, processor, audio_path: str, prompt: str, system_prompt: Optional[str] = None):
    """Process audio inputs for the Megatron sound encoder.

    Uses the HF processor to expand <so_embedding> tokens and extract mel
    spectrogram features via ParakeetFeatureExtractor. The Megatron
    BridgeSoundEncoder expects pre-processed mel spectrograms, not raw waveforms.

    Returns:
        Tuple of (input_ids, sound_clips, sound_length) where sound_clips is a
        mel spectrogram tensor and sound_length is the frame count tensor.
    """
    from transformers import ParakeetFeatureExtractor

    audio_token = getattr(tokenizer, "audio_token", "<so_embedding>")
    messages = [{"role": "user", "content": f"{audio_token}\n{prompt}"}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Processor handles audio loading and <so_embedding> token expansion
    inputs = processor(text=[text], audio=[audio_path], return_tensors="pt")
    raw_sound_clips = inputs.pop("sound_clips", None)

    # Extract mel spectrogram features (Megatron BridgeSoundEncoder expects mel, not raw waveform)
    feature_extractor = ParakeetFeatureExtractor(sampling_rate=16000, feature_size=128)
    audio_features = feature_extractor(
        raw_sound_clips,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
    )
    sound_clips = audio_features.input_features  # [batch, frames, mel_bins]
    sound_length = valid_audio_feature_lengths(
        audio_features.attention_mask,
        num_frames=sound_clips.shape[1],
    )

    # Realign <so_embedding> tokens in the prompt to match the encoder's actual output
    # length (see _align_sound_tokens docstring for why this can differ from the HF count).
    sound_token_id = tokenizer.convert_tokens_to_ids(audio_token)
    expected_sound_tokens = _fastconformer_output_length(int(sound_length.item()))
    input_ids = _align_sound_tokens(inputs.input_ids, sound_token_id, expected_sound_tokens)

    print_rank_0(f"Audio: {audio_path}")
    print_rank_0(
        f"Sound clips shape: {sound_clips.shape}, sound_length: {sound_length}, "
        f"expected_sound_tokens: {expected_sound_tokens}"
    )

    return input_ids, sound_clips, sound_length


def process_video_audio_inputs(
    tokenizer, processor, video_path: str, audio_path: str, prompt: str, system_prompt: Optional[str] = None
):
    """Process combined video + audio inputs for the temporal video embedder path.

    Same temporal preprocessing as ``process_video_inputs`` (frame pairing,
    patchification, imgs_sizes / num_frames), with audio mel features + token
    realignment layered on top, matching the SFT training pipeline.

    Returns:
        Tuple of (input_ids, pixel_values, num_patches, imgs_sizes, num_frames,
                  sound_clips, sound_length).
    """
    from transformers import ParakeetFeatureExtractor

    from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import (
        maybe_path_or_url_to_data_urls,
        pil_image_from_base64,
    )

    tps = _VIDEO_TEMPORAL_PATCH_SIZE

    image_urls, metadata = maybe_path_or_url_to_data_urls(
        video_path,
        fps=max(0, int(_VIDEO_FPS)),
        nframe=max(0, int(_VIDEO_NFRAMES)),
        nframe_max=-1,
    )
    frames = [pil_image_from_base64(url) for url in image_urls]
    print_rank_0(f"Video: extracted {len(frames)} frames, metadata: {metadata}")

    if not frames:
        raise ValueError("Temporal video embedder inference requires at least one sampled frame.")
    model_frames = temporal_model_frames(frames, tps)

    fps_for_ts = float(metadata.fps) if (metadata and metadata.fps) else float(_VIDEO_FPS)
    video_prompt_lines = ["This is a video:"]
    for i in range(0, len(frames), tps):
        group = frames[i : i + tps]
        ts_parts = [
            f"{'Frame' if j == 0 else 'frame'} {i + j + 1} sampled at {(i + j) / fps_for_ts:.2f} seconds"
            for j in range(len(group))
        ]
        video_prompt_lines.append(" and ".join(ts_parts) + f": {COMPACT_IMAGE_PLACEHOLDER}")

    audio_token = getattr(tokenizer, "audio_token", "<so_embedding>")
    content = "\n".join(video_prompt_lines) + f"\nThis is the audio: {audio_token}\n" + prompt
    messages = [{"role": "user", "content": content}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Let the processor expand only the audio placeholder. The visual wrappers
    # are already compact and require no discarded image preprocessing.
    proc_output = processor(text=[text], audio=[audio_path], return_tensors="pt")

    # Extract raw sound clips and convert to mel spectrogram for BridgeSoundEncoder
    raw_sound_clips = proc_output.pop("sound_clips", None)
    feature_extractor = ParakeetFeatureExtractor(sampling_rate=16000, feature_size=128)
    audio_features = feature_extractor(
        raw_sound_clips,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
    )
    sound_clips = audio_features.input_features
    sound_length = valid_audio_feature_lengths(
        audio_features.attention_mask,
        num_frames=sound_clips.shape[1],
    )

    sound_token_id = tokenizer.convert_tokens_to_ids(audio_token)
    expected_sound_tokens = _fastconformer_output_length(int(sound_length.item()))
    input_ids = _align_sound_tokens(proc_output.input_ids, sound_token_id, expected_sound_tokens)

    print_rank_0(
        f"Audio: {audio_path}, sound_clips shape: {sound_clips.shape}, expected_sound_tokens: {expected_sound_tokens}"
    )

    num_patches = torch.ones(len(video_prompt_lines) - 1, dtype=torch.long)
    all_patches = [
        patchify_temporal_frame(
            frame,
            height=_VIDEO_FRAME_H,
            width=_VIDEO_FRAME_W,
            patch_dim=_VISION_PATCH_DIM,
        )
        for frame in model_frames
    ]
    packed_pixel_values = torch.cat(all_patches, dim=0).unsqueeze(0)
    imgs_sizes = torch.tensor([[_VIDEO_FRAME_H, _VIDEO_FRAME_W]] * len(model_frames), dtype=torch.long)
    num_frames = torch.tensor([len(model_frames)], dtype=torch.long)

    return input_ids, packed_pixel_values, num_patches, imgs_sizes, num_frames, sound_clips, sound_length


def _hf_revision_kwargs(revision: str | None) -> dict[str, str]:
    """Build keyword arguments for revision-pinned Hugging Face loads."""
    return {"revision": revision} if revision is not None else {}


def main(args) -> None:
    """Main function for Nemotron Omni VL generation from HuggingFace models.

    Loads a Nemotron Omni model either from HuggingFace (with optional conversion
    to Megatron) or directly from a Megatron checkpoint, then performs greedy
    generation using the provided prompt and optional image input.

    Args:
        args: Parsed command line arguments containing model paths, prompt,
              image path, parallelism settings, and generation parameters
    """
    # pylint: disable=C0115,C0116
    tp = args.tp
    pp = args.pp
    ep = args.ep
    etp = args.etp

    # Select temporal vision settings based on input modality. Nemotron Omni
    # always keeps RADIO in dynamic-resolution mode, including when vision is
    # unused for text/audio-only inference.
    #
    # Image: temporal_patch_dim=1 so that RADIO
    #   runs the packed dynamic-resolution path. Each HF-processor tile is pre-patchified into a packed
    #   [1, N*patches, 3*P*P] tensor and passed with imgs_sizes /
    #   vision_packed_seq_params. The prompt is expanded before model forward
    #   so every pipeline stage sees the canonical sequence length.
    #
    # Video (and video+audio): temporal_patch_dim=2,
    #   separate_video_embedder=True so RADIO exercises the trained
    #   `video_embedder`. The matching data pipeline is in `process_video_inputs`
    #   / `process_video_audio_inputs`.
    #
    # Audio / text-only: temporal_patch_dim=1; RADIO is not called.
    is_video_inference = bool(args.video_path)
    is_image_inference = bool(args.image_path) and not is_video_inference
    image_seq_len = (
        (_VIDEO_FRAME_H // _VISION_PATCH_DIM) * (_VIDEO_FRAME_W // _VISION_PATCH_DIM) // 4 if is_video_inference else 1
    )
    if is_video_inference:
        temporal_patch_dim = 2
        separate_video_embedder = True
        temporal_ckpt_compat = True
    elif is_image_inference:
        temporal_patch_dim = 1
        separate_video_embedder = False
        temporal_ckpt_compat = False
    else:
        temporal_patch_dim = 1
        separate_video_embedder = False
        temporal_ckpt_compat = False

    # Choose loading method based on arguments
    if args.megatron_model_path:
        # Load from Megatron checkpoint
        print_rank_0(f"Loading Megatron model from: {args.megatron_model_path}")

        # We still need HF config for tokenizer, but we'll load the model from Megatron checkpoint
        # Create bridge from HF config only (no weights)
        bridge = AutoBridge.from_hf_pretrained(
            args.hf_model_path,
            trust_remote_code=True,
            **_hf_revision_kwargs(args.hf_revision),
        )

        # Initialize model parallel before loading
        model_provider = bridge.to_megatron_provider(load_weights=False)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp
        model_provider.pipeline_dtype = torch.bfloat16
        model_provider.temporal_patch_dim = temporal_patch_dim
        model_provider.separate_video_embedder = separate_video_embedder
        model_provider.temporal_ckpt_compat = temporal_ckpt_compat
        # Canonical multimodal inputs retain their actual expanded length.
        # PP stages therefore need shape exchange instead of fixed receive buffers.
        model_provider.variable_seq_lengths = pp > 1
        model_provider.initialize_model_parallel(seed=0)

        # Load the Megatron model directly. The mp_overrides values are applied to
        # the loaded model_cfg before the model is built (see model_load_save.py),
        # so the temporal overrides must be passed here -- the
        # `model_provider` mutations above only affect the throwaway provider used
        # for parallel-state init, not the model that load_megatron_model builds.
        model = bridge.load_megatron_model(
            args.megatron_model_path,
            mp_overrides={
                "tensor_model_parallel_size": tp,
                "pipeline_model_parallel_size": pp,
                "expert_model_parallel_size": ep,
                "expert_tensor_parallel_size": etp,
                "pipeline_dtype": torch.bfloat16,
                "temporal_patch_dim": temporal_patch_dim,
                "separate_video_embedder": separate_video_embedder,
                "temporal_ckpt_compat": temporal_ckpt_compat,
                "variable_seq_lengths": pp > 1,
            },
            wrap_with_ddp=False,
        )
    else:
        # Load from HuggingFace and convert to Megatron
        print_rank_0(f"Loading HuggingFace model from: {args.hf_model_path}")
        bridge = AutoBridge.from_hf_pretrained(
            args.hf_model_path,
            trust_remote_code=True,
            **_hf_revision_kwargs(args.hf_revision),
        )
        model_provider = bridge.to_megatron_provider(load_weights=True)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp
        model_provider.pipeline_dtype = torch.bfloat16
        model_provider.temporal_patch_dim = temporal_patch_dim
        model_provider.separate_video_embedder = separate_video_embedder
        model_provider.temporal_ckpt_compat = temporal_ckpt_compat
        model_provider.variable_seq_lengths = pp > 1
        model_provider.initialize_model_parallel(seed=0)
        model_provider.finalize()
        model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    model = [m.cuda() for m in model]
    for m in model:
        m.eval()

    # Clear grad_scale_func for inference. Training checkpoints serialize
    # `optimizer.scale_loss` into the model config; reusing it here would
    # call the unbound MegatronOptimizer.scale_loss inside the MoE branch of
    # forward_step_calc_loss and crash with "missing 1 required positional
    # argument: 'loss'".
    for m in model:
        inner = m.module if hasattr(m, "module") else m
        if hasattr(inner, "config"):
            inner.config.grad_scale_func = None
        if hasattr(inner, "llava_model") and hasattr(inner.llava_model, "config"):
            inner.llava_model.config.grad_scale_func = None

    # Initialize tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_path,
        trust_remote_code=True,
        **_hf_revision_kwargs(args.hf_revision),
    )
    processor = AutoProcessor.from_pretrained(
        args.hf_model_path,
        trust_remote_code=True,
        **_hf_revision_kwargs(args.hf_revision),
    )
    img_start_token_id = tokenizer.convert_tokens_to_ids("<img>")
    img_end_token_id = tokenizer.convert_tokens_to_ids("</img>")
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sound_clips = None
    sound_length = None
    images = None
    imgs_sizes = None
    num_frames = None
    vision_packed_seq_params = None

    if args.video_path and args.audio_path:
        (
            input_ids,
            pixel_values,
            num_patches,
            imgs_sizes,
            num_frames,
            sound_clips,
            sound_length,
        ) = process_video_audio_inputs(
            tokenizer, processor, args.video_path, args.audio_path, args.prompt, args.system_prompt
        )
        images = pixel_values.bfloat16() if pixel_values is not None else None
    elif args.audio_path:
        input_ids, sound_clips, sound_length = process_audio_inputs(
            tokenizer, processor, args.audio_path, args.prompt, args.system_prompt
        )
    elif args.video_path:
        input_ids, pixel_values, num_patches, imgs_sizes, num_frames = process_video_inputs(
            tokenizer, args.video_path, args.prompt, args.system_prompt
        )
        images = pixel_values.bfloat16() if pixel_values is not None else None
    else:
        input_ids, pixel_values, num_patches, imgs_sizes = process_image_inputs(
            tokenizer, processor, args.image_path, args.prompt, args.system_prompt
        )
        images = pixel_values.bfloat16() if pixel_values is not None else None

    if images is not None:
        tile_feature_counts = inference_num_image_tiles(
            imgs_sizes,
            patch_dim=_VISION_PATCH_DIM,
            num_frames=num_frames,
            temporal_patch_size=temporal_patch_dim,
        )
        expanded_counts = inference_expanded_image_token_counts(
            tile_feature_counts,
            num_patches,
            feature_multiplier=image_seq_len,
        )
        # Normalize processor output to the canonical one-placeholder-per-
        # projected-feature contract.
        has_img_wrapper_tokens = (
            img_start_token_id != tokenizer.unk_token_id
            and img_end_token_id != tokenizer.unk_token_id
            and (input_ids == img_start_token_id).any()
        )
        if has_img_wrapper_tokens:
            input_ids = adjust_image_tokens(input_ids, expanded_counts, img_start_token_id, img_end_token_id)
        num_placeholders = int((input_ids == image_token_id).sum().item())
        expected_placeholders = int(expanded_counts.sum().item())
        if num_placeholders != expected_placeholders:
            raise ValueError(
                f"Vision metadata requires {expected_placeholders} expanded image placeholders; "
                f"the prompt contains {num_placeholders}."
            )
    pixel_values = None

    # Move to GPU
    input_ids = input_ids.cuda()
    if images is not None:
        images = images.cuda()
    if sound_clips is not None:
        sound_clips = sound_clips.bfloat16().cuda()
    if sound_length is not None:
        sound_length = sound_length.cuda()
    if imgs_sizes is not None:
        imgs_sizes = imgs_sizes.cuda()
    if num_frames is not None:
        num_frames = num_frames.cuda()

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

            # Rebuild RADIO vision packed-seq params each step. RADIO's forward
            # mutates `cu_seqlens_q/kv` and `max_seqlen_q/kv` in place to insert
            # class tokens (megatron/core/models/vision/radio.py:388-394). Reusing
            # the same object across iterations would compound the +class_token_len
            # shift on every step and eventually drive cu_seqlens past the end of
            # the embedded patch tensor, causing an async CUDA illegal memory
            # access in attention.
            vision_packed_seq_params = _build_vision_packed_seq_params(imgs_sizes) if imgs_sizes is not None else None

            fwd_bwd_function = get_forward_backward_func()
            iterator = SingleBatchIterator(
                input_ids,
                position_ids,
                attention_mask,
                pixel_values=pixel_values,
                images=images,
                sound_clips=sound_clips,
                sound_length=sound_length,
                imgs_sizes=imgs_sizes,
                num_frames=num_frames,
                vision_packed_seq_params=vision_packed_seq_params,
            )

            output = fwd_bwd_function(
                forward_step_func=vlm_forward_step,
                data_iterator=iterator,
                model=model,
                num_microbatches=1,
                forward_only=True,
                # Ignored for PP shape allocation when variable_seq_lengths is enabled.
                seq_length=model_provider.seq_length,
                micro_batch_size=1,
                collect_non_loss_data=True,
            )
            if isinstance(output, list) and len(output) > 0:
                output = output[0]
                if isinstance(output, tuple):
                    output = output[0]

            if parallel_state.is_pipeline_last_stage():
                world_size = parallel_state.get_tensor_model_parallel_world_size()
                gathered_tensors = [torch.zeros_like(output) for _ in range(world_size)]
                dist.all_gather(gathered_tensors, output, group=parallel_state.get_tensor_model_parallel_group())
                output = torch.cat(gathered_tensors, dim=2)
                merged_sequence_length = input_ids.shape[1]
                next_token_ids = select_inference_next_token(output, merged_sequence_length)

                if step < 5:
                    print_rank_0(f"Step {step}: output shape={output.shape}, var={output.var():.4f}")
                    logits = output[0, merged_sequence_length - 1, :]
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

            if next_token_ids.item() in stop_tokens:
                break

    # Decode the generated sequence
    generated_text = tokenizer.decode(list(generated_ids[0]))
    print_rank_0("======== GENERATED TEXT OUTPUT ========")
    if args.image_path:
        print_rank_0(f"Image: {args.image_path}")
    if args.video_path:
        print_rank_0(f"Video: {args.video_path}")
    if args.audio_path:
        print_rank_0(f"Audio: {args.audio_path}")
    print_rank_0(f"Prompt: {args.prompt}")
    print_rank_0(f"Generated: {generated_text}")
    print_rank_0("=======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nemotron Omni VL Generation from HuggingFace Models")
    parser.add_argument(
        "--hf_model_path",
        type=str,
        default="nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
        help="Path to the HuggingFace Nemotron Omni VL model.",
    )
    parser.add_argument(
        "--hf-revision",
        dest="hf_revision",
        default=None,
        help="Immutable Hugging Face Hub revision used for model, tokenizer, and processor loading.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image.",
        help="Input prompt for vision-language generation.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="/no_think",
        help="System prompt for vision-language generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
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
        help="Path or URL to the image for vision-language generation (optional). Multiple image paths can be separated"
        " with commas.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path or URL to the video for vision-language generation (optional).",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Path to an audio file for audio understanding (optional, WAV format).",
    )
    args = parser.parse_args()

    main(args)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
