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

"""Nemotron Omni Energon task encoder -- extends HFEncoderVLMTaskEncoder with audio.

Adds mel spectrogram extraction and ``<so_embedding>`` token insertion so that
the training step receives ``sound_clips`` / ``sound_length`` alongside the
standard vision + language tensors.
"""

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from megatron.energon import Batch, DefaultTaskEncoder

from megatron.bridge.data.energon.task_encoder_utils import (
    IGNORE_INDEX,
    ChatMLSample,
    _images_to_pil,
    _videos_to_pil,
    cook_chatml_sample,
    find_pattern_indices,
    get_ltor_masks_and_position_ids,
)
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample / batch dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NemotronOmniTaskSample:
    """Encoded sample for Nemotron Omni (vision + audio + language)."""

    __key__: str
    __subflavors__: Dict
    input_ids: torch.Tensor  # [seq_len]
    labels: torch.Tensor  # [seq_len]
    loss_mask: torch.Tensor  # [seq_len]
    visual_tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
    num_patches: Optional[torch.Tensor] = None  # [num_images] tile count per image
    sound_clips: Optional[torch.Tensor] = None  # [frames, mel_bins]
    sound_length: Optional[torch.Tensor] = None  # scalar
    imgs_sizes: Optional[torch.Tensor] = None  # [num_frames, 2] per-frame (H, W)
    num_frames: Optional[torch.Tensor] = None  # [num_media_items]
    num_image_tiles: Optional[torch.Tensor] = None  # [num_images] LM-side token count per image


@dataclass
class NemotronOmniTaskBatch(Batch):
    """Batched format for Nemotron Omni."""

    __keys__: List[str] = field(default_factory=list)
    __subflavors__: List[Dict] = field(default_factory=list)
    input_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    labels: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    loss_mask: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    attention_mask: Optional[torch.Tensor] = field(default_factory=lambda: torch.empty(0))
    position_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    visual_tensors: Dict[str, Optional[torch.Tensor]] = field(default_factory=dict)
    num_patches: Optional[torch.Tensor] = None  # tile counts per image
    sound_clips: Optional[torch.Tensor] = None  # [B, max_frames, mel_bins]
    sound_length: Optional[torch.Tensor] = None  # [B]
    imgs_sizes: Optional[torch.Tensor] = None  # [total_frames, 2]
    num_frames: Optional[torch.Tensor] = None  # [num_media_items]
    num_image_tiles: Optional[torch.Tensor] = None  # [total_images] LM-side token count per image
    # Packed-sequence metadata (only populated when pack_sequences=True).
    cu_seqlens: Optional[torch.Tensor] = None
    cu_seqlens_unpadded: Optional[torch.Tensor] = None
    cu_seqlens_argmin: Optional[torch.Tensor] = None
    max_seqlen: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Task encoder
# ---------------------------------------------------------------------------


class NemotronOmniTaskEncoder(DefaultTaskEncoder[ChatMLSample, NemotronOmniTaskSample, NemotronOmniTaskBatch, dict]):
    """Energon task encoder for Nemotron Omni models.

    Processes ChatML samples that may contain images, videos, AND audio
    waveforms (decoded from ``.wav`` fields in the WebDataset shards).

    Audio waveforms are converted to mel spectrograms using
    ``compute_mel_features``, and ``<so_embedding>`` placeholder tokens are
    inserted into ``input_ids`` so that ``LLaVAModel.forward()`` can
    replace them with the projected sound embeddings.

    Args:
        processor: HF ``AutoProcessor`` for the Nemotron Omni model.
        seq_length: Maximum sequence length after tokenization.
        max_audio_duration: Maximum audio duration in seconds. Longer clips
            are truncated.
        num_mel_bins: Number of mel frequency bins (must match the sound
            encoder config, typically 128 for Parakeet).
        visual_keys: Processor output keys to capture as visual tensors.
    """

    def __init__(
        self,
        processor,
        seq_length: int = 4096,
        max_audio_duration: float = 30.0,
        num_mel_bins: int = 128,
        visual_keys: Sequence[str] = ("pixel_values",),
        temporal_patch_size: int = 2,
        video_fps: float = 1.0,
        video_nframes: int = 8,
        use_temporal_video_embedder: bool = False,
        patch_dim: int = 16,
        pack_sequences: bool = False,
    ):
        super().__init__()
        self.processor = processor
        self.seq_length = seq_length
        self.max_audio_duration = max_audio_duration
        self.num_mel_bins = num_mel_bins
        self.visual_keys: Tuple[str, ...] = tuple(visual_keys)
        self.temporal_patch_size = temporal_patch_size
        self.video_fps = video_fps
        self.video_nframes = video_nframes
        self.use_temporal_video_embedder = use_temporal_video_embedder
        self.patch_dim = patch_dim
        self.pack_sequences = pack_sequences

    @staticmethod
    def _decode_video_bytes(video_bytes: bytes, nframes: int = 8, fps: float = 1.0):
        """Decode raw MP4 bytes to a list of PIL frames."""
        import tempfile

        from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import (
            maybe_path_or_url_to_data_urls,
            pil_image_from_base64,
        )

        # Write to temp file since video_io needs a file path
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            tmp.write(video_bytes)
            tmp.flush()
            try:
                image_urls, _ = maybe_path_or_url_to_data_urls(
                    tmp.name,
                    fps=max(0, int(fps)),
                    nframe=max(0, nframes),
                    nframe_max=-1,
                )
                frames = [pil_image_from_base64(url) for url in image_urls]
                return frames if frames else None
            except Exception:
                logger.warning("Failed to decode video bytes")
                return None

    def _patchify_frame(self, pil_img, target_h: int = 512, target_w: int = 512) -> torch.Tensor:
        """Convert a PIL image to [num_patches, C*P*P] patches (normalized).

        Matches the HF processor's normalization (CLIP mean/std).
        """
        from torchvision import transforms

        img = pil_img.resize((target_w, target_h))
        tensor = transforms.ToTensor()(img)  # [3, H, W]
        # Normalize with CLIP / RADIO mean/std
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        P = self.patch_dim
        py, px = target_h // P, target_w // P
        # [3, (py*P), (px*P)] → [py*px, 3*P*P]
        patches = tensor.reshape(3, py, P, px, P).permute(1, 3, 0, 2, 4).reshape(py * px, 3 * P * P)
        return patches

    @property
    def _tokenizer(self):
        return getattr(self.processor, "tokenizer", self.processor)

    @property
    def _pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id or 0

    @property
    def _eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    @property
    def _sound_token_id(self) -> int:
        return self._tokenizer.convert_tokens_to_ids("<so_embedding>")

    # ------------------------------------------------------------------
    # encode_sample
    # ------------------------------------------------------------------

    def encode_sample(self, sample: ChatMLSample) -> NemotronOmniTaskSample:
        """Encode a single ChatML sample with optional audio into model-ready tensors."""
        import math as _math

        from megatron.bridge.models.nemotron_omni.nemotron_omni_utils import (
            compute_mel_features,
        )

        # 1. Decode video → PIL frames
        videos_raw = sample.videos
        video_frames = None
        if videos_raw is not None:
            if isinstance(videos_raw, bytes):
                video_frames = self._decode_video_bytes(videos_raw, nframes=self.video_nframes, fps=self.video_fps)
            elif isinstance(videos_raw, list) and len(videos_raw) > 0:
                if isinstance(videos_raw[0], bytes):
                    all_f = []
                    for vb in videos_raw:
                        f = self._decode_video_bytes(vb, nframes=self.video_nframes, fps=self.video_fps)
                        if f:
                            all_f.extend(f)
                    video_frames = all_f if all_f else None
                else:
                    video_frames = _videos_to_pil(videos_raw)

        images_pil = _images_to_pil(sample.imgs) if sample.imgs is not None and len(sample.imgs) > 0 else None

        # 2. Process audio → mel spectrogram + compute token count
        n_sound_tokens = 0
        sound_clips_t: Optional[torch.Tensor] = None
        sound_length_t: Optional[torch.Tensor] = None

        if sample.audio is not None:
            waveform = sample.audio
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.numpy()
            waveform = waveform[: int(self.max_audio_duration * 16000)]

            mel = compute_mel_features(waveform, sampling_rate=16000, num_mel_bins=self.num_mel_bins)
            sound_clips_t = mel
            sound_length_t = torch.tensor(mel.shape[0], dtype=torch.long)

            token_len = float(mel.shape[0])
            for _ in range(3):
                token_len = _math.floor((token_len + 2 * 1 - 3) / 2 + 1)
            n_sound_tokens = max(1, int(token_len))

        # 3. Build prompt with temporal frame pairing + audio tokens
        #
        # Target format (user turn):
        #   This is a video:
        #   frame 1 sampled at 0.00 seconds and frame 2 sampled at 1.00 seconds: <image>
        #   frame 3 sampled at 2.00 seconds and frame 4 sampled at 3.00 seconds: <image>
        #   <so_start>[N×<so_embedding>]<so_end>
        #   {question text}
        #
        # Each <image> token represents temporal_patch_size consecutive frames combined.

        conversation = cook_chatml_sample(sample.conversation)
        tps = self.temporal_patch_size
        fps = self.video_fps

        # Group video frames by temporal_patch_size → one <image> per group
        paired_images = []
        video_prompt_lines = []
        if video_frames:
            video_prompt_lines.append("This is a video:")
            for i in range(0, len(video_frames), tps):
                group = video_frames[i : i + tps]
                ts_parts = []
                for j in range(len(group)):
                    ts_parts.append(f"frame {i + j + 1} sampled at {(i + j) / fps:.2f} seconds")
                video_prompt_lines.append(" and ".join(ts_parts) + ": <image>")
                paired_images.append(group[0])  # representative frame per group

        # Replace conversation content with the structured prompt
        all_proc_images = list(images_pil) if images_pil else []
        for turn in conversation:
            if not isinstance(turn.get("content"), list):
                continue
            new_parts = []
            for item in turn["content"]:
                if isinstance(item, dict) and item.get("type") == "video":
                    if video_prompt_lines:
                        new_parts.append("\n".join(video_prompt_lines))
                        all_proc_images.extend(paired_images)
                elif isinstance(item, dict) and item.get("type") == "image":
                    new_parts.append("<image>")
                elif isinstance(item, dict) and item.get("type") == "text":
                    new_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    new_parts.append(item)
            turn["content"] = "\n".join(new_parts)

        # 4. Apply chat template → tokenize + process vision
        text_conv = [
            t if isinstance(t["content"], str) else {"role": t["role"], "content": str(t["content"])}
            for t in conversation
        ]
        prompt_text = self._tokenizer.apply_chat_template(text_conv, tokenize=False, add_generation_prompt=False)

        proc_kwargs = {"text": prompt_text, "return_tensors": "pt"}
        if all_proc_images:
            proc_kwargs["images"] = all_proc_images

        orig_tiles = getattr(self.processor.image_processor, "max_num_tiles", None)
        if all_proc_images and orig_tiles is not None:
            self.processor.image_processor.max_num_tiles = 1
        proc_output = self.processor(**proc_kwargs)
        if orig_tiles is not None:
            self.processor.image_processor.max_num_tiles = orig_tiles

        input_ids_np = (
            proc_output["input_ids"][0].numpy()
            if proc_output["input_ids"].dim() == 2
            else proc_output["input_ids"].numpy()
        )

        # 5. Insert audio tokens into input_ids after the last </img>
        #    Format: <so_start> + N × <so_embedding> + <so_end>
        if n_sound_tokens > 0:
            sound_id = self._sound_token_id
            so_start_id = self._tokenizer.convert_tokens_to_ids("<so_start>")
            so_end_id = self._tokenizer.convert_tokens_to_ids("<so_end>")
            img_end_id = self._tokenizer.convert_tokens_to_ids("</img>")

            img_end_positions = np.where(input_ids_np == img_end_id)[0]
            insert_pos = int(img_end_positions[-1]) + 1 if len(img_end_positions) > 0 else 1

            sound_block = np.array(
                [so_start_id] + [sound_id] * n_sound_tokens + [so_end_id],
                dtype=input_ids_np.dtype,
            )
            input_ids_np = np.concatenate([input_ids_np[:insert_pos], sound_block, input_ids_np[insert_pos:]])

        # 5b. Build loss mask FIRST on raw input_ids (before adjust_image_tokens),
        # then adjust image tokens. The loss mask positions must align with the
        # post-adjustment input_ids that the model receives.
        pv = proc_output.get("pixel_values")
        num_patches = None
        if pv is not None:
            num_tiles = pv.shape[0] if isinstance(pv, torch.Tensor) else len(pv)
            num_patches = torch.ones(num_tiles, dtype=torch.long)

        # 6. Build loss mask — only supervise assistant turns
        loss_mask_np = np.zeros(len(input_ids_np), dtype=np.float32)
        search_start = 0
        for turn in conversation:
            if turn["role"] == "assistant":
                answer = turn["content"] if isinstance(turn["content"], str) else turn["content"]
                if isinstance(answer, list):
                    answer = "".join(p.get("text", "") for p in answer if isinstance(p, dict))
                answer_tokens = self._tokenizer.encode(answer, add_special_tokens=False)
                ans_start, ans_end = find_pattern_indices(input_ids_np, answer_tokens, search_start)
                if ans_start >= 0:
                    loss_mask_np[ans_start:ans_end] = 1.0
                    search_start = ans_end

        # 7. Labels = left-shifted input_ids
        labels_np = np.full(len(input_ids_np), IGNORE_INDEX, dtype=np.int64)
        labels_np[:-1] = input_ids_np[1:]
        shifted_loss = np.zeros_like(loss_mask_np)
        shifted_loss[:-1] = loss_mask_np[1:]
        labels_np[shifted_loss == 0.0] = IGNORE_INDEX
        loss_mask_np = shifted_loss

        # 7b. Adjust image tokens — shrink many <image> tokens per tile to one.
        # This must happen AFTER loss mask and labels are built so positions stay aligned.
        img_start_id = self._tokenizer.convert_tokens_to_ids("<img>")
        img_end_id = self._tokenizer.convert_tokens_to_ids("</img>")
        if num_patches is not None and (input_ids_np == img_start_id).any():
            from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import adjust_image_tokens

            proc_num_patches = proc_output.get("num_patches")
            if proc_num_patches is not None:
                if not isinstance(proc_num_patches, torch.Tensor):
                    proc_num_patches = torch.tensor(proc_num_patches)
            else:
                proc_num_patches = num_patches

            # adjust_image_tokens can handle a dict of tensors with matching shapes
            adjusted = adjust_image_tokens(
                {
                    "input_ids": torch.from_numpy(input_ids_np).unsqueeze(0),
                    "labels": torch.from_numpy(labels_np).unsqueeze(0),
                    "loss_mask": torch.from_numpy(loss_mask_np).unsqueeze(0),
                },
                proc_num_patches,
                img_start_id,
                img_end_id,
            )
            input_ids_np = adjusted["input_ids"].squeeze(0).numpy()
            labels_np = adjusted["labels"].squeeze(0).numpy()
            loss_mask_np = adjusted["loss_mask"].squeeze(0).numpy()

        # 8. Truncate
        max_len = self.seq_length
        input_ids_np = input_ids_np[:max_len].copy()
        labels_np = labels_np[:max_len].copy()
        loss_mask_np = loss_mask_np[:max_len].copy()

        # 9. Collect visual tensors (num_patches already computed in step 5b)
        visual_tensors: Dict[str, torch.Tensor] = {}
        for key in self.visual_keys:
            val = proc_output.get(key)
            if val is not None:
                visual_tensors[key] = val if isinstance(val, torch.Tensor) else torch.tensor(val)

        # 10. Temporal video embedder: patchify ALL video frames and pack
        sample_imgs_sizes: Optional[torch.Tensor] = None
        sample_num_frames: Optional[torch.Tensor] = None
        if self.use_temporal_video_embedder and video_frames:
            all_patches = []
            target_h, target_w = 512, 512
            for frame in video_frames:
                patches = self._patchify_frame(frame, target_h, target_w)
                all_patches.append(patches)
            # Pack into [1, total_patches, C*P*P] for dynamic resolution
            packed = torch.cat(all_patches, dim=0).unsqueeze(0)  # [1, N*num_patches_per_frame, feat]
            visual_tensors["pixel_values"] = packed
            sample_imgs_sizes = torch.tensor([[target_h, target_w]] * len(video_frames), dtype=torch.long)
            sample_num_frames = torch.tensor([len(video_frames)], dtype=torch.long)
            # Also include standalone images (each is 1 frame)
            if images_pil:
                for img in images_pil:
                    img_patches = self._patchify_frame(img, target_h, target_w)
                    packed_img = img_patches.unsqueeze(0)
                    visual_tensors["pixel_values"] = torch.cat([packed_img, visual_tensors["pixel_values"]], dim=1)
                    sample_imgs_sizes = torch.cat(
                        [
                            torch.tensor([[target_h, target_w]], dtype=torch.long),
                            sample_imgs_sizes,
                        ],
                        dim=0,
                    )
                    sample_num_frames = torch.cat(
                        [
                            torch.tensor([1], dtype=torch.long),
                            sample_num_frames,
                        ]
                    )

        # Compute per-image num_image_tiles for LM-side image-token expansion
        # (new llava_model.py dynamic_resolution path). num_tiles_i = (H/P * W/P) // 4
        # matching the HF collate's shuffled_count computation.
        sample_num_image_tiles: Optional[torch.Tensor] = None
        if sample_imgs_sizes is not None:
            P = self.patch_dim
            sample_num_image_tiles = torch.tensor(
                [(int(h) // P) * (int(w) // P) // 4 for h, w in sample_imgs_sizes.tolist()],
                dtype=torch.int,
            )

        return NemotronOmniTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            input_ids=torch.from_numpy(input_ids_np),
            labels=torch.from_numpy(labels_np),
            loss_mask=torch.from_numpy(loss_mask_np),
            visual_tensors=visual_tensors,
            num_patches=num_patches,
            sound_clips=sound_clips_t,
            sound_length=sound_length_t,
            imgs_sizes=sample_imgs_sizes,
            num_frames=sample_num_frames,
            num_image_tiles=sample_num_image_tiles,
        )

    # ------------------------------------------------------------------
    # batch
    # ------------------------------------------------------------------

    def batch(self, samples: List[NemotronOmniTaskSample]) -> NemotronOmniTaskBatch:
        """Pad-and-collate (default) OR pack samples along the seq dim when
        ``pack_sequences=True``. Packing emits ``cu_seqlens`` / ``cu_seqlens_unpadded``
        / ``max_seqlen`` so TE's THD kernels handle cross-sample masking (and CP
        partitioning via ``thd_get_partitioned_indices``) without an attention mask.
        """
        pad_id = self._pad_token_id
        batch_size = len(samples)

        cu_seqlens_t: Optional[torch.Tensor] = None
        cu_seqlens_unpadded_t: Optional[torch.Tensor] = None
        cu_seqlens_argmin_t: Optional[torch.Tensor] = None
        max_seqlen_t: Optional[torch.Tensor] = None

        if self.pack_sequences:
            # Concatenate samples along the seq dim into a single [1, total_len]
            # microbatch. TE attention kernels use cu_seqlens for per-sample
            # masking; no attention_mask needed.
            lengths = [int(s.input_ids.size(0)) for s in samples]
            total_len = sum(lengths)
            cu_seqlens = [0]
            for L in lengths:
                cu_seqlens.append(cu_seqlens[-1] + L)

            tokens_flat = torch.cat([s.input_ids for s in samples], dim=0)
            labels_flat = torch.cat([s.labels for s in samples], dim=0)
            loss_mask_flat = torch.cat([s.loss_mask for s in samples], dim=0)
            # Per-sample resetting position ids: [0..L1-1, 0..L2-1, ...]
            position_ids_flat = torch.cat(
                [torch.arange(L, dtype=torch.long) for L in lengths], dim=0
            )

            tokens = tokens_flat.unsqueeze(0)
            tokens[tokens == pad_id] = 0
            labels = labels_flat.unsqueeze(0)
            loss_mask_t = loss_mask_flat.unsqueeze(0)
            position_ids = position_ids_flat.unsqueeze(0)
            attention_mask = None  # TE derives the causal+padding mask from cu_seqlens.

            cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32)
            cu_seqlens_unpadded_t = cu_seqlens_t.clone()
            # get_packed_seq_params truncates cu_seqlens_padded[: argmin.item()]; the
            # trick in the fixed-size-batched case is sentinel=-1 padding with argmin
            # pointing at the first sentinel. Here we emit an unpadded cu_seqlens and
            # set argmin = len(cu_seqlens) so the slice is a no-op (keeps every entry).
            cu_seqlens_argmin_t = torch.tensor(len(cu_seqlens), dtype=torch.int32)
            max_seqlen_t = torch.tensor(max(lengths), dtype=torch.int32)
        else:
            max_seq_len = max(s.input_ids.size(0) for s in samples)
            input_ids_mat = np.full((batch_size, max_seq_len), pad_id, dtype=np.int64)
            labels_mat = np.full((batch_size, max_seq_len), IGNORE_INDEX, dtype=np.int64)
            loss_mask_mat = np.zeros((batch_size, max_seq_len), dtype=np.float32)

            for i, s in enumerate(samples):
                seq_len = min(max_seq_len, s.input_ids.size(0))
                input_ids_mat[i, :seq_len] = s.input_ids.numpy()[:seq_len]
                labels_mat[i, :seq_len] = s.labels.numpy()[:seq_len]
                loss_mask_mat[i, :seq_len] = s.loss_mask.numpy()[:seq_len]

            tokens = torch.from_numpy(input_ids_mat)
            tokens[tokens == pad_id] = 0
            labels = torch.from_numpy(labels_mat)
            loss_mask_t = torch.from_numpy(loss_mask_mat)

            attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                data=tokens,
                eod_token=self._eos_token_id,
                eod_mask_loss=False,
                reset_attention_mask=False,
                reset_position_ids=False,
            )

        # Aggregate visual tensors.
        # The temporal video path ships pixel_values as [1, N_i*patches_per_frame, feat]
        # per sample. When packing, concat along dim=1 so the whole microbatch becomes a
        # single [1, total_patches, feat] packed sequence that matches vision_packed_seq_params
        # cu_seqlens. Without packing, preserve the legacy [MBS, patches, feat] stack.
        all_visual_keys = set()
        for s in samples:
            all_visual_keys.update(s.visual_tensors.keys())
        batched_visual: Dict[str, Optional[torch.Tensor]] = {}
        for key in all_visual_keys:
            tensors = [s.visual_tensors[key] for s in samples if key in s.visual_tensors]
            if not tensors:
                batched_visual[key] = None
                continue
            if self.pack_sequences and tensors[0].dim() == 3:
                batched_visual[key] = torch.cat(tensors, dim=1)
            else:
                batched_visual[key] = torch.cat(tensors, dim=0)

        # Aggregate audio: pad mel spectrograms to max length in batch
        has_audio = any(s.sound_clips is not None for s in samples)
        sound_clips_batch: Optional[torch.Tensor] = None
        sound_length_batch: Optional[torch.Tensor] = None

        if has_audio:
            mel_list = []
            mel_lengths = []
            for s in samples:
                if s.sound_clips is not None:
                    mel_list.append(s.sound_clips)
                    mel_lengths.append(s.sound_clips.shape[0])
                else:
                    mel_list.append(torch.zeros(1, self.num_mel_bins))
                    mel_lengths.append(1)

            max_mel_len = max(mel_lengths)
            mel_dim = mel_list[0].shape[-1]
            sound_clips_batch = torch.zeros(batch_size, max_mel_len, mel_dim)
            for i, mel in enumerate(mel_list):
                sound_clips_batch[i, : mel.shape[0]] = mel
            sound_length_batch = torch.tensor(mel_lengths, dtype=torch.long)

        # Aggregate num_patches
        all_patches = [s.num_patches for s in samples if s.num_patches is not None]
        num_patches_batch = torch.cat(all_patches, dim=0) if all_patches else None

        # Aggregate imgs_sizes / num_frames (temporal video embedder)
        has_temporal = any(s.imgs_sizes is not None for s in samples)
        imgs_sizes_batch: Optional[torch.Tensor] = None
        num_frames_batch: Optional[torch.Tensor] = None
        if has_temporal:
            all_imgs_sizes = [s.imgs_sizes for s in samples if s.imgs_sizes is not None]
            all_num_frames = [s.num_frames for s in samples if s.num_frames is not None]
            imgs_sizes_batch = torch.cat(all_imgs_sizes, dim=0) if all_imgs_sizes else None
            num_frames_batch = torch.cat(all_num_frames, dim=0) if all_num_frames else None
        all_num_image_tiles = [s.num_image_tiles for s in samples if s.num_image_tiles is not None]
        num_image_tiles_batch = torch.cat(all_num_image_tiles, dim=0) if all_num_image_tiles else None

        batch_kwargs: Dict = dict(
            __keys__=[s.__key__ for s in samples],
            __subflavors__=[s.__subflavors__ for s in samples],
            input_ids=tokens,
            labels=labels,
            loss_mask=loss_mask_t,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_tensors=batched_visual,
            num_patches=num_patches_batch,
            sound_clips=sound_clips_batch,
            sound_length=sound_length_batch,
            imgs_sizes=imgs_sizes_batch,
            num_frames=num_frames_batch,
            num_image_tiles=num_image_tiles_batch,
            cu_seqlens=cu_seqlens_t,
            cu_seqlens_unpadded=cu_seqlens_unpadded_t,
            cu_seqlens_argmin=cu_seqlens_argmin_t,
            max_seqlen=max_seqlen_t,
        )
        _batch_fields = {f.name for f in dataclasses.fields(NemotronOmniTaskBatch)}
        if "__key__" in _batch_fields:
            batch_kwargs["__key__"] = samples[0].__key__
        if "__restore_key__" in _batch_fields:
            batch_kwargs["__restore_key__"] = ()

        return NemotronOmniTaskBatch(**batch_kwargs)

    # ------------------------------------------------------------------
    # encode_batch
    # ------------------------------------------------------------------

    def encode_batch(self, batch: NemotronOmniTaskBatch) -> dict:
        """Convert batch to dict for the training step."""
        raw = {
            "tokens": batch.input_ids,
            "labels": batch.labels,
            "loss_mask": batch.loss_mask,
            "attention_mask": batch.attention_mask,
            "position_ids": batch.position_ids,
            "num_patches": batch.num_patches,
            "sound_clips": batch.sound_clips,
            "sound_length": batch.sound_length,
            "imgs_sizes": batch.imgs_sizes,
            "num_frames": batch.num_frames,
            "num_image_tiles": batch.num_image_tiles,
        }

        vt = batch.visual_tensors if batch.visual_tensors else {}
        raw["visual_inputs"] = GenericVisualInputs(**{k: v for k, v in vt.items() if v is not None})

        # Keep sound_clips / sound_length as top-level batch keys
        # (nemotron_omni_step picks them up directly)
        return raw
