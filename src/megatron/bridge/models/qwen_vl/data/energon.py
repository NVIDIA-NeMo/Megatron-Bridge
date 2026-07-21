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

import dataclasses
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from megatron.energon import Batch, DefaultTaskEncoder, SkipSample

from megatron.bridge.data.conversation_processing import (
    normalize_energon_vlm_sample,
    normalized_vlm_sample_to_hf_example,
)
from megatron.bridge.data.energon.metadata import batch_metadata_kwargs
from megatron.bridge.data.energon.task_encoder_utils import (
    ChatMLSample,
    ChatMLWebdataset,  # noqa: F401  -- re-exported for backward compat
    _images_to_pil,  # noqa: F401  -- re-exported for backward compat
    _tensor_to_pil,  # noqa: F401  -- re-exported for backward compat
    _videos_to_pil,  # noqa: F401  -- re-exported for backward compat
    cook_chatml_sample,  # noqa: F401  -- re-exported for backward compat
    find_pattern_indices,  # noqa: F401  -- re-exported for backward compat
    get_ltor_masks_and_position_ids,  # noqa: F401  -- re-exported for backward compat
    videohandler,  # noqa: F401  -- re-exported for backward compat
)
from megatron.bridge.models.qwen_vl.data.collate_fn import qwen2_5_collate_fn
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


def process_vision(
    processor, images, videos, fps=None, model_version: str = "qwen-vl", min_pixels=None, max_pixels=None
):
    """Minimal vision preprocessing wrapper using the provided processor (e.g., HF AutoProcessor)."""
    if images is not None:
        kwargs = {}
        if min_pixels is not None:
            kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            kwargs["max_pixels"] = max_pixels
        image_inputs = processor(images=images, text="", videos=None, return_tensors="pt", **kwargs)
        image_grid_thw = image_inputs.get("image_grid_thw", None)
    else:
        image_inputs = {}
        image_grid_thw = None

    if videos is not None:
        # Pre-decoded frames from WDS are already at the desired sampling rate.
        # do_sample_frames=False prevents the processor from re-sampling them under
        # a spurious 24 fps assumption, which would reduce most clips to T=2.
        videos_inputs = processor.video_processor(videos=videos, return_tensors="pt", do_sample_frames=False)
        video_grid_thw = videos_inputs.get("video_grid_thw", None)
    else:
        videos_inputs = {}
        video_grid_thw = None

    return {
        "image_inputs": image_inputs,
        "image_grid_thw": image_grid_thw,
        "video_inputs": videos_inputs,
        "video_grid_thw": video_grid_thw,
    }


def _resolve_hf_mm_token_ids(hf_tokenizer):
    """Resolve HF tokenizer ids for <image> and <video> tokens without nemo constants."""

    def _get(token_str: str, default_id: int) -> int:
        token_attr = getattr(hf_tokenizer, f"{token_str.strip('<>')}_token_id", None)
        if token_attr is not None:
            return int(token_attr)
        try:
            return int(hf_tokenizer.convert_tokens_to_ids(token_str))
        except Exception:
            return default_id

    image_id = _get("<image>", 151655)
    video_id = _get("<video>", 151656)
    return image_id, video_id


def _visual_token_count(grid_thw: Any, merge_size: int) -> int:
    """Return merged visual token count for THW grid metadata."""
    if grid_thw is None:
        return 0
    grid = torch.as_tensor(grid_thw)
    if grid.numel() == 0:
        return 0
    return int(grid.prod(dim=-1).sum().item()) // (merge_size**2)


@dataclass
class QwenVLTaskSample:
    """HF-style Qwen VLM sample produced from an Energon ``ChatMLSample``.

    Expected input format:
        Produced by ``QwenVLTaskEncoder.encode_sample`` from an Energon
        ``ChatMLSample``.  ``example`` follows the HF VLM collate schema:
        ``{"conversation": [{"role": ..., "content": [...]}, ...]}`` with
        inline ``{"type": "image"|"video", ...}`` media parts.

    Output format:
        Consumed by ``QwenVLTaskEncoder.batch``, which passes the ``example``
        dictionaries to the same Qwen collate function used by HF-style VLM
        datasets.
    """

    __key__: str
    __subflavors__: Dict
    example: Dict[str, Any]


@dataclass
class QwenVLTaskBatch(Batch):
    """Batched Qwen VLM tensors produced by the shared HF collate function."""

    __keys__: List[str]
    __subflavors__: List[Dict]
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    visual_inputs: GenericVisualInputs | None
    attention_mask: torch.Tensor | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_kv: torch.Tensor | None = None
    cu_seqlens_q_padded: torch.Tensor | None = None
    cu_seqlens_kv_padded: torch.Tensor | None = None
    max_seqlen_q: torch.Tensor | None = None
    max_seqlen_kv: torch.Tensor | None = None


def convert_to_qwenvl_content(user_input: str, image_pattern: str = "<image>", video_pattern: str = "<video>"):
    """Split user input into format QwenVL tokenizer accepts."""

    pattern = r"({image}|{video})".format(image=image_pattern, video=video_pattern)
    contents = []
    cur = 0
    mm_idx = defaultdict(int)
    for matched in re.finditer(pattern, user_input):
        start, end = matched.span()
        if start > cur:
            contents.append({"type": "text", "text": user_input[cur:start].strip(" ")})

        contents.append(
            {
                "type": matched.string[start:end][1:-1],
                matched.string[start:end][1:-1]: str(mm_idx[matched.string[start:end][1:-1]]),
            }
        )

        cur = end
        mm_idx[matched.string[start:end][1:-1]] += 1

    if cur < len(user_input):
        contents.append({"type": "text", "text": user_input[cur : len(user_input)].strip(" ")})

    return contents


class QwenVLTaskEncoder(DefaultTaskEncoder[ChatMLSample, QwenVLTaskSample, QwenVLTaskBatch, dict]):
    """Energon task encoder for Qwen VL samples.

    Args:
        tokenizer: HF tokenizer for resolving multimodal token ids.
        image_processor: HF processor passed to the Qwen collate function.
        temporal_patch_size: Temporal patch size used for video token accounting.
        spatial_merge_size: Spatial merge size used for visual token accounting.
        patch_size: Vision patch size.
        max_padding_length: Maximum sequence length accepted after collation.
        min_pixels: Minimum pixel constraint forwarded to Qwen vision processing.
        max_pixels: Maximum pixel constraint forwarded to Qwen vision processing.
        max_num_images: Optional per-sample image count limit.
        max_num_frames: Optional per-video frame limit.
        max_visual_tokens: Optional per-sample visual token budget.
        pad_to_max_length: Whether collate-time padding should pad non-packed
            batches to ``max_padding_length``.
        pad_to_multiple_of: Non-packed collate-time padding multiple used when
            ``pad_to_max_length`` is false.
        enable_in_batch_packing: Whether the Qwen collate should do in-batch
            sequence packing.
        in_batch_packing_pad_to_multiple_of: Per-sample padding multiple used
            only by the in-batch packed path, typically to satisfy CP/SP
            divisibility.
    """

    def __init__(
        self,
        tokenizer,
        image_processor,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        patch_size: int = 14,
        max_padding_length: int = 4096,
        min_pixels: int = 200704,
        max_pixels: int = 1003520,
        max_num_images: int | None = 10,
        max_num_frames: int | None = 60,
        max_visual_tokens: int | None = 16384,
        pad_to_max_length: bool = False,
        pad_to_multiple_of: int = 128,
        enable_in_batch_packing: bool = False,
        in_batch_packing_pad_to_multiple_of: int = 1,
    ):
        super().__init__()

        self.hf_tokenizer = tokenizer
        self.image_processor = image_processor
        self.seq_length = max_padding_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_num_images = max_num_images
        self.max_num_frames = max_num_frames
        self.max_visual_tokens = max_visual_tokens
        self.pad_to_max_length = pad_to_max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.enable_in_batch_packing = enable_in_batch_packing
        self.in_batch_packing_pad_to_multiple_of = in_batch_packing_pad_to_multiple_of

        self.temporal_patch_size = temporal_patch_size
        self.merge_size = spatial_merge_size
        self.patch_size = patch_size

        self.seq_len = max_padding_length
        self.image_token_id, self.video_token_id = _resolve_hf_mm_token_ids(self.hf_tokenizer)

    def encode_sample(self, sample: ChatMLSample):
        """Normalize one Energon sample into the HF-style Qwen collate schema.

        Expected input format:
            ``sample`` is an Energon ``ChatMLSample`` with JSON
            ``conversation`` and optional decoded ``imgs`` / ``videos`` media
            payloads.

        Output format:
            Returns ``QwenVLTaskSample`` whose ``example`` is a HF-style VLM
            collate dictionary:
            ``{"conversation": [{"role": ..., "content": [{"type": ...}, ...]}]}``.
            Tokenization, image/video preprocessing, labels, and loss masks are
            intentionally deferred to ``self.collate_fn`` so HF and Energon data
            paths share the same Qwen model processing.
        """
        normalized_sample = normalize_energon_vlm_sample(sample)
        imgs_for_processing = normalized_sample.images
        videos_for_processing = normalized_sample.videos

        if self.max_num_images is not None and imgs_for_processing is not None:
            if len(imgs_for_processing) > self.max_num_images:
                logging.warning(
                    "Skipping sample %s: %d images exceeds max_num_images=%d",
                    sample.__key__,
                    len(imgs_for_processing),
                    self.max_num_images,
                )
                raise SkipSample()

        if self.max_num_frames is not None and videos_for_processing is not None:
            clipped = []
            for v in videos_for_processing:
                if len(v) > self.max_num_frames:
                    logging.warning(
                        "Truncating %d frames to max_num_frames=%d for sample %s",
                        len(v),
                        self.max_num_frames,
                        sample.__key__,
                    )
                    clipped.append(v[: self.max_num_frames])
                else:
                    clipped.append(v)
            videos_for_processing = clipped

        if self.max_visual_tokens is not None and (
            imgs_for_processing is not None or videos_for_processing is not None
        ):
            processed_vision = process_vision(
                self.image_processor,
                imgs_for_processing,
                videos_for_processing,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            image_tokens = _visual_token_count(processed_vision["image_grid_thw"], self.merge_size)
            video_tokens = _visual_token_count(processed_vision["video_grid_thw"], self.merge_size)
            total_visual_tokens = image_tokens + video_tokens
            if total_visual_tokens > self.max_visual_tokens:
                logging.warning(
                    "Skipping sample %s: %d visual tokens exceeds max_visual_tokens=%d",
                    sample.__key__,
                    total_visual_tokens,
                    self.max_visual_tokens,
                )
                raise SkipSample()

        normalized_sample = dataclasses.replace(
            normalized_sample,
            images=imgs_for_processing,
            videos=videos_for_processing,
        )
        example = normalized_vlm_sample_to_hf_example(normalized_sample, media_first=True)
        return QwenVLTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            example=example,
        )

    def collate_fn(self, examples: List[Dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate Qwen HF-style examples with the shared HF dataset collator.

        Expected input format:
            ``examples`` is a list of dictionaries in the same schema returned
            by HF-style VLM datasets: each item has ``conversation`` messages
            with multimodal ``content`` parts.

        Output format:
            Returns the exact dictionary produced by ``qwen2_5_collate_fn``:
            ``input_ids``, ``labels``, ``loss_mask``, ``position_ids``, optional
            ``attention_mask``, and ``visual_inputs``.
        """
        return qwen2_5_collate_fn(
            examples,
            self.image_processor,
            sequence_length=self.seq_length,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            require_assistant_matches=True,
            pad_to_max_length=self.pad_to_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            enable_in_batch_packing=self.enable_in_batch_packing,
            in_batch_packing_pad_to_multiple_of=self.in_batch_packing_pad_to_multiple_of,
        )

    def batch(self, samples: List[QwenVLTaskSample]) -> QwenVLTaskBatch:
        """Collate normalized Energon samples with the shared Qwen HF collator.

        Expected input format:
            ``samples`` are ``QwenVLTaskSample`` objects whose ``example`` fields
            follow the HF VLM collate schema.

        Output format:
            Returns ``QwenVLTaskBatch`` carrying the same tensors as
            ``qwen2_5_collate_fn`` plus Energon batch metadata.
        """
        examples = [sample.example for sample in samples]
        collated = self.collate_fn(examples)

        if collated["input_ids"].shape[1] > self.seq_len:
            logging.warning("max sequence length larger than passed parameter")

        keys = [s.__key__ for s in samples]
        return QwenVLTaskBatch(
            **batch_metadata_kwargs(keys=keys),
            __keys__=keys,
            __subflavors__=[s.__subflavors__ for s in samples],
            input_ids=collated["input_ids"],
            attention_mask=collated.get("attention_mask"),
            position_ids=collated["position_ids"],
            labels=collated["labels"],
            loss_mask=collated["loss_mask"],
            visual_inputs=collated.get("visual_inputs"),
            cu_seqlens_q=collated.get("cu_seqlens_q"),
            cu_seqlens_kv=collated.get("cu_seqlens_kv"),
            cu_seqlens_q_padded=collated.get("cu_seqlens_q_padded"),
            cu_seqlens_kv_padded=collated.get("cu_seqlens_kv_padded"),
            max_seqlen_q=collated.get("max_seqlen_q"),
            max_seqlen_kv=collated.get("max_seqlen_kv"),
        )

    def encode_batch(self, batch: QwenVLTaskBatch) -> dict:
        """Encode batch in dict"""

        raw = {field.name: getattr(batch, field.name) for field in dataclasses.fields(batch)}
        del raw["__subflavors__"]

        return raw
