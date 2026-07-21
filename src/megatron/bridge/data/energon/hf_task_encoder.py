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

"""Generic HF VLM task encoder for Energon dataloading.

Normalizes Energon ``ChatMLSample`` objects into HF-style multimodal examples
and delegates tokenization, vision preprocessing, masking, and padding to the
selected HF VLM collate function.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from megatron.energon import Batch, DefaultTaskEncoder

from megatron.bridge.data.collators.registry import resolve_model_collate
from megatron.bridge.data.conversation_processing import (
    normalize_energon_vlm_sample,
    normalized_vlm_sample_to_hf_example,
)
from megatron.bridge.data.energon.metadata import batch_metadata_kwargs
from megatron.bridge.data.energon.task_encoder_utils import (
    ChatMLSample,
)
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


@dataclass
class HFEnergonSample:
    """HF-style VLM example produced from an Energon ``ChatMLSample``."""

    __key__: str
    __subflavors__: Dict
    example: Dict[str, Any]


@dataclass
class HFEnergonBatch(Batch):
    """Batched format for a generic HF VLM."""

    __keys__: List[str] = field(default_factory=list)
    __subflavors__: List[Dict] = field(default_factory=list)
    input_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0))  # [B, seq_len]
    labels: torch.Tensor = field(default_factory=lambda: torch.empty(0))  # [B, seq_len]
    loss_mask: torch.Tensor = field(default_factory=lambda: torch.empty(0))  # [B, seq_len]
    position_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0))  # [B, seq_len]
    visual_inputs: GenericVisualInputs | None = None
    attention_mask: torch.Tensor | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_kv: torch.Tensor | None = None
    cu_seqlens_q_padded: torch.Tensor | None = None
    cu_seqlens_kv_padded: torch.Tensor | None = None
    max_seqlen_q: torch.Tensor | None = None
    max_seqlen_kv: torch.Tensor | None = None
    total_tokens: int | None = None


class HFTaskEncoder(DefaultTaskEncoder[ChatMLSample, HFEnergonSample, HFEnergonBatch, dict]):
    """Task encoder for HF VLMs that rely on ``processor()`` for tokenization + vision.

    Args:
        processor: HF ``AutoProcessor`` instance passed to the selected collate
            function.
        seq_length: Maximum sequence length accepted after collation.
        visual_keys: Processor output keys to retain when the selected collate
            function supports configurable visual input selection.
        min_pixels: Optional min pixel constraint forwarded when supported by
            the selected collate function.
        max_pixels: Optional max pixel constraint forwarded when supported by
            the selected collate function.
        collate_fn: Optional collate implementation override. If omitted, the
            implementation is selected from the processor type.
        pad_to_max_length: Whether collate-time padding should pad non-packed
            batches to ``seq_length`` when the selected collate supports it.
        pad_to_multiple_of: Non-packed collate-time padding multiple used when
            ``pad_to_max_length`` is false and the selected collate supports it.
        enable_in_batch_packing: Whether the selected collate should do
            in-batch sequence packing.
        in_batch_packing_pad_to_multiple_of: Per-sample padding multiple used
            only by the in-batch packed path, typically to satisfy CP/SP
            divisibility.
    """

    def __init__(
        self,
        processor,
        seq_length: int = 4096,
        visual_keys: Sequence[str] = ("pixel_values",),
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        collate_fn: Callable[..., dict[str, Any]] | None = None,
        pad_to_max_length: bool = False,
        pad_to_multiple_of: int = 128,
        enable_in_batch_packing: bool = False,
        in_batch_packing_pad_to_multiple_of: int = 1,
    ):
        super().__init__()
        self.processor = processor
        self.seq_length = seq_length
        self.visual_keys: Tuple[str, ...] = tuple(visual_keys)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.pad_to_max_length = pad_to_max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.enable_in_batch_packing = enable_in_batch_packing
        self.in_batch_packing_pad_to_multiple_of = in_batch_packing_pad_to_multiple_of
        collate_key = type(processor).__name__ if processor is not None else "default"
        if collate_fn is not None:
            self._collate_impl = collate_fn
        else:
            self._collate_impl = resolve_model_collate(collate_key)

    def encode_sample(self, sample: ChatMLSample) -> HFEnergonSample:
        """Normalize a single ChatML sample into a HF-style collate example.

        Expected input format:
            ``sample`` is an Energon ``ChatMLSample`` with JSON string
            ``conversation`` plus optional WDS-decoded ``imgs`` and ``videos``.

        Output format:
            Returns ``HFEnergonSample`` whose ``example`` follows the same
            dictionary schema consumed by HF VLM dataset collate functions.
            Tokenization, processor calls, label construction, and visual tensor
            batching are deferred to ``self.collate_fn``.
        """
        normalized_sample = normalize_energon_vlm_sample(sample)
        example = normalized_vlm_sample_to_hf_example(normalized_sample)

        return HFEnergonSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            example=example,
        )

    def collate_fn(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate HF-style examples with this encoder's model collator.

        Expected input format:
            List of HF-style VLM example dictionaries with ``conversation`` and
            optional modality fields.

        Output format:
            The exact batch dictionary returned by the selected HF collate
            function for this processor type.
        """
        return self._collate_impl(
            examples,
            self.processor,
            visual_keys=self.visual_keys,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            sequence_length=self.seq_length,
            pad_to_max_length=self.pad_to_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            enable_in_batch_packing=self.enable_in_batch_packing,
            in_batch_packing_pad_to_multiple_of=self.in_batch_packing_pad_to_multiple_of,
        )

    # ------------------------------------------------------------------
    # batch
    # ------------------------------------------------------------------

    def _collate_batch_kwargs(self, samples: List[HFEnergonSample]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Collate samples and build kwargs shared by generic/model batches."""
        examples = [sample.example for sample in samples]
        collated = self.collate_fn(examples)
        collated_seq_len = (
            int(collated["max_seqlen_q"].max().item())
            if collated.get("max_seqlen_q") is not None
            else collated["input_ids"].shape[1]
        )
        if collated_seq_len > self.seq_length:
            raise ValueError(
                f"Collated seq_len {collated_seq_len} exceeds seq_length {self.seq_length}. "
                "The selected HF VLM collator must enforce seq_length while preserving visual metadata."
            )

        keys = [s.__key__ for s in samples]
        batch_kwargs: Dict = dict(
            **batch_metadata_kwargs(keys=keys),
            __keys__=keys,
            __subflavors__=[s.__subflavors__ for s in samples],
            input_ids=collated["input_ids"],
            labels=collated["labels"],
            loss_mask=collated["loss_mask"],
            attention_mask=collated.get("attention_mask"),
            position_ids=collated["position_ids"],
            visual_inputs=collated.get("visual_inputs"),
            cu_seqlens_q=collated.get("cu_seqlens_q"),
            cu_seqlens_kv=collated.get("cu_seqlens_kv"),
            cu_seqlens_q_padded=collated.get("cu_seqlens_q_padded"),
            cu_seqlens_kv_padded=collated.get("cu_seqlens_kv_padded"),
            max_seqlen_q=collated.get("max_seqlen_q"),
            max_seqlen_kv=collated.get("max_seqlen_kv"),
            total_tokens=collated.get("total_tokens"),
        )
        return collated, batch_kwargs

    def batch(self, samples: List[HFEnergonSample]) -> HFEnergonBatch:
        """Collate normalized samples with the selected HF VLM collator."""
        _, batch_kwargs = self._collate_batch_kwargs(samples)
        return HFEnergonBatch(**batch_kwargs)

    # ------------------------------------------------------------------
    # encode_batch
    # ------------------------------------------------------------------

    def encode_batch(self, batch: HFEnergonBatch) -> dict:
        """Convert batch dataclass to dict without expanding ``visual_inputs``."""
        raw = {field.name: getattr(batch, field.name) for field in dataclasses.fields(batch)}

        # Remove Batch base-class metadata not needed downstream
        for meta_key in ("__key__", "__keys__", "__restore_key__", "__subflavors__", "__sources__"):
            raw.pop(meta_key, None)

        return raw
