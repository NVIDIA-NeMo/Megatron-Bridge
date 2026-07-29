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

"""mbridge ``DatasetProvider`` that exposes the Flickr8k → packed-sample
pipeline for Step3.7 SFT.

This is the single integration point between the data primitives
(template, dataset, samplers, packed dataloader, pack transform) and
Megatron-Bridge's ``setup`` → ``build_pretraining_data_loader`` flow.

What this provider does (deterministic):

  1. Download ``intro/flickr8k`` train CSV + per-row JPGs via
     ``huggingface_hub`` (sync, no async wrapping).
  2. Build :class:`Step37Flickr8kDataset` with a fresh
     :class:`Step37MultimodalTemplate` (loaded from
     ``tokenizer_path`` with ``trust_remote_code=False``).
  3. Build a sync :class:`MixedPackedDataloader` that probes every sample
     for its NTP length, runs the weighted in/cross-domain samplers, then
     runs ``non_truncation.pack(max_len=...)``.
  4. Return the packed dataloader as a map-style ``Dataset`` so mbridge's
     ``MegatronPretrainingSampler`` can drive it. Validation / test
     splits are skipped (Flickr8k only has a train split here).

The collate is the identity — ``MixedPackedDataloader[idx]`` already
returns the packed dict; we just unwrap the singleton mini-batch list.
The downstream forward step (``step37_flickr8k_step``) does the GPU
move + image loading via :func:`preprocess_packed_batch`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple

import torch

from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider
from megatron.bridge.models.stepfun.data.flickr8k.flickr8k_loader import (
    Step37Flickr8kDataset,
    prepare_flickr8k_samples,
)
from megatron.bridge.models.stepfun.data.flickr8k.pack_transform import pack_samples
from megatron.bridge.models.stepfun.data.flickr8k.packed_dataloader import MixedPackedDataloader
from megatron.bridge.models.stepfun.data.flickr8k.template import (
    IMAGE_END_TOKEN,
    IMAGE_START_TOKEN,
    IMAGE_TOKEN,
    IMAGE_TOKEN_COUNT,
    PATCH_END_TOKEN,
    PATCH_START_TOKEN,
    PATCH_TOKEN_COUNT,
    Step37MultimodalTemplate,
)


class _FixedPackDataset(torch.utils.data.Dataset):
    """Pin every ``__getitem__`` to the same pack, regardless of ``idx``.

    Wraps a :class:`MixedPackedDataloader` so the Megatron sampler can hand
    out arbitrary indices on every DP rank, every step, and they all map to
    pack ``fixed_idx``. ``__len__`` is reported as a large sentinel
    (``_SENTINEL_LEN``) because mbridge size-checks ``len(dataset)`` against
    ``global_batch_size × train_iters``.
    """

    _SENTINEL_LEN = 10_000_000

    def __init__(self, inner, fixed_idx: int) -> None:
        super().__init__()
        self._inner = inner
        self._fixed_idx = fixed_idx % len(inner)
        collate_fn = getattr(inner, "collate_fn", None)
        if collate_fn is not None:
            self.collate_fn = collate_fn  # type: ignore[attr-defined]

    def __len__(self) -> int:
        return self._SENTINEL_LEN

    def __getitem__(self, idx: int):
        return self._inner[self._fixed_idx]


@dataclass(kw_only=True)
class Step37Flickr8kSFTDataProvider(DatasetProvider):
    """Step3.7 Flickr8k SFT dataset provider.

    Set ``cfg.dataset = Step37Flickr8kSFTDataProvider(...)`` on a Step3.7
    SFT recipe to swap the default CORD-V2 path for Flickr8k packing. Use
    ``step37_flickr8k_step`` as the forward step so the per-step
    ``preprocess`` loads images + builds ``ImageForInsert``.

    Note: ``trust_remote_code`` is forced ``False`` for the tokenizer
    load. We never instantiate any HF custom Python code.
    """

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer_path: str
    """Local HF snapshot path with ``tokenizer.json`` + ``chat_template.jinja``."""

    # ── Flickr8k loader ───────────────────────────────────────────────────
    repo_id: str = "intro/flickr8k"
    split: str = "train"
    sample_count: Optional[int] = 8
    """Take only the first N samples — default ``8`` for a smoke run.

    The full Flickr8k train split is ~6000 image+caption pairs (~1 GB
    of jpgs); leaving this at ``None`` triggers a full
    ``hf_hub_download`` of every row, which takes 10+ minutes on a cold
    cache and is almost never what a user wants. Set explicitly to
    ``None`` from a recipe / CLI override to opt into the full
    dataset."""
    caption_key: str = "caption_0"
    cache_dir: str = ".cache/step37_flickr8k"
    prompt: str = "Describe this image in one sentence."

    # ── Template / image-placeholder ───────────────────────────────────────
    image_token_count: int = IMAGE_TOKEN_COUNT
    patch_token_count: int = PATCH_TOKEN_COUNT
    image_token: str = IMAGE_TOKEN
    image_start_token: str = IMAGE_START_TOKEN
    image_end_token: str = IMAGE_END_TOKEN
    patch_start_token: str = PATCH_START_TOKEN
    patch_end_token: str = PATCH_END_TOKEN

    # ── Packing ───────────────────────────────────────────────────────────
    max_packing_seqlen: int = 2048
    """Max number of NTP-length tokens per pack."""
    seqlen_divisible_by: int = 64
    oversize_policy: Literal["drop", "extend"] = "drop"
    dataset_sampling: Literal["sequential", "random"] = "random"

    fixed_pack_idx: Optional[int] = None
    """If set, ``__getitem__`` always returns the pack at this index, ignoring
    the requested ``idx``. Used by the smoke recipe to feed identical input
    to every DP rank on every iteration (deterministic single-pack overfit).
    ``__len__`` is reported as a large sentinel so the Megatron sampler can
    request any index without IndexError. Leave ``None`` for normal
    training."""

    # ── Per-step preprocess metadata ─────────────────────────────────────
    img_start_token_id: int = -1
    """Tokenizer id for ``<im_start>``. Resolved at build time from the
    actual tokenizer if left at the sentinel ``-1``."""
    patch_start_token_id: int = -1
    """Tokenizer id for ``<patch_start>``. Same sentinel rule."""
    image_size: int = 728
    patch_image_size: int = 504
    encoder_patch_size: int = 14

    # ── mbridge framework defaults ───────────────────────────────────────
    seq_length: int = 2048
    dataloader_type: Optional[Literal["single", "cyclic", "external"]] = "single"
    skip_getting_attention_mask_from_dataset: bool = True
    global_data_keys: list = field(default_factory=lambda: ["cu_seqlens", "position_id"])
    """Batch keys broadcast to every PP rank (PP > 0 needs cu_seqlens /
    position_id even though ``input_ids`` / ``images`` are only on PP rank 0)."""

    def __post_init__(self):  # type: ignore[override]
        # mbridge DataloaderConfig has its own __post_init__; fall through.
        super_post = getattr(super(), "__post_init__", None)
        if super_post is not None:
            super_post()

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _make_template(self) -> Step37MultimodalTemplate:
        return Step37MultimodalTemplate(
            tokenizer_path=self.tokenizer_path,
            image_token_count=self.image_token_count,
            patch_token_count=self.patch_token_count,
            image_token=self.image_token,
            image_start_token=self.image_start_token,
            image_end_token=self.image_end_token,
            patch_start_token=self.patch_start_token,
            patch_end_token=self.patch_end_token,
            max_sequence_length=self.max_packing_seqlen,
        )

    def _resolve_special_token_ids(self, template: Step37MultimodalTemplate) -> None:
        """Fill in ``img_start_token_id`` / ``patch_start_token_id`` from
        the tokenizer if the user left them at the sentinel value.
        """
        tok = template.tokenizer
        if self.img_start_token_id < 0:
            self.img_start_token_id = int(tok.convert_tokens_to_ids(self.image_start_token))
        if self.patch_start_token_id < 0:
            self.patch_start_token_id = int(tok.convert_tokens_to_ids(self.patch_start_token))

    def _build_train_packed_dataloader(self) -> MixedPackedDataloader:
        samples = prepare_flickr8k_samples(
            repo_id=self.repo_id,
            split=self.split,
            sample_count=self.sample_count,
            caption_key=self.caption_key,
            cache_dir=self.cache_dir,
        )
        template = self._make_template()
        self._resolve_special_token_ids(template)
        dataset = Step37Flickr8kDataset(
            samples=samples,
            template=template,
            prompt=self.prompt,
        )

        def _pack(pieces):
            return pack_samples(pieces, seqlen_divisible_by=self.seqlen_divisible_by)

        return MixedPackedDataloader(
            datasets=[dataset],
            epochs=[1.0],
            max_length=self.max_packing_seqlen,
            oversize_policy=self.oversize_policy,
            transform=_pack,
            dataset_sampling=self.dataset_sampling,
        )

    # ─────────────────────────────────────────────────────────────────────
    # DatasetProvider entrypoint
    # ─────────────────────────────────────────────────────────────────────

    def build_datasets(self, context: DatasetBuildContext) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """Build train (packed) / valid / test datasets.

        Flickr8k has no canonical val/test split here, so we return
        ``None`` for those two and let mbridge skip eval. (Override
        ``split=...`` if you want to repurpose the train split for
        validation instead.)
        """
        train_packed = self._build_train_packed_dataloader()

        # The ``collate_fn`` for the underlying DataLoader: since
        # ``MixedPackedDataloader[idx]`` already returns a fully-formed
        # packed dict and mbridge will call us with micro_batch_size=1
        # for the SFT path, the collate is the singleton-unwrap.
        def collate_fn(batch: list) -> dict:
            assert len(batch) == 1, (
                f"Step37Flickr8kSFTDataProvider expects micro_batch_size=1 (got {len(batch)}). "
                "Each pack already aggregates multiple sub-seqs via cu_seqlens; bump "
                "max_packing_seqlen if you need bigger packs."
            )
            return batch[0]

        train_packed.collate_fn = collate_fn  # type: ignore[attr-defined]

        if self.fixed_pack_idx is not None:
            train_packed = _FixedPackDataset(train_packed, fixed_idx=self.fixed_pack_idx)

        return train_packed, None, None
