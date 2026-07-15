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

"""Flickr8k (``intro/flickr8k``) sample loader — ported verbatim from
``playground/data/sft/step37/flickr8k_sft_data.py``.

Downloads ``train/metadata.csv`` and the per-row ``train/<file_name>.jpg``
images via ``huggingface_hub.hf_hub_download`` (no ``transformers``
involved). Output: a list of :class:`Flickr8kSample`, then wrapped into
:class:`Step37Flickr8kDataset` for the tokenize step.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from megatron.bridge.models.stepfun.data.flickr8k.template import IMAGE_PLACEHOLDER


@dataclass(frozen=True)
class Flickr8kSample:
    """Image-caption sample from the Flickr8k dataset."""

    image_path: str
    caption: str


class Step37Flickr8kDataset:
    """Step3.7 SFT dataset over the CC0 Flickr8k image-caption data.

    Map-style ``torch.utils.data.Dataset`` (no inheritance — duck-typed):
    ``__len__`` returns ``len(samples)``, ``__getitem__(idx)`` returns
    the tokenized :class:`MultimodalSFTSample`.
    """

    def __init__(self, samples: list[Flickr8kSample], template, prompt: str):
        if not samples:
            raise ValueError("samples cannot be empty")
        self.samples = samples
        self.template = template
        self.prompt = prompt

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        dialog = self._to_dialog(sample, self.prompt)
        result = self.template(dialog)
        return result

    @staticmethod
    def _to_dialog(sample: Flickr8kSample, prompt: str) -> dict[str, Any]:
        return {
            "images": [sample.image_path],
            "conversations": [
                {
                    "role": "user",
                    "content": f"{IMAGE_PLACEHOLDER}\n{prompt}",
                },
                {
                    "role": "assistant",
                    "content": sample.caption,
                },
            ],
        }


def get_flickr8k_dataset_file(*, repo_id: str, filename: str, cache_dir: Path) -> Path:
    """Download (or reuse cached) a single Flickr8k file via ``hf_hub_download``.

    The download call is **not multi-process safe** — Hugging Face's
    Xet client and ``_local_folder.read/write_download_metadata`` use
    per-file metadata locks that deadlock when N ranks race against the
    same ``cache_dir``. Callers must serialise concurrent invocations
    (rank-0-only + ``torch.distributed.barrier()`` is the standard
    pattern — see :func:`prepare_flickr8k_samples`).
    """
    local_path = cache_dir / filename
    if local_path.is_file() and local_path.stat().st_size > 0:
        return local_path

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(cache_dir),
        )
    )


def _is_global_rank_0() -> bool:
    """Return True on global rank 0 (or when torch.distributed isn't initialized)."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


def _maybe_barrier() -> None:
    """``torch.distributed.barrier()`` if process group is up, else no-op."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass


def prepare_flickr8k_samples(
    *,
    repo_id: str = "intro/flickr8k",
    split: str = "train",
    sample_count: int | None = 8,
    caption_key: str = "caption_0",
    cache_dir: str = ".cache/step37_flickr8k",
) -> list[Flickr8kSample]:
    """Download metadata.csv + the first ``sample_count`` images and build
    ``Flickr8kSample`` records.

    ``sample_count`` defaults to ``8``. Pass ``None`` to take the
    full Flickr8k train split (~6000 rows, ~1 GB jpgs, slow on a cold
    cache).

    **Distributed-safety**: the actual ``hf_hub_download`` calls only run
    on global rank 0; non-zero ranks wait on a ``torch.distributed.barrier``
    until rank 0 has populated the cache, then they read the same files
    from disk. This avoids the multi-process deadlock seen when N ranks
    race ``huggingface_hub``'s Xet + ``_local_folder.metadata`` locks
    against the same ``cache_dir`` (lustre / NFS shared filesystem).
    """
    cache_root = Path(cache_dir).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)

    is_rank_0 = _is_global_rank_0()

    # ── Phase A (rank 0 only): populate the cache via hf_hub_download ────
    if is_rank_0:
        metadata_path = get_flickr8k_dataset_file(
            repo_id=repo_id, filename=f"{split}/metadata.csv", cache_dir=cache_root
        )
        with open(metadata_path, encoding="utf-8", newline="") as f:
            rows_for_prefetch = list(csv.DictReader(f))
        prefetched = 0
        for row in rows_for_prefetch:
            caption = row.get(caption_key, "").strip()
            file_name = row.get("file_name", "").strip()
            if not caption or not file_name:
                continue
            get_flickr8k_dataset_file(
                repo_id=repo_id,
                filename=f"{split}/{file_name}",
                cache_dir=cache_root,
            )
            prefetched += 1
            if sample_count is not None and prefetched >= sample_count:
                break

    # ── Cross-rank barrier: wait until rank 0 has populated the cache ────
    _maybe_barrier()

    # ── Phase B (all ranks): build the sample list by reading the cache ──
    # On rank 0 this is a re-walk; on other ranks every file is already
    # local thanks to rank 0's prefetch, so ``get_flickr8k_dataset_file``
    # hits the ``local_path.is_file()`` fast path and never calls
    # ``hf_hub_download``.
    metadata_path = get_flickr8k_dataset_file(repo_id=repo_id, filename=f"{split}/metadata.csv", cache_dir=cache_root)
    with open(metadata_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    samples: list[Flickr8kSample] = []
    for row in rows:
        caption = row.get(caption_key, "").strip()
        file_name = row.get("file_name", "").strip()
        if not caption or not file_name:
            continue
        image_path = get_flickr8k_dataset_file(repo_id=repo_id, filename=f"{split}/{file_name}", cache_dir=cache_root)
        samples.append(Flickr8kSample(image_path=str(image_path), caption=caption))
        if sample_count is not None and len(samples) >= sample_count:
            break

    if not samples:
        raise RuntimeError(f"No Flickr8k samples prepared from {repo_id}/{split}")
    return samples
