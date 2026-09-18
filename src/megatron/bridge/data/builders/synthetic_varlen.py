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

"""Synthetic variable-length samples for online sequence packing and dynamic CP.

The Megatron-Core packing scheduler consumes *unpacked* per-sample dicts and
packs them per step. This provider generates deterministic random-token
samples whose lengths follow a configurable distribution, so packing and
dynamic context parallelism can be exercised end to end without any dataset on
disk. Lengths are padded to the multiple that context-parallel THD slicing
requires; padding tokens are never supervised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch

from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider
from megatron.bridge.data.collators.identity import identity_collate


class SyntheticVarlenDataset(torch.utils.data.Dataset):
    """Deterministic random-token samples with variable lengths.

    Every item is a dict with the keys the Megatron-Core scheduler requires:
    ``tokens``, ``labels``, ``position_ids`` (int64, ``[padded_len]``),
    ``loss_mask`` (float32, ``[padded_len]``), ``original_seq_len`` and
    ``padded_seq_len`` (int32, ``[1]``). Item ``i`` is a pure function of
    ``(seed, i)``, so every context-parallel replica of a data-parallel rank
    sees byte-identical samples.
    """

    def __init__(
        self,
        *,
        num_samples: int,
        seq_length: int,
        min_seq_length: int,
        median_seq_length: int,
        lognormal_sigma: float,
        length_distribution: Literal["lognormal", "uniform"],
        pad_to_multiple: int,
        vocab_size: int,
        seed: int,
        fold_padding_into_sequence: bool,
    ) -> None:
        if seq_length % pad_to_multiple != 0:
            raise ValueError(
                f"seq_length ({seq_length}) must be a multiple of the padding multiple ({pad_to_multiple}) "
                "so a cap-length sample stays within the scheduler's bin capacity."
            )
        if not 1 <= min_seq_length <= seq_length:
            raise ValueError(f"min_seq_length must be in [1, seq_length], got {min_seq_length} / {seq_length}.")
        if vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {vocab_size}.")
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.min_seq_length = min_seq_length
        self.median_seq_length = median_seq_length
        self.lognormal_sigma = lognormal_sigma
        self.length_distribution = length_distribution
        self.pad_to_multiple = pad_to_multiple
        self.vocab_size = vocab_size
        self.seed = seed
        self.fold_padding_into_sequence = fold_padding_into_sequence
        # The DataLoader picks this up so each next() yields a list of sample dicts.
        self.collate_fn = identity_collate

    def __len__(self) -> int:
        return self.num_samples

    def _generator(self, idx: int, stream: int) -> torch.Generator:
        """Seed a generator from (seed, sample, stream) so splits and streams never alias."""
        mixed = (self.seed * 0x9E3779B97F4A7C15) ^ (idx * 0xC2B2AE3D27D4EB4F) ^ (stream * 0x165667B19E3779F9)
        return torch.Generator().manual_seed(mixed & (2**63 - 1))

    def sample_length(self, idx: int) -> int:
        """Return the unpadded length of sample ``idx``."""
        generator = self._generator(idx, stream=1)
        if self.length_distribution == "uniform":
            length = int(torch.randint(self.min_seq_length, self.seq_length + 1, (1,), generator=generator))
        else:
            z = float(torch.randn(1, generator=generator))
            length = int(round(math.exp(math.log(self.median_seq_length) + self.lognormal_sigma * z)))
        return max(self.min_seq_length, min(self.seq_length, length))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        length = self.sample_length(idx)
        padded = min(self.seq_length, math.ceil(length / self.pad_to_multiple) * self.pad_to_multiple)
        generator = self._generator(idx, stream=2)
        tokens = torch.randint(1, self.vocab_size, (padded,), dtype=torch.int64, generator=generator)
        tokens[length:] = 0
        labels = torch.roll(tokens, shifts=-1, dims=0)
        labels[length - 1 :] = 0
        loss_mask = torch.ones(padded, dtype=torch.float32)
        loss_mask[length - 1 :] = 0.0
        reported_length = padded if self.fold_padding_into_sequence else length
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": torch.arange(padded, dtype=torch.int64),
            "original_seq_len": torch.tensor([reported_length], dtype=torch.int32),
            "padded_seq_len": torch.tensor([padded], dtype=torch.int32),
        }


@dataclass(kw_only=True)
class SyntheticVarlenDatasetConfig(DatasetProvider):
    """Synthetic variable-length token stream for sequence packing / dynamic CP runs.

    Pair with ``model.sequence_packing_scheduler`` and ``train.micro_batch_size=1``.
    ``ConfigContainer.validate`` fills ``sequence_padding_multiple`` from the
    parallel layout when it is left ``None``.
    """

    seq_length: int
    """Maximum (and cap) sequence length in tokens; also the scheduler bin unit."""

    min_seq_length: int = 8
    """Shortest sample length before padding."""

    median_seq_length: int | None = None
    """Median of the lognormal length distribution. Defaults to ``seq_length // 8``."""

    lognormal_sigma: float = 1.0
    """Spread of the lognormal length distribution (standard deviation of log-length)."""

    length_distribution: Literal["lognormal", "uniform"] = "lognormal"
    """Shape of the per-sample length distribution."""

    sequence_padding_multiple: int | None = None
    """Length multiple for CP THD slicing: ``2 * dp * cp`` under dynamic CP, ``2 * cp`` for
    static CP, times TP with sequence parallelism. ``None`` derives it at validation/build time."""

    fold_padding_into_sequence: bool = False
    """Report the padded length as the sequence length (padding stays loss-masked). Use when the
    attention backend rejects THD bins with padding between sequences."""

    vocab_size: int | None = None
    """Token id range; ``None`` uses the tokenizer vocabulary."""

    random_seed: int = 1234
    """Seed for lengths and tokens (validation and test splits derive their own seeds)."""

    dataloader_type: Literal["single", "cyclic", "batch", "external"] | None = "single"

    def finalize(self) -> None:
        """Validate declarative fields."""
        super().finalize()
        if self.seq_length <= 0:
            raise ValueError(f"seq_length must be positive, got {self.seq_length}.")
        if self.median_seq_length is None:
            self.median_seq_length = max(self.min_seq_length, self.seq_length // 8)
        if not 1 <= self.min_seq_length <= self.seq_length:
            raise ValueError(f"min_seq_length must be in [1, seq_length], got {self.min_seq_length}.")
        if self.lognormal_sigma <= 0:
            raise ValueError(f"lognormal_sigma must be positive, got {self.lognormal_sigma}.")
        if self.dataloader_type not in ("single", "cyclic"):
            raise ValueError(
                f"SyntheticVarlenDatasetConfig requires dataloader_type 'single' or 'cyclic', got {self.dataloader_type!r}."
            )

    def resolve_padding_multiple(self, context: DatasetBuildContext) -> int:
        """Return the configured padding multiple or the conservative ``2 * dp * cp`` default."""
        if self.sequence_padding_multiple is not None:
            return self.sequence_padding_multiple
        if context.pg_collection is None:
            return 1
        return 2 * context.pg_collection.dp.size() * context.pg_collection.cp.size()

    def build_datasets(self, context: DatasetBuildContext) -> tuple[Any | None, Any | None, Any | None]:
        """Build train/valid/test datasets sized to the requested sample counts."""
        vocab_size = self.vocab_size
        if vocab_size is None:
            if context.tokenizer is None:
                raise ValueError("SyntheticVarlenDatasetConfig needs vocab_size or a tokenizer in the build context.")
            vocab_size = int(context.tokenizer.vocab_size)
        pad_to_multiple = self.resolve_padding_multiple(context)
        median = (
            self.median_seq_length
            if self.median_seq_length is not None
            else max(self.min_seq_length, self.seq_length // 8)
        )

        def _split(num_samples: int, seed_offset: int) -> SyntheticVarlenDataset | None:
            if num_samples <= 0:
                return None
            return SyntheticVarlenDataset(
                num_samples=num_samples,
                seq_length=self.seq_length,
                min_seq_length=self.min_seq_length,
                median_seq_length=median,
                lognormal_sigma=self.lognormal_sigma,
                length_distribution=self.length_distribution,
                pad_to_multiple=pad_to_multiple,
                vocab_size=vocab_size,
                seed=self.random_seed + seed_offset,
                fold_padding_into_sequence=self.fold_padding_into_sequence,
            )

        return (
            _split(context.train_samples, 0),
            _split(context.valid_samples, 0x5DEECE66D),
            _split(context.test_samples, 0xBB67AE85),
        )
