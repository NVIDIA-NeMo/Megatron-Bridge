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

"""Serializable config and runtime builder for synthetic variable-length text SFT samples.

The samples exercise global-batch packing and dynamic context parallelism without a dataset or tokenizer. Each sample
is random token IDs whose length is drawn from three buckets defined by the context-parallel capacity it needs: at most
``seq_length // 4`` tokens (fits one rank of a CP4 group), at most ``seq_length // 2`` (CP2), or up to ``seq_length``
(CP4). Lengths are uniform within each bucket, and only the final ``completion_fraction`` of each sample is supervised.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from megatron.bridge.data.base import DataloaderConfig, DatasetBuildContext
from megatron.bridge.data.packing.global_batch import build_unpacked_sequence_sample, fold_alignment_padding


_SPLITS = ("train", "validation", "test")


@dataclass(kw_only=True)
class MockGPTSFTDatasetConfig(DataloaderConfig):
    """Synthetic variable-length text SFT samples for global-batch packing.

    Args:
        seq_length: Maximum training length of one sample, in tokens after the next-token shift.
        vocab_size: Token IDs are drawn from ``[1, vocab_size)``.
        length_bucket_fractions: Fractions of samples whose length needs at most CP1, CP2, and CP4 capacity of a
            ``seq_length`` budget split over four ranks (``seq_length // 4`` and ``seq_length // 2`` boundaries).
        completion_fraction: Share of each sample's final tokens that is supervised.
        random_seed: Seed for lengths and token IDs; each split derives its own stream.
        enable_global_batch_packing: Must be ``True``; samples are yielded unpacked for the global-batch scheduler.
        global_batch_packing_pad_to_multiple_of: Per-sequence CP alignment, set by ``ConfigContainer.validate``.
        fold_alignment_padding: Report alignment padding as part of each sequence, as for
            :class:`~megatron.bridge.data.builders.GPTSFTDatasetConfig`.
    """

    seq_length: int
    vocab_size: int
    length_bucket_fractions: tuple[float, float, float] = (0.5, 0.3, 0.2)
    completion_fraction: float = 0.2
    random_seed: int = 1234
    enable_global_batch_packing: bool = True
    global_batch_packing_pad_to_multiple_of: int = 1
    fold_alignment_padding: bool = False
    dataloader_type: Literal["single", "cyclic"] | None = "single"

    @property
    def yields_unpacked_samples(self) -> bool:
        """Whether the built datasets yield the per-sample dicts global-batch packing consumes."""
        return self.enable_global_batch_packing

    def validate(self) -> None:
        """Validate synthetic data settings."""
        if self.seq_length < 4:
            raise ValueError("seq_length must be at least 4.")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2.")
        fractions = tuple(self.length_bucket_fractions)
        if (
            len(fractions) != 3
            or any(not math.isfinite(value) or value < 0 for value in fractions)
            or not math.isclose(sum(fractions), 1.0, abs_tol=1e-8)
        ):
            raise ValueError("length_bucket_fractions must be three non-negative values summing to 1.")
        if not 0 < self.completion_fraction <= 1:
            raise ValueError("completion_fraction must be in (0, 1].")
        if not self.enable_global_batch_packing:
            raise ValueError("MockGPTSFTDatasetConfig only yields unpacked samples; keep enable_global_batch_packing.")
        if self.global_batch_packing_pad_to_multiple_of < 1:
            raise ValueError("global_batch_packing_pad_to_multiple_of must be at least 1.")
        if self.dataloader_type not in ("single", "cyclic"):
            raise ValueError("Global-batch packing needs dataloader_type 'single' or 'cyclic'.")

    def finalize(self) -> None:
        """Finalize dataloader settings and validate this config."""
        super().finalize()
        self.validate()


def length_bucket_bounds(seq_length: int) -> list[tuple[int, int]]:
    """Return the inclusive length range of each bucket: CP1, CP2, and CP4 capacity of ``seq_length``."""
    quarter, half = seq_length // 4, seq_length // 2
    return [(1, quarter), (quarter + 1, half), (half + 1, seq_length)]


def length_bucket_counts(num_samples: int, fractions: Sequence[float]) -> list[int]:
    """Split ``num_samples`` over the buckets with largest remainders, so the total is exact."""
    raw = [num_samples * value for value in fractions]
    counts = [math.floor(value) for value in raw]
    for bucket in sorted(range(len(raw)), key=lambda i: (-(raw[i] - counts[i]), i))[: num_samples - sum(counts)]:
        counts[bucket] += 1
    return counts


def mock_sft_sequence_lengths(config: MockGPTSFTDatasetConfig, num_samples: int, split: str = "train") -> np.ndarray:
    """Return the deterministic training length of every sample of ``split``, in dataset order.

    Raises:
        ValueError: If ``split`` is unknown or ``num_samples`` is negative.
    """
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {_SPLITS}.")
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative.")
    rng = np.random.default_rng((config.random_seed, _SPLITS.index(split)))
    lengths = [
        rng.integers(low, high + 1, size=count)
        for (low, high), count in zip(
            length_bucket_bounds(config.seq_length), length_bucket_counts(num_samples, config.length_bucket_fractions)
        )
    ]
    lengths = np.concatenate(lengths).astype(np.int64)
    rng.shuffle(lengths)
    return lengths


class MockGPTSFTDataset(torch.utils.data.Dataset):
    """Random-token SFT samples of fixed per-index length, collated as unpacked global-batch packing samples."""

    def __init__(self, config: MockGPTSFTDatasetConfig, num_samples: int, split: str) -> None:
        self.config = config
        self.split = split
        self.lengths = mock_sft_sequence_lengths(config, num_samples, split)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.lengths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return ``length + 1`` token IDs for sample ``index``; the same index always yields the same tokens."""
        length = int(self.lengths[index])
        rng = np.random.default_rng((self.config.random_seed, _SPLITS.index(self.split), int(index)))
        return {"input_ids": torch.from_numpy(rng.integers(1, self.config.vocab_size, size=length + 1))}

    def collate_fn(self, batch: list[dict[str, Any]]) -> list[dict[str, torch.Tensor]]:
        """Shift once, supervise the completion, and pad each sequence to the CP alignment multiple."""
        samples = []
        for item in batch:
            input_ids = torch.as_tensor(item["input_ids"], dtype=torch.long)
            tokens, labels = input_ids[:-1], input_ids[1:]
            loss_mask = torch.zeros(tokens.numel(), dtype=torch.float32)
            supervised = max(1, round(tokens.numel() * self.config.completion_fraction))
            loss_mask[-supervised:] = 1.0
            sample = build_unpacked_sequence_sample(
                tokens,
                labels,
                loss_mask,
                pad_to_multiple_of=self.config.global_batch_packing_pad_to_multiple_of,
                pad_token_id=0,
            )
            if self.config.fold_alignment_padding:
                fold_alignment_padding(sample)
            samples.append(sample)
        return samples


class MockGPTSFTDatasetBuilder:
    """Build synthetic variable-length SFT splits from declarative settings."""

    def __init__(self, config: MockGPTSFTDatasetConfig) -> None:
        config.validate()
        self.config = config

    def build(
        self, context: DatasetBuildContext
    ) -> tuple[MockGPTSFTDataset | None, MockGPTSFTDataset | None, MockGPTSFTDataset | None]:
        """Build the train, validation, and test splits requested by the schedule."""
        sizes = (context.train_samples, context.valid_samples, context.test_samples)
        return tuple(
            MockGPTSFTDataset(self.config, size, split) if size and size > 0 else None
            for size, split in zip(sizes, _SPLITS)
        )


def mock_gpt_sft_train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int],
    dataset_config: MockGPTSFTDatasetConfig,
    tokenizer: Any | None = None,
    pg_collection: Any | None = None,
) -> tuple[MockGPTSFTDataset | None, MockGPTSFTDataset | None, MockGPTSFTDataset | None]:
    """Build synthetic SFT splits through the canonical runtime builder; no tokenizer is used."""
    context = DatasetBuildContext(
        train_samples=train_val_test_num_samples[0],
        valid_samples=train_val_test_num_samples[1],
        test_samples=train_val_test_num_samples[2],
        tokenizer=tokenizer,
        pg_collection=pg_collection,
    )
    return MockGPTSFTDatasetBuilder(dataset_config).build(context)
