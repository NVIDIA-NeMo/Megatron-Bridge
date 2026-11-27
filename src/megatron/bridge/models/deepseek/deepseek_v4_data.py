# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Fixed-shape synthetic THD data for the DeepSeek-V4 Pro benchmarks.

Each pack holds one full-length sequence, as in the fixed-length mock runs those
benchmarks follow. This is neither MCore's variable-length ``dp_balanced`` data
scheduler nor its random token stream.
"""

from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset

from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider
from megatron.bridge.data.packing.config import PackedSequenceSpecs


class _FixedTHDMockDataset(Dataset):
    """Deterministic synthetic next-token samples with static packed metadata."""

    def __init__(
        self, *, num_samples: int, seq_length: int, vocab_size: int, seed: int, eod: int, metadata_capacity: int
    ) -> None:
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.seed = seed
        self.eod = eod
        self.metadata_capacity = metadata_capacity

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.seed + index)
        # MCore MockVarlenDataset samples length-1 token IDs and appends EOD.
        # Next-token shifting leaves length-1 valid targets; pad the final slot.
        token_stream = torch.randint(self.vocab_size, (self.seq_length + 1,), generator=generator)
        token_stream[-2:] = self.eod
        tokens = token_stream[:-1].clone()
        labels = token_stream[1:].clone()
        tokens[-1] = labels[-1] = 0
        loss_mask = torch.ones(self.seq_length, dtype=torch.float32)
        loss_mask[-1] = 0.0
        position_ids = torch.arange(self.seq_length, dtype=torch.int64)
        position_ids[-1] = 0
        padding_mask = torch.zeros(self.seq_length, dtype=torch.bool)
        padding_mask[-1] = True
        cu_seqlens = torch.full((self.metadata_capacity + 1,), self.seq_length - 1, dtype=torch.int32)
        cu_seqlens[0] = 0
        cu_seqlens_padded = torch.full((self.metadata_capacity + 1,), self.seq_length, dtype=torch.int32)
        cu_seqlens_padded[0] = 0
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "padding_mask": padding_mask,
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_kv": cu_seqlens.clone(),
            "cu_seqlens_q_padded": cu_seqlens_padded,
            "cu_seqlens_kv_padded": cu_seqlens_padded.clone(),
            "max_seqlen_q": torch.tensor(self.seq_length, dtype=torch.int32),
            "max_seqlen_kv": torch.tensor(self.seq_length, dtype=torch.int32),
            "pad_between_seqs": torch.tensor(True),
        }


@dataclass(kw_only=True)
class FixedTHDMockDatasetConfig(DatasetProvider):
    """One fixed synthetic sequence per MBS1 pack for the DSv4 benchmarks.

    The offline-packing metadata declaration makes Bridge retain boundaries on
    intermediate PP stages. Samples are generated in memory; no pack file is used.
    """

    seq_length: int
    vocab_size: int
    seed: int
    metadata_capacity: int
    dataloader_type: str = "single"
    num_workers: int = 0
    persistent_workers: bool = False
    skip_getting_attention_mask_from_dataset: bool = True
    enable_offline_packing: bool = field(init=False, default=True)
    pad_to_max_length: bool = field(init=False, default=True)
    offline_packing_specs: PackedSequenceSpecs = field(init=False)

    def __post_init__(self) -> None:
        self.finalize()

    def finalize(self) -> None:
        """Validate the fixed pack dimensions and expose static graph metadata."""
        super().finalize()
        if self.seq_length < 2 or self.vocab_size < 2 or self.metadata_capacity < 1:
            raise ValueError("DSv4 fixed THD data requires seq_length/vocab_size >= 2 and metadata_capacity >= 1.")
        self.offline_packing_specs = PackedSequenceSpecs(
            packed_sequence_size=self.seq_length,
            max_single_sequence_length=self.seq_length,
            pad_cu_seqlens=True,
            pad_seq_to_mult=1,
        )

    def build_datasets(self, context: DatasetBuildContext) -> tuple[Dataset | None, Dataset | None, Dataset | None]:
        """Build reproducible train, validation, and test sample streams."""
        if context.tokenizer is None:
            raise ValueError("DSv4 fixed THD mock data requires the configured tokenizer.")
        eod = context.tokenizer.eod
        if not 0 <= eod < self.vocab_size:
            raise ValueError("The tokenizer EOD ID must be within the configured model vocabulary.")

        def build(num_samples: int, split: int) -> Dataset | None:
            if not num_samples:
                return None
            return _FixedTHDMockDataset(
                num_samples=num_samples,
                seq_length=self.seq_length,
                vocab_size=self.vocab_size,
                seed=self.seed + split * 1_000_000_000,
                eod=eod,
                metadata_capacity=self.metadata_capacity,
            )

        return build(context.train_samples, 0), build(context.valid_samples, 1), build(context.test_samples, 2)
