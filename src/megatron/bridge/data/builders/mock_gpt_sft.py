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

"""Synthetic variable-length GPT SFT data for sequence-packing benchmarks."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy
import torch
from megatron.core.process_groups_config import ProcessGroupCollection
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from megatron.bridge.data.builders.gpt_sft import GPTSFTDatasetConfig
from megatron.bridge.data.packing.in_batch import build_mcore_thd_sequence_batch_from_rows
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer


@dataclass(kw_only=True)
class MockGPTSFTDatasetConfig(GPTSFTDatasetConfig):
    """Configure deterministic synthetic variable-length text SFT data.

    The lognormal generator follows the public Megatron-LM Dynamic CP
    positive-control: ``mu = log(mean) - sigma**2 / 2``, followed by clipping.
    The inherited GPT SFT packing flags intentionally keep this dataset on the
    same data contract as real text SFT data.
    """

    distribution: Literal["lognormal"] = "lognormal"
    min_sequence_length: int = 256
    max_sequence_length: int | None = None
    mean_sequence_length: float | None = None
    lognormal_sigma: float = 1.1
    num_base_samples: int = 1_000_000
    vocab_size: int = 32_000
    seed: int = 0
    enable_in_batch_packing: bool = True
    dataloader_type: Literal["single", "cyclic", "batch", "external"] | None = "cyclic"
    do_validation: bool = False
    do_test: bool = False
    num_workers: int = 2
    persistent_workers: bool = True

    @property
    def resolved_max_sequence_length(self) -> int:
        """Return the synthetic sample-length ceiling."""
        return self.seq_length if self.max_sequence_length is None else self.max_sequence_length

    @property
    def resolved_mean_sequence_length(self) -> float:
        """Return the pre-clipping lognormal mean."""
        if self.mean_sequence_length is not None:
            return self.mean_sequence_length
        return float(min(16_384, self.resolved_max_sequence_length))

    def validate(self) -> None:
        """Validate the synthetic distribution and GPT SFT packing contract."""
        if self.seq_length <= 0:
            raise ValueError("seq_length must be greater than 0.")
        if self.distribution != "lognormal":
            raise ValueError("Mock GPT SFT currently supports only distribution='lognormal'.")
        if self.min_sequence_length <= 0:
            raise ValueError("min_sequence_length must be greater than 0.")
        if self.resolved_max_sequence_length > self.seq_length:
            raise ValueError("max_sequence_length must not exceed seq_length.")
        if self.min_sequence_length > self.resolved_max_sequence_length:
            raise ValueError("min_sequence_length must not exceed max_sequence_length.")
        if not self.min_sequence_length <= self.resolved_mean_sequence_length <= self.resolved_max_sequence_length:
            raise ValueError("mean_sequence_length must be between min_sequence_length and max_sequence_length.")
        if self.lognormal_sigma <= 0:
            raise ValueError("lognormal_sigma must be greater than 0.")
        if self.num_base_samples <= 0:
            raise ValueError("num_base_samples must be greater than 0.")
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1.")
        if not self.enable_in_batch_packing:
            raise ValueError("Mock GPT SFT requires enable_in_batch_packing=True.")
        if self.enable_offline_packing or self.offline_packing_specs is not None:
            raise ValueError("Mock GPT SFT does not support offline packing.")
        if self.in_batch_packing_pad_to_multiple_of <= 0:
            raise ValueError("in_batch_packing_pad_to_multiple_of must be greater than 0.")
        if self.defer_in_batch_packing_to_step and not self.enable_in_batch_packing:
            raise ValueError("defer_in_batch_packing_to_step=True requires enable_in_batch_packing=True.")
        if self.dataloader_type not in {"single", "cyclic"}:
            raise ValueError("Mock GPT SFT requires dataloader_type='single' or 'cyclic'.")
        if (
            any(
                value is not None
                for value in (
                    self.dataset_root,
                    self.hf_dataset,
                    self.hf_validation_dataset,
                    self.hf_test_dataset,
                    self.hf_output_root,
                    self.hf_validation_proportion,
                    self.preprocessing,
                    self.dataset_kwargs,
                )
            )
            or self.hf_rewrite
        ):
            raise ValueError("Mock GPT SFT does not accept local, Hugging Face, or preprocessing settings.")


def generate_lognormal_sequence_lengths(config: MockGPTSFTDatasetConfig) -> numpy.ndarray:
    """Generate the deterministic clipped lognormal base length population."""
    config.validate()
    mean = config.resolved_mean_sequence_length
    mu = numpy.log(mean) - config.lognormal_sigma**2 / 2
    random_state = numpy.random.RandomState(config.seed)
    lengths = random_state.lognormal(mu, config.lognormal_sigma, config.num_base_samples)
    return numpy.clip(
        lengths,
        config.min_sequence_length,
        config.resolved_max_sequence_length,
    ).astype(numpy.int64)


class MockGPTSFTDataset(Dataset):
    """Materialize exact-length token tensors from a reusable length population."""

    def __init__(
        self,
        *,
        config: MockGPTSFTDatasetConfig,
        sequence_lengths: numpy.ndarray,
        target_length: int,
    ) -> None:
        if target_length <= 0:
            raise ValueError("target_length must be greater than 0.")
        if sequence_lengths.ndim != 1 or sequence_lengths.size == 0:
            raise ValueError("sequence_lengths must be a non-empty 1D array.")
        self.config = config
        self.sequence_lengths = sequence_lengths
        self.target_length = target_length

    def __len__(self) -> int:
        """Return the sample count requested by the training schedule."""
        return self.target_length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Build one deterministic shifted-token SFT sample."""
        is_padding_sample = index < 0
        base_index = 0 if is_padding_sample else index % self.sequence_lengths.size
        length = int(self.sequence_lengths[base_index])
        token_ids = torch.arange(length + 1, dtype=torch.long).remainder_(self.config.vocab_size)
        return {
            "tokens": token_ids[:-1],
            "labels": token_ids[1:],
            "loss_mask": torch.zeros(length) if is_padding_sample else torch.ones(length),
            "position_ids": torch.arange(length, dtype=torch.long),
        }

    def _collate_padded_rows(self, rows: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
        """Keep logical samples separate for step-time Dynamic CP scheduling."""
        if not rows:
            raise ValueError("Mock GPT SFT collation requires at least one sample.")
        return {
            "tokens": pad_sequence([row["tokens"] for row in rows], batch_first=True, padding_value=0),
            "labels": pad_sequence([row["labels"] for row in rows], batch_first=True, padding_value=0),
            "loss_mask": pad_sequence([row["loss_mask"] for row in rows], batch_first=True, padding_value=0),
            "position_ids": pad_sequence([row["position_ids"] for row in rows], batch_first=True, padding_value=0),
            "sequence_lengths": torch.tensor([row["tokens"].numel() for row in rows], dtype=torch.long),
            "attention_mask": None,
        }

    def collate_fn(self, rows: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
        """Build a static THD pack or preserve rows for Dynamic CP materialization."""
        if self.config.defer_in_batch_packing_to_step:
            return self._collate_padded_rows(rows)
        return build_mcore_thd_sequence_batch_from_rows(
            rows,
            token_key="tokens",
            sequence_length=self.config.seq_length,
            pad_token_id=0,
            pad_to_multiple_of=self.config.in_batch_packing_pad_to_multiple_of,
            emit_padding_mask=self.config.in_batch_packing_pad_to_multiple_of > 1,
        )


def _build_mock_gpt_sft_split(
    config: MockGPTSFTDatasetConfig,
    sequence_lengths: numpy.ndarray,
    target_length: int,
) -> MockGPTSFTDataset | None:
    """Build one requested synthetic split."""
    if target_length <= 0:
        return None
    return MockGPTSFTDataset(
        config=config,
        sequence_lengths=sequence_lengths,
        target_length=target_length,
    )


def mock_gpt_sft_train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int],
    dataset_config: MockGPTSFTDatasetConfig,
    tokenizer: MegatronTokenizer | None = None,
    pg_collection: ProcessGroupCollection | None = None,
) -> tuple[MockGPTSFTDataset | None, MockGPTSFTDataset | None, MockGPTSFTDataset | None]:
    """Build deterministic synthetic train, validation, and test splits."""
    del tokenizer, pg_collection
    dataset_config.validate()
    sequence_lengths = generate_lognormal_sequence_lengths(dataset_config)
    return tuple(
        _build_mock_gpt_sft_split(dataset_config, sequence_lengths, target_length)
        for target_length in train_val_test_num_samples
    )
