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

"""SFT data for the ERNIE 5.5 training step: text or pre-tokenized JSONL, or synthetic documents, packed into THD rows.

Every dataset item is one THD row: a list of documents, each a pair of equal-length token and label lists with
``labels[t]`` the target of position ``t`` (``-100`` for none). The collate function concatenates the rows of a
micro-batch into the batch contract of :mod:`~megatron.bridge.models.ernie.ernie55_step` (``tokens``, ``labels``,
``cu_seqlens`` and ``pad_token_id``); the step pads every document to a multiple of ``2 * CP`` and slices the
context-parallel rank's rows.

Sources:

* ``path``: a JSONL file with one sample per line, either pre-tokenized, ``{"input_ids": [...], "labels": [...]}``
  (``labels`` follow the Hugging Face convention, ``labels[t] == input_ids[t]`` where supervised, the shift happens
  here, unless ``labels_are_shifted`` says they are already aligned), or text in a format of Megatron-Bridge's
  text-only SFT tutorial: chat rows ``{"messages": [...]}`` or prompt/completion rows ``{"input": ..., "output": ...}``.
  Text rows need ``hf_tokenizer``; they are encoded once, when the dataset is built, with the reference's chat
  template (:mod:`~megatron.bridge.models.ernie.ernie55_chat`), and kept in memory. For large corpora pre-tokenize with
  ``examples/models/ernie55/convert_sft_jsonl.py`` instead.
* ``path=None``: synthetic samples with deterministic random tokens. Every row holds documents of
  ``mock_document_lengths`` tokens whose first ``mock_prompt_fraction`` carries no loss.

A sample longer than ``seq_length`` is split into ``seq_length`` chunks (``truncation="split"``, every chunk its own
sample, like the reference's ``--truncation_strategy split``) or cut (``truncation="right"``). ``pack=True`` bins the
samples into rows of at most ``seq_length`` padded tokens (first fit, longest first; the reference's ``--packing``),
``pack=False`` keeps one sample per row (the reference's production setting). Items past the end of the rows cycle
through them again.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch.utils.data import Dataset

from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider
from megatron.bridge.models.ernie.ernie55_chat import encode_ernie55_sft_row, load_ernie55_chat_template


__all__ = ["Ernie55SFTDataset", "Ernie55SFTDatasetConfig", "collate_ernie55_rows"]

_IGNORE = -100

Document = tuple[list[int], list[int]]


def _shift(input_ids: list[int], labels: list[int]) -> Document:
    """Hugging Face labels to aligned targets: the target of position ``t`` is ``labels[t + 1]``."""
    return input_ids, [*labels[1:], _IGNORE]


def _chunks(input_ids: list[int], labels: list[int], *, seq_length: int, truncation: str) -> list[Document]:
    if truncation == "right" or len(input_ids) <= seq_length:
        return [(input_ids[:seq_length], labels[:seq_length])]
    return [
        (input_ids[start : start + seq_length], labels[start : start + seq_length])
        for start in range(0, len(input_ids), seq_length)
    ]


def collate_ernie55_rows(rows: Sequence[list[Document]], *, pad_token_id: int) -> dict[str, Any]:
    """Concatenate the documents of a micro-batch's rows into one packed row.

    Args:
        rows: The micro-batch's rows, each a list of ``(input_ids, labels)`` documents.
        pad_token_id: Token the training step pads documents with.

    Returns:
        ``tokens`` and ``labels`` ``[1, T]`` int64, ``cu_seqlens`` ``[n + 1]`` int64 and ``pad_token_id``.
    """
    documents = [document for row in rows for document in row]
    lengths = [len(input_ids) for input_ids, _ in documents]
    return {
        "tokens": torch.tensor([token for input_ids, _ in documents for token in input_ids]).unsqueeze(0),
        "labels": torch.tensor([label for _, labels in documents for label in labels]).unsqueeze(0),
        "cu_seqlens": torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()]),
        "pad_token_id": pad_token_id,
    }


class Ernie55SFTDataset(Dataset):
    """Packed ERNIE 5.5 SFT rows (see the module docstring)."""

    def __init__(self, config: Ernie55SFTDatasetConfig, *, num_items: int, seed_offset: int) -> None:
        self.config = config
        self.seed_offset = seed_offset
        if config.path is None:
            # Synthetic rows are generated from their index.
            self.rows: list[list[tuple[int, int]]] = []
            self.samples = None
            self.num_rows = max(num_items, 1)
        else:
            self.samples: list[dict[str, list[int]]] | None = self._encode_text_file() if config.hf_tokenizer else None
            self.rows = self._plan_rows(self._read_lengths())
            if not self.rows:
                raise ValueError(f"{config.path} holds no sample.")
            self.num_rows = len(self.rows)
        self.num_items = max(num_items, self.num_rows)

    def __len__(self) -> int:
        return self.num_items

    def _encode_text_file(self) -> list[dict[str, list[int]]]:
        """Every sample of the JSONL file as ``{"input_ids", "labels"}``, encoding text rows with the tokenizer."""
        from transformers import AutoTokenizer

        config = self.config
        tokenizer = AutoTokenizer.from_pretrained(config.hf_tokenizer)
        template = load_ernie55_chat_template(config.chat_template)
        samples = []
        with open(config.path, encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if "input_ids" in row:
                    samples.append({"input_ids": list(row["input_ids"]), "labels": list(row["labels"])})
                    continue
                try:
                    samples.append(
                        encode_ernie55_sft_row(
                            tokenizer,
                            row,
                            chat_template=template,
                            chat_template_kwargs=dict(config.chat_template_kwargs),
                            prompt_column=config.prompt_column,
                            completion_column=config.completion_column,
                            loss=config.loss,
                        )
                    )
                except Exception as error:
                    raise ValueError(f"{config.path}:{lineno}: {error}") from error
        return samples

    def _chunk_lengths(self, key: int, length: int) -> list[tuple[int, int, int]]:
        """``(key, chunk index, chunk length)`` of the chunks of one ``length``-token sample."""
        config = self.config
        count = 1 if config.truncation == "right" else max(1, math.ceil(length / config.seq_length))
        return [(key, chunk, min(config.seq_length, length - chunk * config.seq_length)) for chunk in range(count)]

    def _read_lengths(self) -> list[tuple[int, int, int]]:
        """``(key, chunk index, chunk length)`` of every sample chunk; the key is the byte offset of the sample in
        the JSONL file, or its index in the encoded samples."""
        config = self.config
        if self.samples is not None:
            return [
                piece
                for index, sample in enumerate(self.samples)
                for piece in self._chunk_lengths(index, len(sample["input_ids"]))
            ]
        chunks = []
        with open(config.path, "rb") as stream:
            offset = 0
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if "input_ids" not in row:
                        raise ValueError(
                            f"{config.path}: a text row at byte {offset} needs dataset.hf_tokenizer "
                            "(or pre-tokenize the file with examples/models/ernie55/convert_sft_jsonl.py)."
                        )
                    chunks.extend(self._chunk_lengths(offset, len(row["input_ids"])))
                offset += len(line)
        return chunks

    def _plan_rows(self, chunks: list[tuple[int, int, int]]) -> list[list[tuple[int, int]]]:
        """Group the sample chunks into rows: one per row, or first fit longest first into ``seq_length`` rows."""
        config = self.config
        if not config.pack:
            return [[(offset, chunk)] for offset, chunk, _ in chunks]
        multiple = config.pad_to_multiple_of
        padded = [(math.ceil(length / multiple) * multiple, offset, chunk) for offset, chunk, length in chunks]
        rows: list[list[tuple[int, int]]] = []
        free: list[int] = []
        for length, offset, chunk in sorted(padded, key=lambda item: -item[0]):
            for index, room in enumerate(free):
                if length <= room:
                    rows[index].append((offset, chunk))
                    free[index] -= length
                    break
            else:
                rows.append([(offset, chunk)])
                free.append(config.seq_length - length)
        # Shuffle the rows (first fit orders them by their longest sample).
        random.Random(config.seed).shuffle(rows)
        return rows

    def _read_document(self, key: int, chunk: int) -> Document:
        config = self.config
        if self.samples is not None:
            sample = self.samples[key]
        else:
            with open(config.path, "rb") as stream:
                stream.seek(key)
                sample = json.loads(stream.readline())
        input_ids, labels = list(sample["input_ids"]), list(sample["labels"])
        if len(labels) != len(input_ids):
            raise ValueError(f"Sample {key} of {config.path}: input_ids and labels differ in length.")
        if config.vocab_size is not None:
            largest = max(input_ids + [label for label in labels if label != _IGNORE], default=0)
            if largest >= config.vocab_size:
                raise ValueError(
                    f"Sample {key} of {config.path} holds token id {largest}, but the model's vocab_size is "
                    f"{config.vocab_size}: the tokenizer does not match the model."
                )
        if not config.labels_are_shifted:
            # Split before shifting, so the last token of every chunk has no target, as in the reference.
            pieces = _chunks(input_ids, labels, seq_length=config.seq_length, truncation=config.truncation)
            return _shift(*pieces[chunk])
        return _chunks(input_ids, labels, seq_length=config.seq_length, truncation=config.truncation)[chunk]

    def _synthetic_row(self, index: int) -> list[Document]:
        config = self.config
        generator = torch.Generator().manual_seed(config.seed + self.seed_offset + index)
        low, high = config.mock_token_range
        documents = []
        for length in config.mock_document_lengths:
            tokens = torch.randint(low, high, (length,), generator=generator).tolist()
            labels = list(tokens)
            prompt = int(length * config.mock_prompt_fraction)
            labels[:prompt] = [_IGNORE] * prompt
            documents.append(_shift(tokens, labels))
        return documents

    def __getitem__(self, index: int) -> list[Document]:
        if self.config.path is None:
            return self._synthetic_row(index)
        return [self._read_document(offset, chunk) for offset, chunk in self.rows[index % self.num_rows]]

    def collate_fn(self, rows: Sequence[list[Document]]) -> dict[str, Any]:
        """Collate a micro-batch for the ERNIE 5.5 step (:func:`collate_ernie55_rows`)."""
        return collate_ernie55_rows(rows, pad_token_id=self.config.pad_token_id)


@dataclass(kw_only=True)
class Ernie55SFTDatasetConfig(DatasetProvider):
    """Packed SFT rows for :mod:`~megatron.bridge.models.ernie.ernie55_step` (see the module docstring)."""

    seq_length: int
    """Longest row (and longest sample chunk), in tokens."""

    pad_token_id: int
    """Token the training step pads documents with (the tokenizer's pad token)."""

    vocab_size: int | None = None
    """The model's vocabulary size; set, every token id and label of a sample must be below it (an id past the
    embedding fails on the GPU only, as a device-side assert). The recipe sets it from the model config."""

    path: str | None = None
    """JSONL file of pre-tokenized or text samples (see the module docstring); ``None`` for synthetic documents."""

    hf_tokenizer: str | None = None
    """Hugging Face tokenizer (directory or Hub id) that encodes text rows; ERNIE 5.5's own for real runs."""

    chat_template: str | None = None
    """Chat template file for text rows; ``None`` uses the reference's ERNIE 5.5 SFT template."""

    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    """Extra template variables for text rows, such as ``{"enable_thinking": False}``."""

    prompt_column: str = "input"
    """Prompt column of prompt/completion text rows."""

    completion_column: str = "output"
    """Completion column of prompt/completion text rows."""

    loss: Literal["assistant", "all"] = "assistant"
    """Supervised tokens of text rows: the template's assistant spans (the reference's default), or all."""

    labels_are_shifted: bool = False
    """Whether the JSONL labels are already aligned (``labels[t]`` is the target of position ``t``)."""

    truncation: Literal["split", "right"] = "split"
    """What happens to samples longer than ``seq_length``: split into chunks, or cut."""

    pack: bool = False
    """Bin several samples into one row of at most ``seq_length`` padded tokens."""

    pad_to_multiple_of: int = 1
    """Padding multiple of every document (``2 * context_parallel_size``), used to size packed rows."""

    mock_document_lengths: tuple[int, ...] = field(default_factory=tuple)
    """Document lengths of every synthetic row."""

    mock_prompt_fraction: float = 0.0
    """Leading fraction of every synthetic document without a loss."""

    mock_token_range: tuple[int, int] = (1, 32000)
    """Half-open range of synthetic token ids."""

    seed: int = 1234
    """Seed of the synthetic tokens and of the packed-row order."""

    dataloader_type: Literal["single", "cyclic", "batch", "external"] | None = "single"
    num_workers: int = 0
    persistent_workers: bool = False

    def finalize(self) -> None:
        """Validate the settings.

        Raises:
            ValueError: On an empty or overlong synthetic row, or a non-positive length or padding multiple.
        """
        super().finalize()
        if self.seq_length < 1 or self.pad_to_multiple_of < 1:
            raise ValueError("seq_length and pad_to_multiple_of must be positive.")
        if self.hf_tokenizer is not None and self.path is None:
            raise ValueError("dataset.hf_tokenizer encodes the text rows of dataset.path; set dataset.path too.")
        if self.path is None:
            lengths = self.mock_document_lengths
            if not lengths or any(length < 1 for length in lengths):
                raise ValueError("Synthetic ERNIE 5.5 data needs positive mock_document_lengths.")
            padded = sum(math.ceil(length / self.pad_to_multiple_of) * self.pad_to_multiple_of for length in lengths)
            if padded > self.seq_length:
                raise ValueError(
                    f"mock_document_lengths {lengths} pad to {padded} tokens, more than seq_length={self.seq_length}."
                )

    def build_datasets(
        self, context: DatasetBuildContext
    ) -> tuple[Ernie55SFTDataset, Ernie55SFTDataset | None, Ernie55SFTDataset | None]:
        """Build the train, validation and test datasets.

        Args:
            context: Requested sample counts.

        Returns:
            Train, validation (when requested) and test (when requested) datasets.
        """
        train = Ernie55SFTDataset(self, num_items=context.train_samples, seed_offset=0)
        valid = (
            Ernie55SFTDataset(self, num_items=context.valid_samples, seed_offset=10**7)
            if context.valid_samples
            else None
        )
        test = (
            Ernie55SFTDataset(self, num_items=context.test_samples, seed_offset=2 * 10**7)
            if context.test_samples
            else None
        )
        return train, valid, test
