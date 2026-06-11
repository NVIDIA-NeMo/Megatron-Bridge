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

"""Text SFT provider for Hugging Face datasets with offline packing support."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from datasets import Dataset

from megatron.bridge.data.builders.finetuning_dataset import FinetuningDatasetBuilder
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.data.datasets.sft import get_dataset_root
from megatron.bridge.data.hf_datasets.makers import get_hf_dataset_maker
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider
from megatron.bridge.utils.common_utils import get_rank_safe, print_rank_0


@dataclass(kw_only=True)
class HFTextSFTDatasetProvider(DatasetProvider):
    """Build text SFT datasets from Hugging Face makers via the standard SFT builder.

    Maker outputs are written as JSONL chat rows and then loaded through
    ``FinetuningDatasetBuilder``. This preserves optional offline packed-sequence
    preparation through ``enable_offline_packing`` and ``PackedSequenceSpecs`` while keeping Hugging Face row
    normalization in ``megatron.bridge.data.hf_datasets``.
    """

    seq_length: int
    maker_name: str
    maker_kwargs: dict[str, Any] | None = None
    val_maker_kwargs: dict[str, Any] | None = None
    test_maker_kwargs: dict[str, Any] | None = None
    dataset_root: str | Path | None = None
    seed: int = 5678
    memmap_workers: int = 1
    max_train_samples: int | None = None
    enable_offline_packing: bool = False
    offline_packing_specs: PackedSequenceSpecs | None = None
    dataset_kwargs: dict[str, Any] | None = None
    val_proportion: float | None = None
    do_validation: bool = True
    do_test: bool = True
    rewrite: bool = False
    dataloader_type: Literal["single", "cyclic", "batch", "external"] | None = "batch"

    def _default_dataset_root(self) -> Path:
        dataset_name = str((self.maker_kwargs or {}).get("path_or_dataset", self.maker_name))
        return get_dataset_root(f"{dataset_name}-{self.maker_name}")

    def _dataset_root(self) -> Path:
        return Path(self.dataset_root) if self.dataset_root is not None else self._default_dataset_root()

    def _effective_dataset_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "chat": True,
            "use_hf_tokenizer_chat_template": True,
        }
        if self.dataset_kwargs:
            kwargs.update(self.dataset_kwargs)
        return kwargs

    def _output_path(self, root: Path, output_name: str) -> Path:
        return root / f"{output_name}.jsonl"

    def _needs_write(self, root: Path, output_name: str) -> bool:
        output_path = self._output_path(root, output_name)
        if output_path.exists() and not self.rewrite:
            print_rank_0(f"Skipping HF text SFT {output_name} data preparation - already exists: {output_path}")
            return False
        return True

    def _load_examples(self, *, split: str, extra_kwargs: dict[str, Any] | None) -> list[dict[str, Any]]:
        kwargs = dict(self.maker_kwargs or {})
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        kwargs.setdefault("split", split)
        examples = get_hf_dataset_maker(self.maker_name)(**kwargs)
        if not isinstance(examples, list) or len(examples) == 0:
            raise ValueError(f"Maker '{self.maker_name}' returned no examples for split='{split}'")
        return examples

    def _write_examples(self, *, root: Path, output_name: str, examples: list[dict[str, Any]]) -> None:
        output_path = root / f"{output_name}.jsonl"
        root.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fp:
            for example in examples:
                fp.write(json.dumps(example, ensure_ascii=False) + "\n")
        print_rank_0(f"Prepared HF text SFT {output_name} data at {output_path}")

    def _write_split(self, *, root: Path, output_name: str, split: str, extra_kwargs: dict[str, Any] | None) -> None:
        if not self._needs_write(root, output_name):
            return
        self._write_examples(
            root=root,
            output_name=output_name,
            examples=self._load_examples(split=split, extra_kwargs=extra_kwargs),
        )

    def _split_training_for_validation(
        self, examples: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if self.val_proportion is None or not 0.0 < self.val_proportion < 1.0:
            raise ValueError("val_proportion must be between 0 and 1 when deriving validation from training data.")
        split_dataset = Dataset.from_list(examples).train_test_split(test_size=self.val_proportion, seed=self.seed)
        return list(split_dataset["train"]), list(split_dataset["test"])

    def _write_train_and_validation_from_train(self, root: Path) -> None:
        write_train = self._needs_write(root, "training")
        write_validation = self._needs_write(root, "validation")
        if not write_train and not write_validation:
            return

        train_examples, validation_examples = self._split_training_for_validation(
            self._load_examples(split="train", extra_kwargs=None)
        )
        if write_train:
            self._write_examples(root=root, output_name="training", examples=train_examples)
        if write_validation:
            self._write_examples(root=root, output_name="validation", examples=validation_examples)

    def _prepare_jsonl_data(self, root: Path) -> None:
        if self.do_validation and self.val_proportion is not None and self.val_maker_kwargs is None:
            self._write_train_and_validation_from_train(root)
        else:
            self._write_split(root=root, output_name="training", split="train", extra_kwargs=None)

        if self.do_validation:
            if self.val_proportion is None or self.val_maker_kwargs is not None:
                self._write_split(
                    root=root,
                    output_name="validation",
                    split="validation",
                    extra_kwargs=self.val_maker_kwargs,
                )
        if self.do_test:
            self._write_split(root=root, output_name="test", split="test", extra_kwargs=self.test_maker_kwargs)

    def build_datasets(self, context: DatasetBuildContext) -> tuple[Any | None, Any | None, Any | None]:
        if context.tokenizer is None:
            raise ValueError("HFTextSFTDatasetProvider requires a tokenizer in DatasetBuildContext.")

        root = self._dataset_root()
        if get_rank_safe() == 0:
            self._prepare_jsonl_data(root)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return tuple(
            FinetuningDatasetBuilder(
                dataset_root=root,
                tokenizer=context.tokenizer,
                seq_length=self.seq_length,
                seed=self.seed,
                memmap_workers=self.memmap_workers,
                max_train_samples=self.max_train_samples,
                enable_offline_packing=self.enable_offline_packing,
                offline_packing_specs=self.offline_packing_specs,
                dataset_kwargs=self._effective_dataset_kwargs(),
                do_validation=self.do_validation,
                do_test=self.do_test,
            ).build()
        )
