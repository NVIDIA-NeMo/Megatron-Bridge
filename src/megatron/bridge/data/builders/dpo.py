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

"""``ConfigContainer.dataset`` entry for DPO preference training.

``DPODatasetConfig`` + ``DPODatasetBuilder`` + the registered
``dpo_train_valid_test_datasets_provider`` follow the config-and-builder dataset
architecture (registered in ``data/utils.py``); loader construction is the DPO
branch of ``data/loaders.py``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from torch.utils.data import Subset
from transformers import AutoTokenizer

from megatron.bridge.data.base import DataloaderConfig, DatasetBuildContext
from megatron.bridge.data.datasets.preference import load_ref_logprobs
from megatron.bridge.data.datasets.preference_pair import PreferencePairDataset
from megatron.bridge.data.datasets.utils import _JSONLMemMapDataset
from megatron.bridge.data.sources.hf import HFDatasetSourceConfig, load_hf_dataset_source, prepare_hf_dataset_sources
from megatron.bridge.data.sources.jsonl import JSONLSourceConfig
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo


if TYPE_CHECKING:
    from megatron.core.process_groups_config import ProcessGroupCollection

    from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer


PreferenceSource = HFDatasetSourceConfig | JSONLSourceConfig

_JSONL_SUFFIXES = (".jsonl", ".json")


def _as_source(entry: "PreferenceSource | Mapping[str, Any]") -> Any:
    """Rebuild a source that an override round trip flattened into a mapping."""
    if not isinstance(entry, Mapping):
        return entry
    return JSONLSourceConfig(**dict(entry)) if "paths" in entry else HFDatasetSourceConfig(**dict(entry))


@dataclass(kw_only=True)
class DPODatasetConfig(DataloaderConfig):
    """Preference dataset plus its offline-scored ref-logprob artifact.

    ``source`` is an ``HFDatasetSourceConfig`` for anything HuggingFace ``datasets``
    can read, or a ``JSONLSourceConfig`` for JSONL that may live in object storage.
    The source identity, ``tokenizer_name``, and ``seq_length`` must equal the
    scoring run's values, which ``dpo_train`` checks against ``scoring_metadata.json``
    (where the length is recorded under its dataset-level name, ``max_seq_length``).

    A validation split is optional and mirrors the train shape: one ``validation_source``
    plus its own scored ``validation_ref_artifact`` (``tokenizer_name`` and
    ``seq_length`` are shared with training, so the validation split must be
    scored with the same values). Like the SFT dataset configs, a configured
    validation split is built only when the run actually validates
    (``validation.eval_iters``/``eval_interval`` > 0).

    Batch sizes stay row-denominated; pair halving happens only in the loader builder.
    """

    tokenizer_name: str
    seq_length: int
    """Truncation ceiling for tokenized pairs."""
    source: PreferenceSource
    ref_artifact: str | None = None
    """Scorer artifact directory or URL. Required to build datasets; only
    source-only helpers (``load_source``) may leave it unset."""
    validation_ref_artifact: str | None = None
    """Scorer artifact for the validation split. Required when a validation source is set."""
    validation_source: PreferenceSource | None = None
    validation_num_pairs: int = 0
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"
    prompt_key: str | None = None
    index_mapping_dir: str | None = None
    num_pairs: int = 0
    dataloader_type: Literal["batch"] | None = "batch"
    """Pinned: the batch sampler's cyclic wrap provides epochs beyond one dataset pass."""
    pad_seq_length_to_mult: int = 1
    """Round each batch's padded row length up to this multiple. Sequence parallelism needs
    TP | seq_len; ``dpo_train`` derives it from the model config, the scorer from its flags."""
    shuffle: bool = True

    def resolve_source(self, source: PreferenceSource, *, field: str) -> PreferenceSource:
        """Route a JSONL path to the memmap reader, then validate whichever source type results."""
        source = _as_source(source)
        if isinstance(source, HFDatasetSourceConfig) and (source.path_or_dataset or "").endswith(_JSONL_SUFFIXES):
            source = JSONLSourceConfig(paths=[source.path_or_dataset], index_mapping_dir=self.index_mapping_dir)
        if not isinstance(source, (HFDatasetSourceConfig, JSONLSourceConfig)):
            raise TypeError(f"{field} must be an HFDatasetSourceConfig or JSONLSourceConfig; got {source!r}.")

        source.validate()
        return source

    def validate(self) -> None:
        """Resolve and validate the sources; pin the dataloader type."""
        if self.dataloader_type != "batch":
            raise ValueError(f"DPO supports only dataloader_type='batch'; got {self.dataloader_type!r}.")
        if self.seq_length <= 0:
            raise ValueError("seq_length must be greater than 0.")

        source = self.resolve_source(self.source, field="source")
        validation_source = None

        if self.validation_source is not None:
            validation_source = self.resolve_source(self.validation_source, field="validation_source")
            if not self.validation_ref_artifact:
                raise ValueError(
                    "validation_ref_artifact must be set with a validation split: validation needs its own "
                    "offline-scored artifact."
                )
        elif self.validation_ref_artifact:
            raise ValueError("validation_ref_artifact is set but no validation split is: set validation_source.")

        self.source, self.validation_source = source, validation_source

    def finalize(self) -> None:
        """Finalize dataloader settings and validate. ``ref_artifact`` may be local or msc://."""
        super().finalize()
        self.validate()

    @property
    def has_validation_split(self) -> bool:
        """Whether a validation source is configured."""
        return self.validation_source is not None

    @staticmethod
    def source_identity(source: PreferenceSource) -> tuple[str, str | None]:
        """``(dataset, split)`` as recorded in and checked against ``scoring_metadata.json``.

        JSONL sources have no split: their paths name the rows directly.
        """
        if isinstance(source, JSONLSourceConfig):
            return ",".join(source.paths), None
        return source.dataset_name or source.path_or_dataset, source.split

    def load_tokenizer(self):
        """The HF tokenizer both the scorer and the trainer tokenize pairs with."""
        trust_remote_code = is_safe_repo(hf_path=self.tokenizer_name, trust_remote_code=self.trust_remote_code)
        return AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=trust_remote_code)

    def load_source(self, split: Literal["train", "validation"] = "train"):
        """Load the raw preference rows of ``split``, truncated to that split's ``num_pairs``.

        The offline scorer calls this too: both jobs must see the same rows in the same
        order for ``pair_id`` to line up between the artifact and training.
        """
        self.validate()
        if split == "validation":
            if self.validation_source is None:
                raise ValueError("No validation split is configured: set validation_source.")
            return self._load_rows(self.validation_source, self.validation_num_pairs)
        return self._load_rows(self.source, self.num_pairs)

    @staticmethod
    def _load_rows(source: PreferenceSource, num_pairs: int):
        if isinstance(source, JSONLSourceConfig):
            rows = _JSONLMemMapDataset(
                dataset_paths=source.paths,
                tokenizer=None,
                header_lines=0,
                index_mapping_dir=source.index_mapping_dir,
            )
        else:
            # Rank 0 materializes the HuggingFace cache first: concurrent builders on a
            # shared filesystem are not safe. No-op for single-process callers.
            prepare_hf_dataset_sources([source])
            rows = load_hf_dataset_source(source)

        if num_pairs and num_pairs < len(rows):
            # Truncate by position so row index == pair_id survives.
            rows = rows.select(range(num_pairs)) if hasattr(rows, "select") else Subset(rows, range(num_pairs))
        return rows


def build_preference_split(
    config: DPODatasetConfig,
    split: Literal["train", "validation"],
    tokenizer,
    ref_artifact: str | None = None,
) -> PreferencePairDataset:
    """Load one split's rows and wrap them as a pair dataset.

    The scorer builds without ``ref_artifact``; the trainer supplies the split's artifact,
    which must cover every loaded pair.
    """
    rows = config.load_source(split)
    ref_logprobs = load_ref_logprobs(ref_artifact, expected_num_pairs=len(rows)) if ref_artifact else None
    return PreferencePairDataset(
        rows,
        tokenizer,
        max_seq_length=config.seq_length,
        pad_seq_length_to_mult=config.pad_seq_length_to_mult,
        chosen_key=config.chosen_key,
        rejected_key=config.rejected_key,
        prompt_key=config.prompt_key,
        ref_logprobs=ref_logprobs,
    )


class DPODatasetBuilder:
    """Build the runtime preference-pair dataset with reference logprobs attached."""

    def __init__(self, config: DPODatasetConfig) -> None:
        config.validate()
        if not config.ref_artifact:
            raise ValueError(
                "ref_artifact is not set: training needs the offline scorer's artifact "
                "(score_reference_logprobs.py --output) to anchor the margins."
            )
        self.config = config

    def build(
        self,
        context: DatasetBuildContext,
    ) -> tuple[PreferencePairDataset, PreferencePairDataset | None, None]:
        tokenizer = self.config.load_tokenizer()
        train_dataset = build_preference_split(self.config, "train", tokenizer, self.config.ref_artifact)
        valid_dataset = (
            build_preference_split(self.config, "validation", tokenizer, self.config.validation_ref_artifact)
            if self.config.has_validation_split and context.valid_samples > 0
            else None
        )
        return train_dataset, valid_dataset, None


def dpo_train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int],
    dataset_config: DPODatasetConfig,
    tokenizer: "MegatronTokenizer | None" = None,
    pg_collection: "ProcessGroupCollection | None" = None,
) -> tuple[PreferencePairDataset, PreferencePairDataset | None, None]:
    """Build DPO preference datasets through the canonical runtime builder."""
    context = DatasetBuildContext(
        train_samples=train_val_test_num_samples[0],
        valid_samples=train_val_test_num_samples[1],
        test_samples=train_val_test_num_samples[2],
        tokenizer=tokenizer,
        pg_collection=pg_collection,
    )
    return DPODatasetBuilder(dataset_config).build(context)
