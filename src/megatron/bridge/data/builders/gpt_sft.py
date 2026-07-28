# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Serializable config and runtime builder for text-only GPT SFT datasets."""

import hashlib
import json
import logging
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union

import torch
from megatron.core.msc_utils import MultiStorageClientFeature
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tokenizers.text.libraries import HuggingFaceTokenizer

from megatron.bridge.data.base import DataloaderConfig, validate_declarative_mapping
from megatron.bridge.data.datasets.gpt_sft import GPTSFTChatDataset, GPTSFTDataset, get_dataset_root
from megatron.bridge.data.packing import PackedSequenceSpecs
from megatron.bridge.data.packing.paths import (
    is_packed_parquet_spec,
    resolve_packed_parquet_paths,
    resolve_packed_parquet_paths_with_retry,
)
from megatron.bridge.data.sft_processing import (
    ChatSFTPreprocessingConfig,
    PromptCompletionSFTPreprocessingConfig,
    SFTPreprocessingConfig,
    normalize_sft_examples,
    validate_sft_preprocessing_config,
)
from megatron.bridge.data.sources.hf import (
    HFDatasetSourceConfig,
    hf_dataset_supports_split,
    load_and_adapt_hf_dataset,
    resolve_hf_dataset_source,
)
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer
from megatron.bridge.utils.common_utils import get_rank_safe, print_rank_0


logger = logging.getLogger(__name__)

_SEMANTIC_DATASET_KWARGS = {
    "add_bos",
    "add_eos",
    "add_sep",
    "answer_only_loss",
    "chat",
    "chat_loss_mode",
    "label_key",
    "prompt_completion_config",
    "prompt_template",
    "truncation_field",
    "truncation_method",
    "use_hf_tokenizer_chat_template",
}


def _default_gpt_sft_preprocessing() -> PromptCompletionSFTPreprocessingConfig:
    """Return preprocessing compatible with the established local JSONL schema."""
    return PromptCompletionSFTPreprocessingConfig(
        prompt_column="input",
        completion_column="output",
        separator=" ",
    )


def _packing_fingerprint(config: "GPTSFTDatasetConfig", dataset_kwargs: dict[str, Any] | None) -> str:
    """Fingerprint every setting that can change builder-managed packed rows."""
    preprocessing = resolve_gpt_sft_preprocessing(config)
    packing_identity = {
        "seq_length": config.seq_length,
        "seed": config.seed,
        "preprocessing_explicit": config.preprocessing is not None,
        "preprocessing_type": type(preprocessing).__name__,
        "preprocessing": asdict(preprocessing),
        "dataset_kwargs": dataset_kwargs,
    }
    return hashlib.sha256(
        json.dumps(packing_identity, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:12]


@dataclass(kw_only=True)
class GPTSFTDatasetConfig(DataloaderConfig):
    """Serializable configuration for text-only ``GPTSFTDataset`` construction.

    Exactly one source is required: ``dataset_root`` selects existing local
    JSONL/packed artifacts, while ``hf_dataset`` selects a declarative Hugging
    Face source that is materialized before construction. New callers should
    set ``preprocessing`` explicitly. ``None`` preserves the established local
    prompt-completion and Hugging Face chat defaults for compatibility.
    """

    seq_length: int
    dataset_root: str | Path | None = None
    hf_dataset: HFDatasetSourceConfig | None = None
    hf_validation_dataset: HFDatasetSourceConfig | None = None
    hf_test_dataset: HFDatasetSourceConfig | None = None
    hf_output_root: str | Path | None = None
    hf_validation_proportion: float | None = None
    hf_rewrite: bool = False
    seed: int = 1234
    memmap_workers: int = 1
    max_train_samples: int | None = None
    preprocessing: SFTPreprocessingConfig | None = None
    enable_offline_packing: bool = False
    offline_packing_specs: PackedSequenceSpecs | None = None
    dataset_kwargs: dict[str, Any] | None = None
    do_validation: bool = True
    do_test: bool = True
    dataloader_type: Literal["single", "cyclic", "batch", "external"] | None = "batch"

    def validate(self) -> None:
        """Validate source selection and text-only SFT settings."""
        has_local_source = self.dataset_root is not None
        has_hf_source = self.hf_dataset is not None
        if has_local_source == has_hf_source:
            raise ValueError("Exactly one text-only SFT source must be set: dataset_root or hf_dataset.")
        if has_local_source and not str(self.dataset_root).strip():
            raise ValueError("dataset_root must be a non-empty path.")
        hf_only_fields_set = (
            any(
                value is not None
                for value in (
                    self.hf_validation_dataset,
                    self.hf_test_dataset,
                    self.hf_output_root,
                    self.hf_validation_proportion,
                )
            )
            or self.hf_rewrite
        )
        if has_local_source and hf_only_fields_set:
            raise ValueError("Hugging Face split and materialization settings require hf_dataset, not dataset_root.")
        if self.hf_dataset is not None:
            self.hf_dataset.validate()
        if self.hf_validation_dataset is not None:
            self.hf_validation_dataset.validate()
        if self.hf_test_dataset is not None:
            self.hf_test_dataset.validate()
        if self.hf_output_root is not None and not str(self.hf_output_root).strip():
            raise ValueError("hf_output_root must be a non-empty path when set.")
        if self.hf_validation_proportion is not None and not 0.0 < self.hf_validation_proportion < 1.0:
            raise ValueError("hf_validation_proportion must be between 0 and 1.")
        if self.hf_validation_dataset is not None and self.hf_validation_proportion is not None:
            raise ValueError("Set either hf_validation_dataset or hf_validation_proportion, not both.")
        if (
            self.hf_dataset is not None
            and self.do_validation
            and self.hf_validation_dataset is None
            and self.hf_validation_proportion is None
            and not hf_dataset_supports_split(self.hf_dataset, "validation")
        ):
            raise ValueError(
                "The selected Hugging Face source has no validation split; disable validation or set one."
            )
        if (
            self.hf_dataset is not None
            and self.do_test
            and self.hf_test_dataset is None
            and not hf_dataset_supports_split(self.hf_dataset, "test")
        ):
            raise ValueError("The selected Hugging Face source has no test split; disable test or set one.")
        if not self.do_validation and (
            self.hf_validation_dataset is not None or self.hf_validation_proportion is not None
        ):
            raise ValueError("Hugging Face validation settings require do_validation=True.")
        if not self.do_test and self.hf_test_dataset is not None:
            raise ValueError("hf_test_dataset requires do_test=True.")
        if self.seq_length <= 0:
            raise ValueError("seq_length must be greater than 0.")
        validate_sft_preprocessing_config(resolve_gpt_sft_preprocessing(self))
        if self.enable_offline_packing and self.offline_packing_specs is None:
            raise ValueError("offline_packing_specs must be set when enable_offline_packing=True.")
        if self.offline_packing_specs is not None and not self.enable_offline_packing:
            raise ValueError("enable_offline_packing must be True when offline_packing_specs is set.")
        if self.offline_packing_specs is not None and self.offline_packing_specs.packed_sequence_size <= 0:
            raise ValueError("offline_packing_specs.packed_sequence_size must be greater than 0.")
        if self.hf_rewrite and self.offline_packing_specs is not None:
            explicit_packed_paths = (
                self.offline_packing_specs.packed_train_data_path,
                self.offline_packing_specs.packed_val_data_path,
                self.offline_packing_specs.packed_metadata_path,
            )
            if any(path is not None for path in explicit_packed_paths):
                raise ValueError(
                    "hf_rewrite cannot safely replace explicit packed data paths; use builder-managed packed paths."
                )
        validate_declarative_mapping(self.dataset_kwargs, field_name="dataset_kwargs")
        if self.dataset_kwargs is not None and "max_num_samples" in self.dataset_kwargs:
            raise ValueError("Set max_train_samples directly; dataset_kwargs must not contain max_num_samples.")
        semantic_kwargs = _SEMANTIC_DATASET_KWARGS.intersection(self.dataset_kwargs or {})
        if semantic_kwargs:
            keys = ", ".join(sorted(semantic_kwargs))
            if self.preprocessing is not None:
                raise ValueError(
                    f"Do not combine explicit preprocessing with deprecated semantic dataset_kwargs: {keys}."
                )
            warnings.warn(
                "SFT preprocessing through dataset_kwargs is deprecated; set preprocessing to "
                "ChatSFTPreprocessingConfig or PromptCompletionSFTPreprocessingConfig instead. "
                f"Deprecated keys: {keys}.",
                DeprecationWarning,
                stacklevel=2,
            )

    def finalize(self) -> None:
        """Finalize dataloader settings and validate this config."""
        super().finalize()
        self.validate()


# =============================================================================
# Deprecated compatibility APIs
# =============================================================================


@dataclass(kw_only=True)
class FinetuningDatasetConfig(GPTSFTDatasetConfig):
    """Deprecated compatibility name for :class:`GPTSFTDatasetConfig`."""

    def __post_init__(self) -> None:
        warnings.warn(
            "FinetuningDatasetConfig is deprecated; use megatron.bridge.data.builders.GPTSFTDatasetConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )


def resolve_gpt_sft_preprocessing(config: GPTSFTDatasetConfig) -> SFTPreprocessingConfig:
    """Resolve explicit preprocessing or source-compatible legacy defaults."""
    if config.preprocessing is not None:
        return config.preprocessing
    if config.hf_dataset is not None:
        return ChatSFTPreprocessingConfig()
    return _default_gpt_sft_preprocessing()


def resolve_gpt_sft_dataset_root(config: GPTSFTDatasetConfig) -> str | Path:
    """Resolve the local JSONL root for the configured source."""
    config.validate()
    if config.dataset_root is not None:
        return config.dataset_root

    source = config.hf_dataset
    assert source is not None
    if config.hf_output_root is not None:
        return Path(config.hf_output_root)
    preprocessing = resolve_gpt_sft_preprocessing(config)
    materialization_identity = {
        "source": asdict(resolve_hf_dataset_source(source)),
        "validation_source": (
            asdict(resolve_hf_dataset_source(config.hf_validation_dataset)) if config.hf_validation_dataset else None
        ),
        "test_source": (asdict(resolve_hf_dataset_source(config.hf_test_dataset)) if config.hf_test_dataset else None),
        "validation_proportion": config.hf_validation_proportion,
        "validation_seed": config.seed if config.hf_validation_proportion is not None else None,
        "do_validation": config.do_validation,
        "do_test": config.do_test,
        "preprocessing": asdict(preprocessing),
        "preprocessing_type": type(preprocessing).__name__,
    }
    encoded_identity = json.dumps(
        materialization_identity,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    fingerprint = hashlib.sha256(encoded_identity).hexdigest()[:16]
    resolved_source = resolve_hf_dataset_source(source)
    adapter_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", resolved_source.schema_adapter or "native").strip("-")
    return get_dataset_root(f"hf-sft-{adapter_name}-{fingerprint}")


def normalize_gpt_sft_dataset_kwargs(config: GPTSFTDatasetConfig) -> dict[str, Any]:
    """Return runtime dataset kwargs for the selected preprocessing variant."""
    config.validate()
    dataset_kwargs = dict(config.dataset_kwargs or {})
    if config.preprocessing is None:
        if config.hf_dataset is not None:
            return {
                "chat": True,
                "use_hf_tokenizer_chat_template": True,
                **dataset_kwargs,
            }
        return dataset_kwargs
    preprocessing = resolve_gpt_sft_preprocessing(config)
    if isinstance(preprocessing, ChatSFTPreprocessingConfig):
        preprocessing_kwargs = {
            "chat": True,
            "use_hf_tokenizer_chat_template": True,
            "chat_loss_mode": preprocessing.loss_mode,
        }
    else:
        preprocessing_kwargs = {
            "chat": False,
            "prompt_completion_config": preprocessing,
        }
    return {**preprocessing_kwargs, **dataset_kwargs}


def _load_hf_examples(
    source: HFDatasetSourceConfig,
    preprocessing: SFTPreprocessingConfig,
) -> list[dict[str, Any]]:
    return normalize_sft_examples(load_and_adapt_hf_dataset(source), preprocessing)


def _write_hf_examples(root: Path, output_name: str, examples: list[dict[str, Any]]) -> None:
    output_path = root / f"{output_name}.jsonl"
    root.mkdir(parents=True, exist_ok=True)
    for index_path in _hf_jsonl_index_paths(output_path):
        index_path.unlink(missing_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for example in examples:
            output_file.write(json.dumps(example, ensure_ascii=False) + "\n")
    print_rank_0(f"Prepared Hugging Face text SFT {output_name} data at {output_path}")


def _hf_jsonl_index_paths(output_path: Path) -> tuple[Path, Path]:
    """Return memmap index sidecars associated with one materialized JSONL split."""
    return Path(f"{output_path}.idx.npy"), Path(f"{output_path}.idx.info")


def _remove_hf_materialized_split(root: Path, output_name: str) -> None:
    """Remove one materialized split and any stale memmap index sidecars."""
    output_path = root / f"{output_name}.jsonl"
    output_path.unlink(missing_ok=True)
    for index_path in _hf_jsonl_index_paths(output_path):
        index_path.unlink(missing_ok=True)


def _needs_hf_write(config: GPTSFTDatasetConfig, root: Path, output_name: str) -> bool:
    output_path = root / f"{output_name}.jsonl"
    if output_path.exists() and not config.hf_rewrite:
        print_rank_0(f"Skipping Hugging Face text SFT {output_name} preparation - already exists: {output_path}")
        return False
    return True


def _materialize_hf_split(
    config: GPTSFTDatasetConfig,
    source: HFDatasetSourceConfig,
    root: Path,
    *,
    output_name: str,
) -> None:
    if _needs_hf_write(config, root, output_name):
        _write_hf_examples(
            root,
            output_name,
            _load_hf_examples(source, resolve_gpt_sft_preprocessing(config)),
        )


def materialize_hf_dataset(config: GPTSFTDatasetConfig, root: Path) -> None:
    """Materialize and normalize a Hugging Face source as JSONL splits."""
    source = config.hf_dataset
    if source is None:
        raise ValueError("materialize_hf_dataset requires an hf_dataset source.")
    if config.hf_rewrite:
        if not config.do_validation:
            _remove_hf_materialized_split(root, "validation")
        if not config.do_test:
            _remove_hf_materialized_split(root, "test")

    derive_validation = (
        config.do_validation and config.hf_validation_proportion is not None and config.hf_validation_dataset is None
    )
    if derive_validation:
        write_train = _needs_hf_write(config, root, "training")
        write_validation = _needs_hf_write(config, root, "validation")
        if write_train or write_validation:
            from datasets import Dataset

            examples = _load_hf_examples(source, resolve_gpt_sft_preprocessing(config))
            split_dataset = Dataset.from_list(examples).train_test_split(
                test_size=config.hf_validation_proportion,
                seed=config.seed,
            )
            if write_train:
                _write_hf_examples(root, "training", list(split_dataset["train"]))
            if write_validation:
                _write_hf_examples(root, "validation", list(split_dataset["test"]))
    else:
        _materialize_hf_split(config, source, root, output_name="training")

    if config.do_validation and not derive_validation:
        validation_source = config.hf_validation_dataset or source.with_split("validation")
        _materialize_hf_split(
            config,
            validation_source,
            root,
            output_name="validation",
        )
    if config.do_test:
        test_source = config.hf_test_dataset or source.with_split("test")
        _materialize_hf_split(
            config,
            test_source,
            root,
            output_name="test",
        )


def build_gpt_sft_split(
    path: str | Path,
    *,
    tokenizer: MegatronTokenizer,
    seq_length: int,
    memmap_workers: int,
    seed: int,
    packed_sequence_size: int,
    pack_metadata_path: str | Path | None = None,
    pad_cu_seqlens: bool = False,
    pad_seq_to_mult: int | None = None,
    is_test: bool = False,
    dataset_kwargs: dict[str, Any] | None = None,
) -> Any | None:
    """Build one GPT SFT split from a local JSONL or packed-data path."""
    path_str = str(path)
    if is_packed_parquet_spec(path_str):
        try:
            # Retry with directory-metadata refresh: on NFS filesystems a non-producer
            # node can transiently see zero files right after rank 0 wrote them (#4207).
            path_exists = bool(resolve_packed_parquet_paths_with_retry(path_str))
        except ValueError:
            path_exists = False
    elif MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        path_exists = msc.Path(path_str).exists()
    else:
        path_exists = Path(path_str).exists()

    if not path_exists:
        print_rank_0(f"Warning: Dataset path {path} does not exist")
        return None

    is_not_packing = packed_sequence_size <= 0
    effective_metadata_path = None
    if not is_not_packing:
        if pad_cu_seqlens:
            effective_metadata_path = pack_metadata_path
        elif not is_packed_parquet_spec(path_str):
            effective_metadata_path = pack_metadata_path

    options = dict(dataset_kwargs or {})
    chat = options.pop("chat", False)
    use_hf_tokenizer_chat_template = options.pop("use_hf_tokenizer_chat_template", False)
    chat_loss_mode = options.pop("chat_loss_mode", "assistant")
    tool_schemas = options.pop("tool_schemas", None)
    dataset_init_kwargs = {
        "file_path": path_str,
        "tokenizer": tokenizer,
        "max_seq_length": seq_length if is_not_packing else packed_sequence_size,
        "memmap_workers": memmap_workers,
        "hf_dataset": options.pop("hf_dataset", False),
        "global_sample_mapping": options.pop("global_sample_mapping", False),
        "add_bos": options.pop("add_bos", False),
        "add_eos": options.pop("add_eos", True),
        "add_sep": options.pop("add_sep", False),
        "seed": seed,
        "label_key": options.pop("label_key", "output"),
        "answer_only_loss": options.pop("answer_only_loss", True),
        "truncation_field": options.pop("truncation_field", "input"),
        "pad_to_max_length": options.pop("pad_to_max_length", False),
        "index_mapping_dir": options.pop("index_mapping_dir", None),
        "prompt_template": options.pop("prompt_template", "{input} {output}"),
        "truncation_method": options.pop("truncation_method", "right"),
        "get_attention_mask_from_fusion": options.pop("get_attention_mask_from_fusion", True),
        "prompt_completion_config": options.pop("prompt_completion_config", None),
        "is_test": is_test,
    }

    if path_str.lower().endswith(".npy"):
        from megatron.bridge.data.packing.gpt_sft import GPTSFTPackedDataset

        return GPTSFTPackedDataset(
            pack_metadata_file_path=effective_metadata_path,
            pad_cu_seqlens=pad_cu_seqlens,
            pad_seq_to_mult=pad_seq_to_mult,
            **dataset_init_kwargs,
            **options,
        )

    if is_packed_parquet_spec(path_str):
        from megatron.bridge.data.packing.parquet import GPTSFTPackedParquetDataset

        return GPTSFTPackedParquetDataset(
            pack_metadata_file_path=effective_metadata_path,
            pad_cu_seqlens=pad_cu_seqlens,
            pad_seq_to_mult=pad_seq_to_mult,
            **dataset_init_kwargs,
            **options,
        )

    if chat:
        return GPTSFTChatDataset(
            **dataset_init_kwargs,
            use_hf_tokenizer_chat_template=use_hf_tokenizer_chat_template,
            loss_mode=chat_loss_mode,
            tool_schemas=tool_schemas,
            **options,
        )
    return GPTSFTDataset(**dataset_init_kwargs, **options)


class GPTSFTDatasetBuilder:
    """Runtime builder for :class:`GPTSFTDatasetConfig`.

    The config remains serializable and declarative. This builder resolves the
    selected source, performs any Hugging Face materialization or offline
    packing, and constructs the runtime GPT SFT datasets.

    Args:
        config: Serializable GPT SFT dataset configuration.
        tokenizer: Tokenizer used to preprocess text.
    """

    def __init__(
        self,
        config: GPTSFTDatasetConfig,
        tokenizer: MegatronTokenizer,
    ) -> None:
        if tokenizer is None:
            raise ValueError("GPTSFTDatasetBuilder requires an initialized tokenizer.")
        config.validate()
        dataset_root = resolve_gpt_sft_dataset_root(config)
        self._source_root = Path(dataset_root) if config.hf_dataset is not None else None

        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            self.dataset_root = msc.Path(dataset_root)
        else:
            self.dataset_root = Path(dataset_root)
        self.tokenizer = tokenizer
        self.config = config
        self.seq_length = config.seq_length
        self.seed = config.seed
        self.memmap_workers = config.memmap_workers
        self.max_train_samples = config.max_train_samples
        self.enable_offline_packing = config.enable_offline_packing
        self.offline_packing_specs = config.offline_packing_specs
        self.packed_sequence_size = (
            -1 if config.offline_packing_specs is None else config.offline_packing_specs.packed_sequence_size
        )
        self.dataset_kwargs = normalize_gpt_sft_dataset_kwargs(config)
        self._pad_cu_seqlens = (
            False if config.offline_packing_specs is None else config.offline_packing_specs.pad_cu_seqlens
        )
        self._pad_seq_to_mult = (
            None if config.offline_packing_specs is None else config.offline_packing_specs.pad_seq_to_mult
        )
        self._num_tokenizer_workers = (
            -1 if config.offline_packing_specs is None else config.offline_packing_specs.num_tokenizer_workers
        )
        self._rewrite_packed_data = config.hf_dataset is not None and config.hf_rewrite
        self._packing_fingerprint = _packing_fingerprint(config, config.dataset_kwargs)

        self.do_validation = config.do_validation
        self.do_test = config.do_test

        print_rank_0(f"Building GPTSFTDatasetBuilder with root={self.dataset_root}")

        if self.packed_sequence_size > 0:
            print_rank_0(f"Using packed sequences with size {self.packed_sequence_size}")

    def prepare_data(self) -> None:
        """Materialize the selected source and prepare packed data if needed.

        Call this entry point on one rank before dataset construction. It is
        also used by the standalone pre-packing script.
        """
        if self.config.hf_dataset is not None:
            assert self._source_root is not None
            materialize_hf_dataset(self.config, self._source_root)
        self.prepare_packed_data()

    def prepare_packed_data(self) -> None:
        """Prepare packed sequence data files if configured.

        Skips preparation if:
        - packed_sequence_size <= 0 (packing disabled)
        - packed data files already exist (parquet or legacy .npy), unless
          ``hf_rewrite`` requested regeneration
        """
        if self.packed_sequence_size <= 0:
            return

        if self._rewrite_packed_data:
            metadata_path = self.pack_metadata
            if metadata_path.exists():
                with metadata_path.open("w") as metadata_file:
                    json.dump([], metadata_file)
            if not self.do_validation:
                self._remove_packed_path(self.validation_path_packed)

        self._prepare_packed_split(
            split_name="training",
            packed_path=self.train_path_packed,
            input_path=self.train_path,
        )

        if not self.do_validation:
            return

        self._prepare_packed_split(
            split_name="validation",
            packed_path=self.validation_path_packed,
            input_path=self.validation_path,
        )

    def _prepare_packed_split(
        self,
        split_name: str,
        packed_path: Union[str, Path],
        input_path: Path,
    ) -> None:
        """Prepare a single packed data split if it doesn't already exist.

        Args:
            split_name: Name of the split (for logging).
            packed_path: Output path for the packed data.
            input_path: Input path to the raw dataset.
        """
        from megatron.bridge.data.packing.offline import prepare_gpt_sft_packed_data

        if self._packed_path_exists(packed_path) and not self._rewrite_packed_data:
            print_rank_0(f"Skipping packed {split_name} data preparation - already exists: {packed_path}")
            return

        packed_path_str = str(packed_path)
        if packed_path_str.lower().endswith(".npy"):
            warnings.warn(
                "Automatic .npy packed sequence preparation is deprecated and will be removed in the next release. "
                "Please use packed parquet format instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            return

        print_rank_0(f"Preparing packed {split_name} data at {packed_path}")
        prepare_gpt_sft_packed_data(
            input_path=input_path,
            output_path=packed_path,
            output_metadata_path=self.pack_metadata,
            packed_sequence_size=self.packed_sequence_size,
            tokenizer=self.tokenizer,
            max_seq_length=self.seq_length,
            seed=self.seed,
            dataset_kwargs=self.dataset_kwargs,
            pad_seq_to_mult=self._pad_seq_to_mult,
            num_tokenizer_workers=self._num_tokenizer_workers,
            dataset_builder=build_gpt_sft_split,
        )

    def _packed_path_exists(self, path: Union[str, Path]) -> bool:
        """Check if a packed data path exists.

        For .npy files: check file exists
        For packed parquet specs: check if resolution returns non-empty

        Args:
            path: The path to check

        Returns:
            True if the packed data exists
        """
        path_str = str(path)

        # For packed parquet specs, check if resolution returns files
        if is_packed_parquet_spec(path_str):
            try:
                resolved = resolve_packed_parquet_paths(path_str)
                return len(resolved) > 0
            except ValueError:
                return False

        # For .npy or other files, check existence
        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            return msc.Path(path_str).is_file()
        else:
            return Path(path_str).is_file()

    def _remove_packed_path(self, path: Union[str, Path]) -> None:
        """Remove one builder-managed packed artifact if it exists."""
        path_str = str(path)
        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            path_obj = msc.Path(path_str)
            if path_obj.exists():
                path_obj.unlink()
        else:
            Path(path_str).unlink(missing_ok=True)

    def build(self) -> list[Optional[Any]]:
        """Build train, validation, and test datasets.

        This method creates the necessary datasets based on the configuration.
        It first ensures data preparation (e.g., packing) is done (on rank 0),
        then builds the datasets potentially using the prepared files.

        Returns:
            A list containing the train, validation, and test datasets.
            Elements can be None if the corresponding data file doesn't exist
            or if dataset building is skipped for the split.
        """
        # Prepare packed data if needed
        if get_rank_safe() == 0:
            self.prepare_data()

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # This needs to be called on all ranks
        datasets: list[Optional[Any]] = self._build_datasets()
        return datasets

    def _build_datasets(self) -> list[Optional[Any]]:
        """Internal method to build all datasets.

        Returns:
            list[Optional[Any]]: The train, validation, and test datasets.
        """
        train_ds = build_gpt_sft_split(
            self.train_path if self.packed_sequence_size <= 0 else self.train_path_packed,
            tokenizer=self.tokenizer,
            seq_length=self.seq_length,
            memmap_workers=self.memmap_workers,
            seed=self.seed,
            packed_sequence_size=self.packed_sequence_size,
            pack_metadata_path=None if self.packed_sequence_size <= 0 else self.pack_metadata,
            pad_cu_seqlens=self._pad_cu_seqlens,
            pad_seq_to_mult=self._pad_seq_to_mult,
            dataset_kwargs={"max_num_samples": self.max_train_samples, **self.dataset_kwargs},
        )

        if self.do_validation:
            valid_ds = build_gpt_sft_split(
                self.validation_path if self.packed_sequence_size <= 0 else self.validation_path_packed,
                tokenizer=self.tokenizer,
                seq_length=self.seq_length,
                memmap_workers=self.memmap_workers,
                seed=self.seed,
                packed_sequence_size=self.packed_sequence_size,
                pack_metadata_path=None if self.packed_sequence_size <= 0 else self.pack_metadata,
                pad_cu_seqlens=self._pad_cu_seqlens,
                pad_seq_to_mult=self._pad_seq_to_mult,
                is_test=True,
                dataset_kwargs=self.dataset_kwargs,
            )
        else:
            valid_ds = None

        if self.do_test:
            test_ds = build_gpt_sft_split(
                self.test_path,
                tokenizer=self.tokenizer,
                seq_length=self.seq_length,
                memmap_workers=self.memmap_workers,
                seed=self.seed,
                packed_sequence_size=-1,
                is_test=True,
                dataset_kwargs=self.dataset_kwargs,
            )
        else:
            test_ds = None

        return [train_ds, valid_ds, test_ds]

    @property
    def train_path(self) -> Path:
        """Path to the training dataset file (training.jsonl)."""
        return self.dataset_root / "training.jsonl"

    @property
    def default_pack_path(self) -> Path:
        """The default directory path for storing packed sequence files.

        Constructed based on the dataset root and tokenizer model name.
        Creates the directory if it doesn't exist.

        Returns:
            The Path object for the default packing directory.
        """
        tokenizer_model_name = self._extract_tokenizer_model_name()
        default_pack_path = (
            self.dataset_root
            / "packed"
            / (f"{tokenizer_model_name}_pad_seq_to_mult{self._pad_seq_to_mult}_sft_{self._packing_fingerprint}")
        )
        if not default_pack_path.exists():
            try:
                # Shared filesystems can expose stale parent-dir state despite exist_ok=True.
                default_pack_path.mkdir(parents=True, exist_ok=True)
            except (FileExistsError, FileNotFoundError):
                pass
            logger.info(f"Using default path for packing files: {str(default_pack_path)}")

        return default_pack_path

    @property
    def pack_metadata(self) -> Path:
        """Path to the metadata file for packed sequences.

        Determined by `offline_packing_specs` or defaults based on the
        `default_pack_path` and `packed_sequence_size`.

        Returns:
            The Path object for the packed sequence metadata file.

        Raises:
            ValueError: If packed sequences are not configured.
        """
        if self.packed_sequence_size > 0:
            if self.offline_packing_specs.packed_metadata_path is not None:
                return self.offline_packing_specs.packed_metadata_path
            return self.default_pack_path / f"{self.packed_sequence_size}_metadata.jsonl"
        else:
            raise ValueError("pack_metadata invalid since packed sequence size is not specified.")

    @property
    def train_path_packed(self) -> Path:
        """Path to the packed training dataset file.

        Determined by `offline_packing_specs` or defaults based on the
        `default_pack_path` and `packed_sequence_size`.

        Returns:
            The Path object for the packed training data file.

        Raises:
            ValueError: If packed sequences are not configured.
        """
        if self.packed_sequence_size > 0:
            if self.offline_packing_specs.packed_train_data_path is not None:
                return self.offline_packing_specs.packed_train_data_path
            return self.default_pack_path / f"training_{self.packed_sequence_size}.idx.parquet"
        else:
            raise ValueError("`train_path_packed` invalid since packed sequence size is not specified.")

    @property
    def validation_path_packed(self) -> Path:
        """Path to the packed validation dataset file.

        Determined by `offline_packing_specs` or defaults based on the
        `default_pack_path` and `packed_sequence_size`.

        Returns:
            The Path object for the packed validation data file.

        Raises:
            ValueError: If packed sequences are not configured.
        """
        if self.packed_sequence_size > 0:
            if self.offline_packing_specs.packed_val_data_path is not None:
                return self.offline_packing_specs.packed_val_data_path
            return self.default_pack_path / f"validation_{self.packed_sequence_size}.idx.parquet"
        else:
            raise ValueError("`validation_path_packed` invalid since packed sequence size is not specified.")

    @property
    def validation_path(self) -> Path:
        """Path to the validation dataset file (validation.jsonl)."""
        return self.dataset_root / "validation.jsonl"

    @property
    def test_path(self) -> Path:
        """Path to the test dataset file (test.jsonl)."""
        return self.dataset_root / "test.jsonl"

    def _extract_tokenizer_model_name(self) -> str:
        """Automatically get the model name from model path."""
        # Legacy tokenizer compatibility
        tokenizer_cls = HuggingFaceTokenizer
        tokenizer_instance = self.tokenizer._tokenizer

        if self.offline_packing_specs and self.offline_packing_specs.tokenizer_model_name is not None:
            return self.offline_packing_specs.tokenizer_model_name
        elif isinstance(tokenizer_instance, tokenizer_cls):
            name = self.tokenizer.path

            if name.endswith("context/nemo_tokenizer"):
                # NEMO_HOME/hf_org/hf_model/context/nemo_tokenizer => hf_org--hf_model
                tokenizer_model_name = "--".join(name.split("/")[-4:-2])
            elif name.endswith("nemo_tokenizer"):
                # NEMO_HOME/hf_org/hf_model/nemo_tokenizer => hf_org--hf_model
                tokenizer_model_name = "--".join(name.split("/")[-3:-1])
            else:
                # hf_org/hf_model => hf_org--hf_model
                tokenizer_model_name = name.replace("/", "--")
            return tokenizer_model_name
        else:
            identifiers = getattr(self.tokenizer, "unique_identifiers", None)
            if not isinstance(identifiers, dict):
                identifiers = {"class": f"{type(self.tokenizer).__module__}.{type(self.tokenizer).__qualname__}"}
            encoded_identifiers = json.dumps(identifiers, sort_keys=True, separators=(",", ":"), default=str)
            fingerprint = hashlib.sha256(encoded_identifiers.encode("utf-8")).hexdigest()[:12]
            return f"unknown_tokenizer_{fingerprint}"


def gpt_sft_train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int],
    dataset_config: GPTSFTDatasetConfig,
    tokenizer: MegatronTokenizer | None = None,
    pg_collection: ProcessGroupCollection | None = None,
) -> tuple[Any | None, Any | None, Any | None]:
    """Build text-only SFT datasets through the canonical runtime builder."""
    del train_val_test_num_samples, pg_collection
    if tokenizer is None:
        raise ValueError("GPTSFTDatasetBuilder requires an initialized tokenizer.")
    return tuple(GPTSFTDatasetBuilder(config=dataset_config, tokenizer=tokenizer).build())


# =============================================================================
# Deprecated compatibility APIs
# =============================================================================


class FinetuningDatasetBuilder(GPTSFTDatasetBuilder):
    """Deprecated constructor-compatible adapter for :class:`GPTSFTDatasetBuilder`."""

    def __init__(
        self,
        dataset_root: str | Path,
        tokenizer: MegatronTokenizer,
        seq_length: int = 2048,
        seed: int = 1234,
        memmap_workers: int = 1,
        max_train_samples: int | None = None,
        enable_offline_packing: bool = False,
        offline_packing_specs: PackedSequenceSpecs | None = None,
        dataset_kwargs: dict[str, Any] | None = None,
        do_validation: bool = True,
        do_test: bool = True,
    ) -> None:
        legacy_dataset_kwargs = dict(dataset_kwargs or {})
        validate_declarative_mapping(legacy_dataset_kwargs, field_name="dataset_kwargs")
        warnings.warn(
            "FinetuningDatasetBuilder is deprecated; use megatron.bridge.data.builders.GPTSFTDatasetBuilder instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            config=GPTSFTDatasetConfig(
                dataset_root=dataset_root,
                seq_length=seq_length,
                seed=seed,
                memmap_workers=memmap_workers,
                max_train_samples=max_train_samples,
                enable_offline_packing=enable_offline_packing,
                offline_packing_specs=offline_packing_specs,
                dataset_kwargs=legacy_dataset_kwargs,
                do_validation=do_validation,
                do_test=do_test,
            ),
            tokenizer=tokenizer,
        )
