# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Dataset configuration utilities for recipes and training scripts."""

from dataclasses import replace
from functools import partial
from typing import Any, Callable, List, Literal, Optional, Tuple, TypeAlias

from megatron.bridge.data.builders import (
    ChatSFTPreprocessingConfig,
    DirectHFSFTDatasetConfig,
    GPTSFTDatasetConfig,
    HFDatasetSourceConfig,
    PromptCompletionSFTPreprocessingConfig,
    SFTPreprocessingConfig,
)
from megatron.bridge.data.loaders import get_blend_and_blend_per_split
from megatron.bridge.data.packing import PackedSequenceSpecs
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.dora import DoRA
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.training.config import (
    ConfigContainer,
    GPTDatasetConfig,
    MockGPTDatasetConfig,
)


_BLEND_TYPE = Optional[Tuple[List[str], Optional[List[float]]]]
_BLEND_PER_SPLIT_TYPE = Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
_SPLIT_TYPE = Optional[str]


def default_peft_config(peft_scheme: str | PEFT | None, **kwargs: Any) -> PEFT | None:
    """Create the default PEFT configuration for a finetuning recipe.

    Args:
        peft_scheme: PEFT scheme (``"lora"``, ``"dora"``), an existing PEFT
            instance, or ``None`` for full finetuning.
        **kwargs: Keyword arguments passed to the selected PEFT configuration.

    Returns:
        A PEFT configuration, or ``None`` for full finetuning.

    Raises:
        ValueError: If ``peft_scheme`` is not supported.
    """
    if peft_scheme is None:
        return None

    if isinstance(peft_scheme, PEFT):
        return peft_scheme

    if isinstance(peft_scheme, str):
        if peft_scheme.lower() == "none":
            return None
        if peft_scheme.lower() == "lora":
            return LoRA(**kwargs)
        if peft_scheme.lower() == "dora":
            return DoRA(**kwargs)
        raise ValueError(f"Unknown PEFT scheme: {peft_scheme}. Supported: 'lora', 'dora', or None")

    raise ValueError(f"Invalid peft type: {type(peft_scheme)}. Expected str, PEFT instance, or None")


def _text_hf_dataset_config(
    *,
    seq_length: int,
    source: HFDatasetSourceConfig,
    preprocessing: SFTPreprocessingConfig,
    validation_source: HFDatasetSourceConfig | None = None,
    test_source: HFDatasetSourceConfig | None = None,
    do_validation: bool = True,
    do_test: bool = False,
    enable_offline_packing: bool = False,
    offline_packing_specs: PackedSequenceSpecs | None = None,
    dataset_kwargs: dict[str, Any] | None = None,
    val_proportion: float | None = None,
    num_workers: int = 2,
) -> GPTSFTDatasetConfig:
    """Create an HF-backed text SFT config with optional offline packing."""
    return GPTSFTDatasetConfig(
        seq_length=seq_length,
        hf_dataset=source,
        hf_validation_dataset=validation_source,
        hf_test_dataset=test_source,
        hf_validation_proportion=val_proportion,
        do_validation=do_validation,
        do_test=do_test,
        preprocessing=preprocessing,
        enable_offline_packing=enable_offline_packing,
        offline_packing_specs=offline_packing_specs,
        dataset_kwargs=dataset_kwargs,
        seed=5678,
        dataloader_type="batch",
        num_workers=num_workers,
        data_sharding=True,
        pin_memory=True,
        persistent_workers=False,
    )


def default_squad_config(
    seq_length: int, enable_offline_packing: bool = True, pad_seq_to_mult: int = 1
) -> GPTSFTDatasetConfig:
    """Create the default SQuAD dataset configuration for finetuning recipes.

    Args:
        seq_length: Sequence length for the dataset.
        enable_offline_packing: Whether to enable offline packed-sequence preparation.
        pad_seq_to_mult: Multiple to pad each sequence to when packing.

    Returns:
        A dataset configuration for SQuAD finetuning.
    """
    dataset_kwargs = {}
    offline_packing_specs = None
    if enable_offline_packing:
        dataset_kwargs["pad_to_max_length"] = True
        offline_packing_specs = PackedSequenceSpecs(packed_sequence_size=seq_length, pad_seq_to_mult=pad_seq_to_mult)

    return _text_hf_dataset_config(
        source=HFDatasetSourceConfig(dataset_name="squad"),
        preprocessing=PromptCompletionSFTPreprocessingConfig(
            prompt_column="input",
            completion_column="output",
            separator=" ",
        ),
        seq_length=seq_length,
        enable_offline_packing=enable_offline_packing,
        offline_packing_specs=offline_packing_specs,
        dataset_kwargs=dataset_kwargs,
        val_proportion=0.1,
        num_workers=1,
    )


def default_tulu3_config(
    seq_length: int = 4096,
    enable_offline_packing: bool = False,
    pad_seq_to_mult: int = 1,
) -> GPTSFTDatasetConfig:
    """Create the default Tulu 3 SFT mixture dataset configuration.

    Args:
        seq_length: Maximum sequence length.
        enable_offline_packing: Whether to enable offline text SFT packing.
        pad_seq_to_mult: Sequence-length multiple used by offline packing.

    Returns:
        A chat SFT configuration for ``allenai/tulu-3-sft-mixture``.
    """
    offline_packing_specs = None
    if enable_offline_packing:
        offline_packing_specs = PackedSequenceSpecs(packed_sequence_size=seq_length, pad_seq_to_mult=pad_seq_to_mult)

    return _text_hf_dataset_config(
        source=HFDatasetSourceConfig(dataset_name="tulu3"),
        preprocessing=ChatSFTPreprocessingConfig(),
        seq_length=seq_length,
        enable_offline_packing=enable_offline_packing,
        offline_packing_specs=offline_packing_specs,
        val_proportion=0.05,
        num_workers=2,
    )


def default_openmathinstruct2_config(
    seq_length: int = 4096,
    enable_offline_packing: bool = False,
    pad_seq_to_mult: int = 1,
) -> GPTSFTDatasetConfig:
    """Create the default OpenMathInstruct-2 finetuning dataset.

    Args:
        seq_length: Maximum sequence length.
        enable_offline_packing: Whether to enable offline text SFT packing.
        pad_seq_to_mult: Sequence-length multiple used by offline packing.

    Returns:
        An OpenMathInstruct-2 dataset configuration.
    """
    offline_packing_specs = None
    if enable_offline_packing:
        offline_packing_specs = PackedSequenceSpecs(packed_sequence_size=seq_length, pad_seq_to_mult=pad_seq_to_mult)

    return _text_hf_dataset_config(
        source=HFDatasetSourceConfig(dataset_name="openmathinstruct2"),
        preprocessing=PromptCompletionSFTPreprocessingConfig(
            prompt_column="input",
            completion_column="output",
            separator=" ",
        ),
        seq_length=seq_length,
        enable_offline_packing=enable_offline_packing,
        offline_packing_specs=offline_packing_specs,
        val_proportion=0.05,
        num_workers=2,
    )


def default_gsm8k_config(
    seq_length: int = 2048,
    enable_offline_packing: bool = False,
    pad_seq_to_mult: int = 1,
) -> GPTSFTDatasetConfig:
    """Create the default GSM8K dataset configuration for finetuning recipes.

    Args:
        seq_length: Maximum sequence length.
        enable_offline_packing: Whether to enable offline text SFT packing.
        pad_seq_to_mult: Sequence-length multiple used by offline packing.

    Returns:
        A GSM8K dataset configuration.
    """
    offline_packing_specs = None
    if enable_offline_packing:
        offline_packing_specs = PackedSequenceSpecs(packed_sequence_size=seq_length, pad_seq_to_mult=pad_seq_to_mult)

    return _text_hf_dataset_config(
        source=HFDatasetSourceConfig(dataset_name="gsm8k"),
        preprocessing=PromptCompletionSFTPreprocessingConfig(
            prompt_column="input",
            completion_column="output",
            separator=" ",
        ),
        test_source=HFDatasetSourceConfig(dataset_name="gsm8k", split="test"),
        do_validation=False,
        do_test=True,
        seq_length=seq_length,
        enable_offline_packing=enable_offline_packing,
        offline_packing_specs=offline_packing_specs,
        num_workers=2,
    )


def default_openmathinstruct2_thinking_config(
    seq_length: int = 4096,
    enable_offline_packing: bool = False,
    pad_seq_to_mult: int = 1,
) -> GPTSFTDatasetConfig:
    """Create the thinking/chat variant of the OpenMathInstruct-2 dataset.

    Args:
        seq_length: Maximum sequence length.
        enable_offline_packing: Whether to enable offline text SFT packing.
        pad_seq_to_mult: Sequence-length multiple used by offline packing.

    Returns:
        An OpenMathInstruct-2 thinking dataset configuration.
    """
    config = default_openmathinstruct2_config(
        seq_length=seq_length,
        enable_offline_packing=enable_offline_packing,
        pad_seq_to_mult=pad_seq_to_mult,
    )
    assert config.hf_dataset is not None
    config.hf_dataset = HFDatasetSourceConfig(dataset_name="openmathinstruct2_thinking")
    config.preprocessing = ChatSFTPreprocessingConfig()
    return config


def get_blend_fields_from_data_paths(
    data_paths: Optional[List[str]] = None,
    data_args_path: Optional[str] = None,
    train_data_path: Optional[List[str]] = None,
    valid_data_path: Optional[List[str]] = None,
    test_data_path: Optional[List[str]] = None,
    per_split_data_args_path: Optional[str] = None,
    mock: bool = False,
) -> Tuple[_BLEND_TYPE, _BLEND_PER_SPLIT_TYPE, _SPLIT_TYPE]:
    """
    Common configuration logic for blend, blend_per_split, split dataset config fields.

    Handles mock and real data. If no path to data is provided, mock data will be used.
    Prioritizes `data_paths` over split data paths. For all of `data_paths`, `train_data_path`,
    `valid_data_path`, and `test_data_path`, two formats are accepted: either (1) a list of prefixes,
    e.g. ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], or (2) a flattened, zipped
    list of weights and prefixes, e.g. ["30", "path/to/dataset_1_prefix", "70", "path/to/dataset_2_prefix"]

    Args:
        data_paths (Optional[List[str]]): List of paths to dataset files.
        data_args_path (Optional[str]): Path to file containing data arguments.
        train_data_path (Optional[List[str]]): List of training data paths.
        valid_data_path (Optional[List[str]]): List of validation data paths.
        test_data_path (Optional[List[str]]): List of test data paths.
        per_split_data_args_path (Optional[str]): Path to JSON file with per-split data configuration.
        mock (bool): Whether to use mock data. If True, ignores data_paths.

    Returns:
        A tuple (blend, blend_per_split, split), the corresponding fields to be passed to GPTDatasetConfig.
    """
    has_any_data_config = any(
        [data_paths, data_args_path, train_data_path, valid_data_path, test_data_path, per_split_data_args_path]
    )

    if mock or not has_any_data_config:
        # Mock data configuration
        blend = None  # Will trigger mock mode automatically
        blend_per_split = None  # Will trigger mock mode automatically
        split = "1,1,1"  # Equal splits for testing
    else:
        # Real data configuration
        blend, blend_per_split = get_blend_and_blend_per_split(
            data_paths=data_paths,
            data_args_path=data_args_path,
            train_data_paths=train_data_path,
            valid_data_paths=valid_data_path,
            test_data_paths=test_data_path,
            per_split_data_args_path=per_split_data_args_path,
        )

        if blend_per_split is not None:
            # When using blend_per_split, split should be None
            split = None
        elif blend is not None:
            # When using regular blend, we can use split
            split = "9999,8,2"
        else:
            # No data provided, fall back to mock mode
            split = "1,1,1"

    return blend, blend_per_split, split


PublicDatasetConfig: TypeAlias = GPTDatasetConfig | GPTSFTDatasetConfig | DirectHFSFTDatasetConfig
DatasetPreset: TypeAlias = Callable[[ConfigContainer], PublicDatasetConfig]


def _resolve_seq_length(config: ConfigContainer) -> int:
    """Use the selected recipe's model sequence length for a dataset preset."""
    if hasattr(config, "model") and config.model is not None and hasattr(config.model, "seq_length"):
        return int(config.model.seq_length)
    return 4096


def _mock_dataset_config(config: ConfigContainer) -> MockGPTDatasetConfig:
    """Build the mock pretraining dataset preset."""
    return MockGPTDatasetConfig(
        seq_length=_resolve_seq_length(config),
        random_seed=1234,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        num_dataset_builder_threads=1,
        split="9999,8,2",
        data_sharding=True,
        dataloader_type="single",
        skip_getting_attention_mask_from_dataset=True,
    )


def _megatron_indexed_dataset_config(config: ConfigContainer) -> GPTDatasetConfig:
    """Build the Megatron indexed pretraining dataset preset."""
    return GPTDatasetConfig(
        seq_length=_resolve_seq_length(config),
        random_seed=1234,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        num_dataset_builder_threads=1,
        blend=None,
        blend_per_split=None,
        split="9999,8,2",
        data_sharding=True,
        dataloader_type="single",
        skip_getting_attention_mask_from_dataset=True,
    )


def _squad_dataset_config(config: ConfigContainer) -> GPTSFTDatasetConfig:
    """Build the SQuAD text SFT dataset preset."""
    return default_squad_config(seq_length=_resolve_seq_length(config), enable_offline_packing=False)


def _tulu3_dataset_config(config: ConfigContainer) -> GPTSFTDatasetConfig:
    """Build the Tulu 3 chat SFT dataset preset."""
    return default_tulu3_config(seq_length=_resolve_seq_length(config))


def _openmathinstruct2_dataset_config(config: ConfigContainer) -> GPTSFTDatasetConfig:
    """Build the OpenMathInstruct-2 prompt-completion preset."""
    return default_openmathinstruct2_config(seq_length=_resolve_seq_length(config))


def _openmathinstruct2_thinking_dataset_config(config: ConfigContainer) -> GPTSFTDatasetConfig:
    """Build the OpenMathInstruct-2 thinking/chat preset."""
    return default_openmathinstruct2_thinking_config(seq_length=_resolve_seq_length(config))


def _gsm8k_dataset_config(config: ConfigContainer) -> GPTSFTDatasetConfig:
    """Build the GSM8K text SFT dataset preset."""
    return default_gsm8k_config(seq_length=_resolve_seq_length(config))


def _local_jsonl_dataset_config(config: ConfigContainer) -> GPTSFTDatasetConfig:
    """Build the local prompt-completion JSONL config before path overrides."""
    return GPTSFTDatasetConfig(
        seq_length=_resolve_seq_length(config),
        dataset_root=None,
        preprocessing=PromptCompletionSFTPreprocessingConfig(
            prompt_column="input",
            completion_column="output",
            separator=" ",
        ),
        dataloader_type="batch",
        seed=5678,
    )


def _local_vlm_json_source(split: str) -> HFDatasetSourceConfig:
    """Build an override-ready local JSON source for one VLM split."""
    return HFDatasetSourceConfig(
        path_or_dataset="json",
        split=split,
        load_kwargs={"data_files": {split: None}},
    )


def _require_direct_hf_config(config: ConfigContainer, dataset_name: str) -> DirectHFSFTDatasetConfig:
    """Return the recipe's direct-HF config or reject an incompatible preset."""
    if not isinstance(config.dataset, DirectHFSFTDatasetConfig):
        raise ValueError(f"{dataset_name} requires a recipe using DirectHFSFTDatasetConfig.")
    return config.dataset


def _local_vlm_dataset_config(config: ConfigContainer) -> DirectHFSFTDatasetConfig:
    """Build an override-ready local JSON/JSONL VLM preset."""
    existing = _require_direct_hf_config(config, "local-vlm")
    return replace(
        existing,
        seq_length=_resolve_seq_length(config),
        source=_local_vlm_json_source("train"),
        validation_source=_local_vlm_json_source("validation"),
        test_source=_local_vlm_json_source("test"),
        do_validation=False,
        do_test=False,
    )


def _hf_vlm_dataset_config(
    config: ConfigContainer,
    *,
    public_name: str,
    hf_dataset_name: str,
    train_only: bool = False,
    supports_test: bool = False,
    adapter_kwargs: dict[str, object] | None = None,
) -> DirectHFSFTDatasetConfig:
    """Build a named direct-HF VLM dataset preset."""
    existing = _require_direct_hf_config(config, public_name)
    source = HFDatasetSourceConfig(dataset_name=hf_dataset_name, adapter_kwargs=adapter_kwargs)
    validation_source = None
    if train_only and existing.do_validation:
        validation_source = source.with_split("train[95%:]")
        source = source.with_split("train[:95%]")
    return replace(
        existing,
        seq_length=_resolve_seq_length(config),
        source=source,
        validation_source=validation_source,
        test_source=None,
        do_test=existing.do_test and supports_test,
    )


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "mock": _mock_dataset_config,
    "megatron-indexed": _megatron_indexed_dataset_config,
    "squad": _squad_dataset_config,
    "tulu3": _tulu3_dataset_config,
    "openmathinstruct2": _openmathinstruct2_dataset_config,
    "openmathinstruct2-thinking": _openmathinstruct2_thinking_dataset_config,
    "gsm8k": _gsm8k_dataset_config,
    "local-jsonl": _local_jsonl_dataset_config,
    "local-vlm": _local_vlm_dataset_config,
    "cord-v2": partial(
        _hf_vlm_dataset_config,
        public_name="cord-v2",
        hf_dataset_name="cord_v2",
        supports_test=True,
    ),
    "llava-video-178k": partial(
        _hf_vlm_dataset_config,
        public_name="llava-video-178k",
        hf_dataset_name="llava_video_178k",
        train_only=True,
        adapter_kwargs={"video_root_path": None},
    ),
    "medpix": partial(_hf_vlm_dataset_config, public_name="medpix", hf_dataset_name="medpix"),
    "raven": partial(
        _hf_vlm_dataset_config,
        public_name="raven",
        hf_dataset_name="raven",
        train_only=True,
    ),
    "rdr": partial(
        _hf_vlm_dataset_config,
        public_name="rdr",
        hf_dataset_name="rdr",
        train_only=True,
    ),
}


def build_dataset_config(config: ConfigContainer, dataset_name: str) -> PublicDatasetConfig:
    """Build a dataset config from a public preset name.

    Args:
        config: Recipe config supplying model and model-specific dataset defaults.
        dataset_name: Public dataset preset or local source selector.

    Returns:
        A new dataset config. Callers may then apply ordinary ``dataset.*``
        ConfigContainer overrides before validation and runtime builder selection.

    Raises:
        ValueError: If the name is unknown or the recipe's dataset config is incompatible.
    """
    try:
        preset = DATASET_PRESETS[dataset_name]
    except KeyError:
        choices = ", ".join(DATASET_PRESETS)
        raise ValueError(f"Unknown dataset name: '{dataset_name}'. Choose from: {choices}") from None
    return preset(config)


def dataset_train_mode(dataset_config: PublicDatasetConfig) -> Literal["pretrain", "finetune"]:
    """Return the training loop required by a built dataset config."""
    return "pretrain" if isinstance(dataset_config, GPTDatasetConfig) else "finetune"
