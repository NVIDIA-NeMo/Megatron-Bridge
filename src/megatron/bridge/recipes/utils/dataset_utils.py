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

import logging
from dataclasses import dataclass, replace
from typing import Callable, List, Literal, Optional, Tuple

from megatron.bridge.data.builders import (
    ChatSFTPreprocessingConfig,
    DirectHFSFTDatasetConfig,
    EnergonDatasetConfig,
    GPTSFTDatasetConfig,
    HFDatasetSourceConfig,
    PromptCompletionSFTPreprocessingConfig,
)
from megatron.bridge.data.loaders import get_blend_and_blend_per_split
from megatron.bridge.data.packing import PackedSequenceSpecs
from megatron.bridge.data.sources.hf import hf_dataset_supports_split
from megatron.bridge.recipes.utils.finetune_utils import (
    default_gsm8k_config,
    default_openmathinstruct2_config,
    default_openmathinstruct2_thinking_packed_config,
    default_squad_config,
)
from megatron.bridge.training.config import (
    ConfigContainer,
    GPTDatasetConfig,
    MockGPTDatasetConfig,
)


logger = logging.getLogger(__name__)


_BLEND_TYPE = Optional[Tuple[List[str], Optional[List[float]]]]
_BLEND_PER_SPLIT_TYPE = Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
_SPLIT_TYPE = Optional[str]


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


# ---------------------------------------------------------------------------
# Unified dataset type registry
# ---------------------------------------------------------------------------

DATASET_TYPES = [
    "llm-pretrain",
    "llm-pretrain-mock",
    "llm-finetune",
    "llm-finetune-preloaded",
    "vlm-energon",
    "vlm-hf",
]


@dataclass(frozen=True, kw_only=True)
class PublicDatasetSpec:
    """Launcher behavior owned by one public dataset name."""

    train_mode: Literal["pretrain", "finetune"]
    modality: Literal["text", "vlm"] = "text"
    supports_offline_packing: bool = False
    indexed_data: bool = False
    hf_dataset_name: str | None = None


PUBLIC_DATASETS: dict[str, PublicDatasetSpec] = {
    "mock": PublicDatasetSpec(train_mode="pretrain"),
    "megatron-indexed": PublicDatasetSpec(train_mode="pretrain", indexed_data=True),
    "squad": PublicDatasetSpec(train_mode="finetune", supports_offline_packing=True),
    "openmathinstruct2": PublicDatasetSpec(train_mode="finetune", supports_offline_packing=True),
    "openmathinstruct2-thinking": PublicDatasetSpec(train_mode="finetune", supports_offline_packing=True),
    "gsm8k": PublicDatasetSpec(train_mode="finetune", supports_offline_packing=True),
    "local-jsonl": PublicDatasetSpec(train_mode="finetune", supports_offline_packing=True),
    "local-vlm": PublicDatasetSpec(train_mode="finetune", modality="vlm"),
    "cord-v2": PublicDatasetSpec(train_mode="finetune", modality="vlm", hf_dataset_name="cord_v2"),
    "llava-video-178k": PublicDatasetSpec(
        train_mode="finetune",
        modality="vlm",
        hf_dataset_name="llava_video_178k",
    ),
    "medpix": PublicDatasetSpec(train_mode="finetune", modality="vlm", hf_dataset_name="medpix"),
    "raven": PublicDatasetSpec(train_mode="finetune", modality="vlm", hf_dataset_name="raven"),
    "rdr": PublicDatasetSpec(train_mode="finetune", modality="vlm", hf_dataset_name="rdr"),
}

PUBLIC_DATASET_NAMES = list(PUBLIC_DATASETS)

LLM_FINETUNE_PRESETS: dict[str, Callable] = {
    "squad": default_squad_config,
    "openmathinstruct2": default_openmathinstruct2_config,
    "gsm8k": default_gsm8k_config,
}


def extract_and_remove_override(cli_overrides: list[str], key: str, default: str | None = None) -> str | None:
    """Extract a Hydra-style override (key=value) from *cli_overrides* and remove it.

    Returns the value if found, otherwise *default*.
    """
    prefix = f"{key}="
    for i, override in enumerate(cli_overrides):
        if override.startswith(prefix):
            value = override[len(prefix) :]
            cli_overrides.pop(i)
            return value
    return default


def _resolve_seq_length(config: ConfigContainer, seq_length: int | None) -> int:
    """Resolve sequence length: explicit arg > model config > 4096 fallback."""
    if seq_length is not None:
        return seq_length
    if hasattr(config, "model") and config.model is not None and hasattr(config.model, "seq_length"):
        return config.model.seq_length
    return 4096


def _local_vlm_json_source(path: str, split: str) -> HFDatasetSourceConfig:
    """Build a direct-HF source for one local VLM JSON or JSONL split."""
    return HFDatasetSourceConfig(
        path_or_dataset="json",
        split=split,
        load_kwargs={"data_files": {split: path}},
    )


def apply_dataset_override(
    config: ConfigContainer,
    dataset_type: str,
    enable_offline_packing: bool = False,
    seq_length: int | None = None,
    cli_overrides: list[str] | None = None,
) -> ConfigContainer:
    """Replace the recipe's dataset config based on the requested dataset type.

    Args:
        config: The recipe config to modify.
        dataset_type: One of :data:`DATASET_TYPES`.
        enable_offline_packing: Whether to enable offline packed-sequence preparation.
        seq_length: Explicit sequence length (None = use model's or default 4096).
        cli_overrides: Mutable list of Hydra-style CLI overrides. For ``llm-finetune``,
            ``dataset.hf_dataset.dataset_name`` is extracted to select the preset.

    Returns:
        The modified ConfigContainer.
    """
    resolved_seq_length = _resolve_seq_length(config, seq_length)
    if cli_overrides is None:
        cli_overrides = []

    if dataset_type == "llm-pretrain":
        config.dataset = GPTDatasetConfig(
            seq_length=resolved_seq_length,
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

    elif dataset_type == "llm-pretrain-mock":
        config.dataset = MockGPTDatasetConfig(
            seq_length=resolved_seq_length,
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

    elif dataset_type == "llm-finetune":
        preset_name = extract_and_remove_override(cli_overrides, "dataset.hf_dataset.dataset_name", default="squad")
        if preset_name not in LLM_FINETUNE_PRESETS:
            raise ValueError(
                f"Unknown finetune dataset preset: '{preset_name}'. "
                f"Choose from: {', '.join(sorted(LLM_FINETUNE_PRESETS.keys()))}"
            )
        factory = LLM_FINETUNE_PRESETS[preset_name]
        kwargs: dict = {"enable_offline_packing": enable_offline_packing, "pad_seq_to_mult": 1}
        kwargs["seq_length"] = resolved_seq_length
        config.dataset = factory(**kwargs)

    elif dataset_type == "llm-finetune-preloaded":
        dataset_root = extract_and_remove_override(cli_overrides, "dataset.dataset_root")
        if not dataset_root:
            raise ValueError(
                "llm-finetune-preloaded requires dataset.dataset_root=<path> to select the local JSONL source."
            )
        config.dataset = GPTSFTDatasetConfig(
            seq_length=resolved_seq_length,
            dataset_root=dataset_root,
            preprocessing=PromptCompletionSFTPreprocessingConfig(
                prompt_column="input",
                completion_column="output",
                separator=" ",
            ),
            dataloader_type="batch",
            seed=5678,
        )

    elif dataset_type == "vlm-energon":
        if not isinstance(config.dataset, EnergonDatasetConfig):
            raise ValueError(
                "vlm-energon requires a recipe that defines EnergonDatasetConfig with a model-specific "
                "task_encoder config; a generic runtime encoder cannot be inferred."
            )
        if seq_length is not None:
            config.dataset.seq_length = resolved_seq_length
        logger.info("Recipe already provides EnergonDatasetConfig; keeping its task-encoder configuration.")

    elif dataset_type == "vlm-hf":
        config.dataset = DirectHFSFTDatasetConfig(
            seq_length=resolved_seq_length,
            preprocessing=ChatSFTPreprocessingConfig(),
            hf_processor_path=None,
            source=HFDatasetSourceConfig(dataset_name="cord_v2"),
            num_workers=2,
            dataloader_type="single",
            data_sharding=True,
            pin_memory=True,
            persistent_workers=False,
            enable_in_batch_packing=False,
        )

    else:
        raise ValueError(f"Unknown dataset type: '{dataset_type}'. Choose from: {', '.join(DATASET_TYPES)}")

    if seq_length is not None and hasattr(config, "model") and config.model is not None:
        config.model.seq_length = seq_length

    return config


def apply_public_dataset_override(
    config: ConfigContainer,
    dataset_name: str,
    *,
    enable_offline_packing: bool = False,
    pad_seq_to_mult: int = 1,
    seq_length: int | None = None,
    dataset_root: str | None = None,
    train_data_path: str | None = None,
    validation_data_path: str | None = None,
    test_data_path: str | None = None,
    media_root: str | None = None,
    hf_processor_path: str | None = None,
) -> ConfigContainer:
    """Replace a recipe dataset using a public launcher dataset name.

    Args:
        config: The recipe config to modify.
        dataset_name: One of :data:`PUBLIC_DATASET_NAMES`.
        enable_offline_packing: Whether to prepare offline packed sequences for a text SFT dataset.
        pad_seq_to_mult: Sequence padding multiple for packed SFT datasets.
        seq_length: Explicit sequence length. Uses the model value when unset.
        dataset_root: Directory containing local text SFT JSONL splits.
        train_data_path: Local VLM training JSON or JSONL file.
        validation_data_path: Optional local VLM validation JSON or JSONL file.
        test_data_path: Optional local VLM test JSON or JSONL file.
        media_root: Root containing the required LLaVA-Video assets.
        hf_processor_path: Optional Hugging Face processor model ID or local path.

    Returns:
        The modified ConfigContainer.
    """
    dataset_spec = PUBLIC_DATASETS.get(dataset_name)
    if dataset_spec is None:
        raise ValueError(f"Unknown dataset name: '{dataset_name}'. Choose from: {', '.join(PUBLIC_DATASET_NAMES)}")
    if enable_offline_packing and not dataset_spec.supports_offline_packing:
        raise ValueError("--offline-packing is supported only for text SFT datasets.")
    if dataset_root is not None and dataset_name != "local-jsonl":
        raise ValueError("--dataset-root is used only by local-jsonl.")
    if any(path is not None for path in (train_data_path, validation_data_path, test_data_path)) and dataset_name != (
        "local-vlm"
    ):
        raise ValueError("Local VLM split paths are used only by local-vlm.")
    if media_root is not None and dataset_name != "llava-video-178k":
        raise ValueError("--media-root is used only by llava-video-178k.")
    if hf_processor_path is not None and dataset_spec.modality != "vlm":
        raise ValueError("--hf-processor-path is used only by VLM datasets.")

    resolved_seq_length = _resolve_seq_length(config, seq_length)

    if dataset_name == "mock":
        config.dataset = MockGPTDatasetConfig(
            seq_length=resolved_seq_length,
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
    elif dataset_name == "megatron-indexed":
        config.dataset = GPTDatasetConfig(
            seq_length=resolved_seq_length,
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
    elif dataset_name == "squad":
        config.dataset = default_squad_config(
            seq_length=resolved_seq_length,
            enable_offline_packing=enable_offline_packing,
            pad_seq_to_mult=pad_seq_to_mult,
        )
    elif dataset_name == "openmathinstruct2":
        config.dataset = default_openmathinstruct2_config(
            seq_length=resolved_seq_length,
            enable_offline_packing=enable_offline_packing,
            pad_seq_to_mult=pad_seq_to_mult,
        )
    elif dataset_name == "openmathinstruct2-thinking":
        config.dataset = default_openmathinstruct2_thinking_packed_config(
            seq_length=resolved_seq_length,
            enable_offline_packing=enable_offline_packing,
            pad_seq_to_mult=pad_seq_to_mult,
        )
    elif dataset_name == "gsm8k":
        config.dataset = default_gsm8k_config(
            seq_length=resolved_seq_length,
            enable_offline_packing=enable_offline_packing,
            pad_seq_to_mult=pad_seq_to_mult,
        )
    elif dataset_name == "local-jsonl":
        if not dataset_root:
            raise ValueError("local-jsonl requires --dataset-root=<path> to select the local JSONL source.")
        offline_packing_specs = None
        if enable_offline_packing:
            offline_packing_specs = PackedSequenceSpecs(
                packed_sequence_size=resolved_seq_length,
                pad_seq_to_mult=pad_seq_to_mult,
            )
        config.dataset = GPTSFTDatasetConfig(
            seq_length=resolved_seq_length,
            dataset_root=dataset_root,
            preprocessing=PromptCompletionSFTPreprocessingConfig(
                prompt_column="input",
                completion_column="output",
                separator=" ",
            ),
            enable_offline_packing=enable_offline_packing,
            offline_packing_specs=offline_packing_specs,
            dataloader_type="batch",
            seed=5678,
        )
    elif dataset_name == "local-vlm":
        existing_dataset = config.dataset
        if not isinstance(existing_dataset, DirectHFSFTDatasetConfig):
            raise ValueError("local-vlm requires a VLM recipe using DirectHFSFTDatasetConfig.")
        processor_path = hf_processor_path or existing_dataset.hf_processor_path
        if not train_data_path:
            raise ValueError("local-vlm requires --train-data-path=<json-or-jsonl-path>.")
        if not processor_path:
            raise ValueError("local-vlm requires --hf-processor-path or a processor configured by the VLM recipe.")
        config.dataset = replace(
            existing_dataset,
            seq_length=resolved_seq_length,
            hf_processor_path=processor_path,
            source=_local_vlm_json_source(train_data_path, "train"),
            validation_source=(
                _local_vlm_json_source(validation_data_path, "validation") if validation_data_path else None
            ),
            test_source=_local_vlm_json_source(test_data_path, "test") if test_data_path else None,
            do_validation=validation_data_path is not None,
            do_test=test_data_path is not None,
        )
    else:
        existing_dataset = config.dataset
        if not isinstance(existing_dataset, DirectHFSFTDatasetConfig):
            raise ValueError(f"{dataset_name} requires a VLM recipe using DirectHFSFTDatasetConfig.")
        if dataset_spec.hf_dataset_name is None:
            raise ValueError(f"Dataset '{dataset_name}' does not map to a Hugging Face VLM preset.")
        processor_path = hf_processor_path or existing_dataset.hf_processor_path
        if not processor_path:
            raise ValueError(
                f"{dataset_name} requires --hf-processor-path or a processor configured by the VLM recipe."
            )

        adapter_kwargs = None
        if dataset_name == "llava-video-178k":
            if not media_root:
                raise ValueError("llava-video-178k requires --media-root=<video-root-path>.")
            adapter_kwargs = {"video_root_path": media_root}

        source = HFDatasetSourceConfig(
            dataset_name=dataset_spec.hf_dataset_name,
            adapter_kwargs=adapter_kwargs,
        )
        supports_native_validation = hf_dataset_supports_split(source, "validation")
        validation_source = None
        if existing_dataset.do_validation and not supports_native_validation:
            validation_source = source.with_split("train[95%:]")
            source = source.with_split("train[:95%]")
        supports_test = hf_dataset_supports_split(source, "test")
        config.dataset = replace(
            existing_dataset,
            source=source,
            validation_source=validation_source,
            test_source=None,
            hf_processor_path=processor_path,
            do_test=existing_dataset.do_test and supports_test,
        )

    if seq_length is not None and hasattr(config, "model") and config.model is not None:
        config.model.seq_length = seq_length
    return config


def infer_mode_from_dataset(dataset_type: str) -> str:
    """Infer training mode from the dataset type prefix."""
    if dataset_type.startswith("llm-pretrain"):
        return "pretrain"
    return "finetune"
