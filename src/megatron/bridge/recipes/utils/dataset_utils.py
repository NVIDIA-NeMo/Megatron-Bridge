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
from typing import Callable, List, Optional, Tuple

from megatron.bridge.data.energon.energon_provider import EnergonProvider
from megatron.bridge.data.loaders import get_blend_and_blend_per_split
from megatron.bridge.data.vlm_datasets.hf_provider import HFDatasetConversationProvider
from megatron.bridge.data.vlm_datasets.preloaded_provider import PreloadedVLMConversationProvider
from megatron.bridge.recipes.utils.finetune_utils import (
    default_gsm8k_config,
    default_openmathinstruct2_config,
    default_squad_config,
)
from megatron.bridge.training.config import (
    ConfigContainer,
    FinetuningDatasetConfig,
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
    "vlm-preloaded",
]

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


def apply_dataset_override(
    config: ConfigContainer,
    dataset_type: str,
    packed_sequence: bool = False,
    seq_length: int | None = None,
    cli_overrides: list[str] | None = None,
) -> ConfigContainer:
    """Replace the recipe's dataset config based on the requested dataset type.

    Args:
        config: The recipe config to modify.
        dataset_type: One of :data:`DATASET_TYPES`.
        packed_sequence: Whether to enable packed sequences.
        seq_length: Explicit sequence length (None = use model's or default 4096).
        cli_overrides: Mutable list of Hydra-style CLI overrides. For ``llm-finetune``,
            ``dataset.dataset_name`` is extracted and consumed here to select the preset.

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
        preset_name = extract_and_remove_override(cli_overrides, "dataset.dataset_name", default="squad")
        if preset_name not in LLM_FINETUNE_PRESETS:
            raise ValueError(
                f"Unknown finetune dataset preset: '{preset_name}'. "
                f"Choose from: {', '.join(sorted(LLM_FINETUNE_PRESETS.keys()))}"
            )
        factory = LLM_FINETUNE_PRESETS[preset_name]
        kwargs: dict = {"packed_sequence": packed_sequence, "pad_seq_to_mult": 1}
        if seq_length is not None:
            kwargs["seq_length"] = seq_length
        config.dataset = factory(**kwargs)

    elif dataset_type == "llm-finetune-preloaded":
        config.dataset = FinetuningDatasetConfig(
            seq_length=resolved_seq_length,
            dataset_root=None,
            dataloader_type="batch",
            seed=5678,
        )

    elif dataset_type == "vlm-energon":
        if isinstance(config.dataset, EnergonProvider):
            logger.info("Recipe already provides EnergonProvider; keeping it (preserves task_encoder).")
        else:
            logger.warning(
                "Creating bare EnergonProvider. task_encoder and image_processor are unset; "
                "use a recipe that provides them or set via code."
            )
            config.dataset = EnergonProvider(
                path="",
                seq_length=resolved_seq_length,
                micro_batch_size=config.train.micro_batch_size,
                global_batch_size=config.train.global_batch_size,
                num_workers=2,
            )

    elif dataset_type == "vlm-hf":
        config.dataset = HFDatasetConversationProvider(
            seq_length=resolved_seq_length,
            hf_processor_path=None,
            maker_name="make_cord_v2_dataset",
            num_workers=2,
            dataloader_type="single",
            data_sharding=True,
            pin_memory=True,
            persistent_workers=False,
            pack_sequences_in_batch=False,
        )

    elif dataset_type == "vlm-preloaded":
        config.dataset = PreloadedVLMConversationProvider(
            seq_length=resolved_seq_length,
            hf_processor_path=None,
            train_data_path=None,
            valid_data_path=None,
            test_data_path=None,
            dataloader_type="single",
            num_workers=2,
        )

    else:
        raise ValueError(f"Unknown dataset type: '{dataset_type}'. Choose from: {', '.join(DATASET_TYPES)}")

    if seq_length is not None and hasattr(config, "model") and config.model is not None:
        config.model.seq_length = seq_length

    return config


def infer_mode_from_dataset(dataset_type: str) -> str:
    """Infer training mode from the dataset type prefix."""
    if dataset_type.startswith("llm-pretrain"):
        return "pretrain"
    return "finetune"
