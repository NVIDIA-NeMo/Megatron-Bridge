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

from typing import Any, Callable

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.gpt_dataset import GPTDataset, MockGPTDataset
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider
from megatron.bridge.data.builders.direct_hf_sft import (
    DirectHFSFTDatasetConfig,
    direct_hf_sft_train_valid_test_datasets_provider,
)
from megatron.bridge.data.builders.energon import EnergonDatasetConfig, energon_train_valid_test_datasets_provider
from megatron.bridge.data.builders.gpt_sft import (
    GPTSFTDatasetConfig,
    gpt_sft_train_valid_test_datasets_provider,
)
from megatron.bridge.data.builders.mock_vlm_sft import (
    MockVLMSFTDatasetConfig,
    mock_vlm_sft_train_valid_test_datasets_provider,
)
from megatron.bridge.data.datasets.fim_dataset import GPTFIMDataset
from megatron.bridge.training.config import GPTDatasetConfig, GPTFIMDatasetConfig, MockGPTDatasetConfig
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer
from megatron.bridge.utils.common_utils import print_rank_0


def is_dataset_built_on_rank(pg_collection: ProcessGroupCollection) -> bool:
    """Determines whether the dataset should be built on the current rank.

    Datasets are typically built only on the first and last pipeline stages
    and the first tensor parallel rank to save memory and avoid redundancy.

    Returns:
        True if the dataset should be built on the current rank, False otherwise.
    """
    return (is_pp_first_stage(pg_collection.pp) or is_pp_last_stage(pg_collection.pp)) and (
        pg_collection.tp.rank() == 0
    )


def pretrain_train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int], dataset_config: BlendedMegatronDatasetConfig
) -> tuple[GPTDataset, GPTDataset, GPTDataset]:
    """Build pretraining train, validation, and test datasets.

    Uses BlendedMegatronDatasetBuilder to create GPTDataset or MockGPTDataset instances.

    Args:
        train_val_test_num_samples: A list containing the number of samples for
                                    train, validation, and test datasets.
        dataset_config: Configuration object for the blended Megatron dataset.

    Returns:
        A tuple containing the train, validation, and test datasets.
    """

    if dataset_config.mock:
        dataset_type = MockGPTDataset
    elif hasattr(dataset_config, "fim_data"):
        dataset_type = GPTFIMDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    # Build the dataset on all ranks for TP-replicated loading
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, lambda: True, dataset_config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


_REGISTRY: dict[type[Any], Callable[..., Any]] = {
    GPTDatasetConfig: pretrain_train_valid_test_datasets_provider,
    GPTFIMDatasetConfig: pretrain_train_valid_test_datasets_provider,
    MockGPTDatasetConfig: pretrain_train_valid_test_datasets_provider,
    GPTSFTDatasetConfig: gpt_sft_train_valid_test_datasets_provider,
    DirectHFSFTDatasetConfig: direct_hf_sft_train_valid_test_datasets_provider,
    EnergonDatasetConfig: energon_train_valid_test_datasets_provider,
    MockVLMSFTDatasetConfig: mock_vlm_sft_train_valid_test_datasets_provider,
}


def get_dataset_provider(
    dataset_config: (
        BlendedMegatronDatasetConfig
        | GPTSFTDatasetConfig
        | DirectHFSFTDatasetConfig
        | EnergonDatasetConfig
        | MockVLMSFTDatasetConfig
        | DatasetProvider
    ),
) -> Callable[..., Any]:
    """Get the appropriate dataset provider function based on the config type.

    Supports both registry-based providers and protocol-based providers.

    Args:
        dataset_config: The dataset configuration object.

    Returns:
        The callable dataset provider function corresponding to the config type.
    """
    for config_type, provider in _REGISTRY.items():
        if isinstance(dataset_config, config_type):
            return provider

    # Check if config implements the DatasetProvider protocol
    if isinstance(dataset_config, DatasetProvider):

        def protocol_adapter(
            train_val_test_num_samples: list[int],
            config: DatasetProvider,
            tokenizer: MegatronTokenizer | None = None,
            pg_collection: ProcessGroupCollection | None = None,
        ) -> tuple[Any | None, Any | None, Any | None]:
            """Adapter function that bridges the protocol interface with the legacy interface."""
            context = DatasetBuildContext(
                train_samples=train_val_test_num_samples[0],
                valid_samples=train_val_test_num_samples[1],
                test_samples=train_val_test_num_samples[2],
                tokenizer=tokenizer,
                pg_collection=pg_collection,
            )
            return config.build_datasets(context)

        return protocol_adapter

    raise KeyError(type(dataset_config))
