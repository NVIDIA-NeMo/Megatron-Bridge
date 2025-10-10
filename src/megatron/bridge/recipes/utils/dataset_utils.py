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

from typing import List, Optional, Tuple
from megatron.bridge.training.config import GPTDatasetConfig

from megatron.bridge.data.loaders import get_blend_and_blend_per_split


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


def get_pretrain_dataset(
    seq_length: int, 
    blend: _BLEND_TYPE, 
    blend_per_split: _BLEND_PER_SPLIT_TYPE, 
    split: _SPLIT_TYPE
):
    return GPTDatasetConfig(
        random_seed=1234,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        sequence_length=seq_length,
        num_dataset_builder_threads=1,
        blend=blend,
        blend_per_split=blend_per_split,
        split=split,
        data_sharding=True,
        dataloader_type="single",
        skip_getting_attention_mask_from_dataset=True,
    )