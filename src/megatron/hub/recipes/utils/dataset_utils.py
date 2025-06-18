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

from megatron.hub.training.config import GPTDatasetConfig


def mock_dataset_config(
    random_seed: int = 1234,
    reset_attention_mask: bool = False,
    reset_position_ids: bool = False,
    eod_mask_loss: bool = False,
    sequence_length: int = 2048,
    num_workers: int = 1,
    num_dataset_builder_threads: int = 1,
):
    return GPTDatasetConfig(
        random_seed=random_seed,
        reset_attention_mask=reset_attention_mask,
        reset_position_ids=reset_position_ids,
        eod_mask_loss=eod_mask_loss,
        sequence_length=sequence_length,
        num_workers=num_workers,
        num_dataset_builder_threads=num_dataset_builder_threads,
    )
