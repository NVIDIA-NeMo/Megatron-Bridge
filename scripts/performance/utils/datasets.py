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
def create_mock_dataset_config(seq_length):
    """Create mock dataset configuration for Megatron-Bridge."""
    from megatron.bridge.training.config import MockGPTDatasetConfig

    # Create mock dataset using MockGPTDatasetConfig which enforces blend=None, blend_per_split=None
    return MockGPTDatasetConfig(
        seq_length=seq_length,
        random_seed=1234,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        num_dataset_builder_threads=1,
        split="99990,8,2",  # Standard train/val/test split
        # Dataloader config parameters
        data_sharding=True,
        dataloader_type="single",
        num_workers=8,
    )


def create_rp2_dataset_config(dataset_paths, seq_length, index_mapping_dir=None):
    """Create RedPajama2 dataset configuration for Megatron-Bridge."""
    from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths
    from megatron.bridge.training.config import GPTDatasetConfig

    # Get blend configuration for rp2 data paths
    blend, blend_per_split, split = get_blend_fields_from_data_paths(data_paths=dataset_paths, mock=False)

    return GPTDatasetConfig(
        random_seed=1234,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        seq_length=seq_length,
        num_dataset_builder_threads=1,
        blend=blend,
        blend_per_split=blend_per_split,
        split=split or "99990,8,2",
        path_to_cache=index_mapping_dir,
        # Dataloader config parameters
        data_sharding=True,
        dataloader_type="single",
        num_workers=1,
        persistent_workers=True,
    )


def create_squad_dataset_config(dataset_root, seq_length, packed=False, pad_seq_to_mult=1):
    """Create SQuAD dataset configuration for Megatron-Bridge using HF dataset."""
    from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig
    from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
    from megatron.bridge.data.hf_processors import process_squad_example

    # Create packed sequence specs if needed
    packed_sequence_specs = None
    if packed:
        packed_sequence_specs = PackedSequenceSpecs(packed_sequence_size=seq_length, pad_seq_to_mult=pad_seq_to_mult)

    return HFDatasetConfig(
        dataset_name="squad",  # Hugging Face dataset name
        process_example_fn=process_squad_example,  # Processing function
        dataset_root=dataset_root,  # Local cache/processed files location
        seq_length=seq_length,
        seed=1234,
        memmap_workers=1,
        # Dataloader config parameters
        dataloader_type="single",
        num_workers=2,
        data_sharding=True,
        pin_memory=True,
        persistent_workers=False,
        packed_sequence_specs=packed_sequence_specs,
        rewrite=False,  # Rewrite existing processed files
        delete_raw=False,  # Keep raw HF dataset cache
    )
