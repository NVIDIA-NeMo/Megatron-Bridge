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
def create_mock_dataset_config(seq_length, num_workers=8, pin_memory=False, persistent_workers=False):
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
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def create_rp2_dataset_config(
    dataset_paths, seq_length, index_mapping_dir=None, num_workers=1, pin_memory=False, persistent_workers=True
):
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
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def create_c4_dataset_config(
    seq_length,
    c4_root,
    train_shards=(6,),
    num_workers=1,
    pin_memory=False,
    persistent_workers=True,
    index_mapping_dir=None,
):
    """Create C4 dataset configuration for Megatron-Bridge.

    Uses Megatron-mmap C4 files at:
      <c4_root>/c4-train.en_<shard>_text_document.{bin,idx}
      <c4_root>/c4-validation-91205-samples.en_text_document.{bin,idx}

    Produces a blend_per_split with (train, val, test) tuples where val and
    test point to the same validation file (matches NVIDIA's MLPerf DSV3
    reference layout).
    """
    from megatron.bridge.training.config import GPTDatasetConfig

    val_prefix = f"{c4_root}/c4-validation-91205-samples.en_text_document"
    train_prefixes = [f"{c4_root}/c4-train.en_{i}_text_document" for i in train_shards]
    train_weights = [50.0] * len(train_prefixes)

    # GPTDatasetConfig.blend_per_split format: list of 3 (prefixes, weights) tuples
    # for (train, val, test). Mirrors NVIDIA's MLPerf DSV3 reference setup with
    # val and test pointing to the same validation file.
    blend_per_split = [
        (train_prefixes, train_weights),
        ([val_prefix], None),
        ([val_prefix], None),
    ]

    return GPTDatasetConfig(
        random_seed=1234,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        seq_length=seq_length,
        num_dataset_builder_threads=1,
        blend=None,
        blend_per_split=blend_per_split,
        split=None,  # blend_per_split takes precedence; split must be None.
        path_to_cache=index_mapping_dir,
        data_sharding=True,
        dataloader_type="single",
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def create_squad_dataset_config(
    dataset_root, seq_length, packed=False, pad_seq_to_mult=1, num_workers=2, pin_memory=True, persistent_workers=False
):
    """Create SQuAD dataset configuration for Megatron-Bridge using HF text SFT data."""
    from megatron.bridge.data.builders import (
        GPTSFTDatasetConfig,
        HFDatasetSourceConfig,
        PromptCompletionSFTPreprocessingConfig,
    )
    from megatron.bridge.data.packing import PackedSequenceSpecs

    dataset_kwargs = {}
    offline_packing_specs = None
    if packed:
        dataset_kwargs["pad_to_max_length"] = True
        offline_packing_specs = PackedSequenceSpecs(packed_sequence_size=seq_length, pad_seq_to_mult=pad_seq_to_mult)

    return GPTSFTDatasetConfig(
        seq_length=seq_length,
        hf_dataset=HFDatasetSourceConfig(dataset_name="squad"),
        hf_output_root=dataset_root,
        hf_validation_proportion=0.1,
        seed=5678,
        do_validation=True,
        do_test=False,
        preprocessing=PromptCompletionSFTPreprocessingConfig(
            prompt_column="input",
            completion_column="output",
            separator=" ",
        ),
        dataset_kwargs=dataset_kwargs,
        enable_offline_packing=packed,
        offline_packing_specs=offline_packing_specs,
        dataloader_type="single",
        num_workers=num_workers,
        data_sharding=True,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
