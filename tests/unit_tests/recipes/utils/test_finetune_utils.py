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

"""Tests for finetune_utils module: default_openmathinstruct2_config and default_gsm8k_config."""

import pytest

from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.data.hf_processors.gsm8k import process_gsm8k_example
from megatron.bridge.data.hf_processors.openmathinstruct2 import process_openmathinstruct2_example
from megatron.bridge.recipes.utils.finetune_utils import (
    default_gsm8k_config,
    default_openmathinstruct2_config,
)


@pytest.mark.unit
class TestDefaultOpenmathinstruct2Config:
    """Test cases for default_openmathinstruct2_config."""

    def test_returns_hf_dataset_config(self):
        cfg = default_openmathinstruct2_config()
        assert isinstance(cfg, HFDatasetConfig)

    def test_default_dataset_name(self):
        cfg = default_openmathinstruct2_config()
        assert cfg.dataset_name == "nvidia/OpenMathInstruct-2"

    def test_default_split(self):
        cfg = default_openmathinstruct2_config()
        assert cfg.split == "train_1M"

    def test_default_seq_length(self):
        cfg = default_openmathinstruct2_config()
        assert cfg.seq_length == 4096

    def test_custom_seq_length(self):
        cfg = default_openmathinstruct2_config(seq_length=8192)
        assert cfg.seq_length == 8192

    def test_process_fn_is_openmathinstruct2(self):
        cfg = default_openmathinstruct2_config()
        assert cfg.process_example_fn is process_openmathinstruct2_example

    def test_default_seed(self):
        cfg = default_openmathinstruct2_config()
        assert cfg.seed == 5678

    def test_dataloader_type_batch(self):
        cfg = default_openmathinstruct2_config()
        assert cfg.dataloader_type == "batch"

    def test_validation_enabled(self):
        cfg = default_openmathinstruct2_config()
        assert cfg.do_validation is True
        assert cfg.do_test is False
        assert cfg.val_proportion == 0.05

    def test_worker_settings(self):
        cfg = default_openmathinstruct2_config()
        assert cfg.num_workers == 2
        assert cfg.memmap_workers == 1

    def test_data_sharding_and_pin_memory(self):
        cfg = default_openmathinstruct2_config()
        assert cfg.data_sharding is True
        assert cfg.pin_memory is True
        assert cfg.persistent_workers is False

    def test_rewrite_disabled(self):
        cfg = default_openmathinstruct2_config()
        assert cfg.rewrite is False

    def test_no_packed_sequence_by_default(self):
        cfg = default_openmathinstruct2_config()
        assert cfg.packed_sequence_specs is None

    def test_packed_sequence_enabled(self):
        cfg = default_openmathinstruct2_config(packed_sequence=True)
        assert isinstance(cfg.packed_sequence_specs, PackedSequenceSpecs)
        assert cfg.packed_sequence_specs.packed_sequence_size == 4096
        assert cfg.packed_sequence_specs.pad_seq_to_mult == 1

    def test_packed_sequence_with_custom_seq_length(self):
        cfg = default_openmathinstruct2_config(seq_length=8192, packed_sequence=True)
        assert cfg.packed_sequence_specs.packed_sequence_size == 8192

    def test_packed_sequence_with_pad_seq_to_mult(self):
        cfg = default_openmathinstruct2_config(packed_sequence=True, pad_seq_to_mult=4)
        assert cfg.packed_sequence_specs.pad_seq_to_mult == 4

    def test_pad_seq_to_mult_ignored_without_packing(self):
        cfg = default_openmathinstruct2_config(packed_sequence=False, pad_seq_to_mult=4)
        assert cfg.packed_sequence_specs is None


@pytest.mark.unit
class TestDefaultGsm8kConfig:
    """Test cases for default_gsm8k_config."""

    def test_returns_hf_dataset_config(self):
        cfg = default_gsm8k_config()
        assert isinstance(cfg, HFDatasetConfig)

    def test_default_dataset_name(self):
        cfg = default_gsm8k_config()
        assert cfg.dataset_name == "openai/gsm8k"

    def test_default_dataset_subset(self):
        cfg = default_gsm8k_config()
        assert cfg.dataset_subset == "main"

    def test_no_split_restriction(self):
        cfg = default_gsm8k_config()
        assert cfg.split is None

    def test_default_seq_length(self):
        cfg = default_gsm8k_config()
        assert cfg.seq_length == 2048

    def test_custom_seq_length(self):
        cfg = default_gsm8k_config(seq_length=4096)
        assert cfg.seq_length == 4096

    def test_process_fn_is_gsm8k(self):
        cfg = default_gsm8k_config()
        assert cfg.process_example_fn is process_gsm8k_example

    def test_default_seed(self):
        cfg = default_gsm8k_config()
        assert cfg.seed == 5678

    def test_dataloader_type_batch(self):
        cfg = default_gsm8k_config()
        assert cfg.dataloader_type == "batch"

    def test_uses_published_test_split(self):
        cfg = default_gsm8k_config()
        assert cfg.do_validation is False
        assert cfg.do_test is True

    def test_worker_settings(self):
        cfg = default_gsm8k_config()
        assert cfg.num_workers == 2
        assert cfg.memmap_workers == 1

    def test_data_sharding_and_pin_memory(self):
        cfg = default_gsm8k_config()
        assert cfg.data_sharding is True
        assert cfg.pin_memory is True
        assert cfg.persistent_workers is False

    def test_rewrite_disabled(self):
        cfg = default_gsm8k_config()
        assert cfg.rewrite is False

    def test_no_packed_sequence_by_default(self):
        cfg = default_gsm8k_config()
        assert cfg.packed_sequence_specs is None

    def test_packed_sequence_enabled(self):
        cfg = default_gsm8k_config(packed_sequence=True)
        assert isinstance(cfg.packed_sequence_specs, PackedSequenceSpecs)
        assert cfg.packed_sequence_specs.packed_sequence_size == 2048
        assert cfg.packed_sequence_specs.pad_seq_to_mult == 1

    def test_packed_sequence_with_custom_seq_length(self):
        cfg = default_gsm8k_config(seq_length=4096, packed_sequence=True)
        assert cfg.packed_sequence_specs.packed_sequence_size == 4096

    def test_packed_sequence_with_pad_seq_to_mult(self):
        cfg = default_gsm8k_config(packed_sequence=True, pad_seq_to_mult=4)
        assert cfg.packed_sequence_specs.pad_seq_to_mult == 4

    def test_pad_seq_to_mult_ignored_without_packing(self):
        cfg = default_gsm8k_config(packed_sequence=False, pad_seq_to_mult=4)
        assert cfg.packed_sequence_specs is None


@pytest.mark.unit
class TestConfigDifferences:
    """Verify key differences between the two dataset configs."""

    def test_different_default_seq_lengths(self):
        omi2 = default_openmathinstruct2_config()
        gsm8k = default_gsm8k_config()
        assert omi2.seq_length == 4096
        assert gsm8k.seq_length == 2048

    def test_different_validation_strategies(self):
        omi2 = default_openmathinstruct2_config()
        gsm8k = default_gsm8k_config()
        assert omi2.do_validation is True
        assert omi2.val_proportion == 0.05
        assert gsm8k.do_validation is False
        assert gsm8k.do_test is True

    def test_different_dataset_names(self):
        omi2 = default_openmathinstruct2_config()
        gsm8k = default_gsm8k_config()
        assert omi2.dataset_name == "nvidia/OpenMathInstruct-2"
        assert gsm8k.dataset_name == "openai/gsm8k"

    def test_different_process_fns(self):
        omi2 = default_openmathinstruct2_config()
        gsm8k = default_gsm8k_config()
        assert omi2.process_example_fn is not gsm8k.process_example_fn

    def test_gsm8k_has_subset_omi2_has_split(self):
        omi2 = default_openmathinstruct2_config()
        gsm8k = default_gsm8k_config()
        assert gsm8k.dataset_subset == "main"
        assert omi2.split == "train_1M"
