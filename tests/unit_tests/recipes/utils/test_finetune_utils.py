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

"""Tests for finetune_utils Hugging Face semantic dataset defaults."""

import pytest

from megatron.bridge.data.builders import GPTSFTDatasetConfig, PromptCompletionSFTPreprocessingConfig
from megatron.bridge.recipes.utils.finetune_utils import (
    default_gsm8k_config,
    default_openmathinstruct2_config,
    default_openmathinstruct2_thinking_packed_config,
    default_squad_config,
)


@pytest.mark.unit
class TestDefaultOpenmathinstruct2Config:
    """Test cases for default_openmathinstruct2_config."""

    def test_returns_gpt_sft_config(self):
        cfg = default_openmathinstruct2_config()
        assert isinstance(cfg, GPTSFTDatasetConfig)

    def test_default_dataset_name(self):
        cfg = default_openmathinstruct2_config()
        assert cfg.hf_dataset.dataset_name == "openmathinstruct2"

    def test_default_split(self):
        cfg = default_openmathinstruct2_config()
        assert cfg.hf_dataset.split is None

    def test_default_seq_length(self):
        cfg = default_openmathinstruct2_config()
        assert cfg.seq_length == 4096

    def test_custom_seq_length(self):
        cfg = default_openmathinstruct2_config(seq_length=8192)
        assert cfg.seq_length == 8192

    def test_preset_is_openmathinstruct2(self):
        cfg = default_openmathinstruct2_config()
        assert cfg.hf_dataset.dataset_name == "openmathinstruct2"

    def test_dataloader_type_batch(self):
        cfg = default_openmathinstruct2_config()
        assert cfg.dataloader_type == "batch"

    def test_validation_enabled(self):
        cfg = default_openmathinstruct2_config()
        assert cfg.hf_validation_dataset is None
        assert cfg.hf_validation_proportion == 0.05
        assert cfg.do_validation is True
        assert cfg.do_test is False

    def test_worker_settings(self):
        cfg = default_openmathinstruct2_config()
        assert cfg.num_workers == 2

    def test_data_sharding_and_pin_memory(self):
        cfg = default_openmathinstruct2_config()
        assert cfg.data_sharding is True
        assert cfg.pin_memory is True
        assert cfg.persistent_workers is False

    def test_packing_disabled_by_default(self):
        cfg = default_openmathinstruct2_config()
        assert cfg.enable_offline_packing is False
        assert cfg.offline_packing_specs is None

    def test_packed_sequence_request_enables_offline_packing(self):
        cfg = default_openmathinstruct2_config(packed_sequence=True)
        assert cfg.enable_offline_packing is True
        assert cfg.offline_packing_specs is not None
        assert cfg.offline_packing_specs.packed_sequence_size == 4096

    def test_pad_seq_to_mult_applies_to_packing(self):
        cfg = default_openmathinstruct2_config(packed_sequence=True, pad_seq_to_mult=4)
        assert cfg.offline_packing_specs.pad_seq_to_mult == 4


@pytest.mark.unit
class TestDefaultGsm8kConfig:
    """Test cases for default_gsm8k_config."""

    def test_returns_gpt_sft_config(self):
        cfg = default_gsm8k_config()
        assert isinstance(cfg, GPTSFTDatasetConfig)

    def test_default_dataset_name(self):
        cfg = default_gsm8k_config()
        assert cfg.hf_dataset.dataset_name == "gsm8k"

    def test_default_dataset_subset(self):
        cfg = default_gsm8k_config()
        assert cfg.hf_dataset.subset is None

    def test_no_split_restriction(self):
        cfg = default_gsm8k_config()
        assert cfg.hf_dataset.split is None

    def test_default_seq_length(self):
        cfg = default_gsm8k_config()
        assert cfg.seq_length == 2048

    def test_custom_seq_length(self):
        cfg = default_gsm8k_config(seq_length=4096)
        assert cfg.seq_length == 4096

    def test_preset_is_gsm8k(self):
        cfg = default_gsm8k_config()
        assert cfg.hf_dataset.dataset_name == "gsm8k"

    def test_dataloader_type_batch(self):
        cfg = default_gsm8k_config()
        assert cfg.dataloader_type == "batch"

    def test_uses_published_test_split(self):
        cfg = default_gsm8k_config()
        assert cfg.hf_validation_dataset is None
        assert cfg.hf_test_dataset.split == "test"
        assert cfg.do_validation is False
        assert cfg.do_test is True

    def test_worker_settings(self):
        cfg = default_gsm8k_config()
        assert cfg.num_workers == 2

    def test_data_sharding_and_pin_memory(self):
        cfg = default_gsm8k_config()
        assert cfg.data_sharding is True
        assert cfg.pin_memory is True
        assert cfg.persistent_workers is False

    def test_runtime_packing_disabled(self):
        cfg = default_gsm8k_config()
        assert cfg.enable_offline_packing is False
        assert cfg.offline_packing_specs is None

    def test_packed_sequence_request_enables_offline_packing(self):
        cfg = default_gsm8k_config(packed_sequence=True)
        assert cfg.enable_offline_packing is True
        assert cfg.offline_packing_specs is not None
        assert cfg.offline_packing_specs.packed_sequence_size == 2048

    def test_pad_seq_to_mult_applies_to_packing(self):
        cfg = default_gsm8k_config(packed_sequence=True, pad_seq_to_mult=4)
        assert cfg.offline_packing_specs.pad_seq_to_mult == 4


@pytest.mark.unit
class TestDefaultSquadConfig:
    """Test cases for default_squad_config."""

    def test_returns_gpt_sft_config(self):
        cfg = default_squad_config(seq_length=512)
        assert isinstance(cfg, GPTSFTDatasetConfig)

    def test_default_preset_config(self):
        cfg = default_squad_config(seq_length=512)
        assert cfg.hf_dataset.dataset_name == "squad"
        assert cfg.hf_dataset.split is None
        assert cfg.hf_validation_dataset is None
        assert cfg.hf_validation_proportion == 0.1
        assert cfg.do_validation is True
        assert cfg.do_test is False
        assert isinstance(cfg.preprocessing, PromptCompletionSFTPreprocessingConfig)
        assert cfg.preprocessing.separator == " "

    def test_packed_sequence_request_enables_offline_packing(self):
        cfg = default_squad_config(seq_length=512, packed_sequence=True)
        assert cfg.enable_offline_packing is True
        assert cfg.offline_packing_specs is not None
        assert cfg.offline_packing_specs.packed_sequence_size == 512
        assert cfg.dataset_kwargs["pad_to_max_length"] is True


@pytest.mark.unit
class TestConfigDifferences:
    """Verify key differences between the two dataset configs."""

    def test_semantic_presets_use_prompt_completion_without_chat_templates(self):
        configs = (
            default_squad_config(seq_length=512),
            default_openmathinstruct2_config(),
            default_gsm8k_config(),
        )
        for cfg in configs:
            assert isinstance(cfg.preprocessing, PromptCompletionSFTPreprocessingConfig)
            assert cfg.preprocessing.separator == " "
            assert cfg.preprocessing.loss_mode == "completion"

    def test_different_default_seq_lengths(self):
        omi2 = default_openmathinstruct2_config()
        gsm8k = default_gsm8k_config()
        assert omi2.seq_length == 4096
        assert gsm8k.seq_length == 2048

    def test_different_validation_strategies(self):
        omi2 = default_openmathinstruct2_config()
        gsm8k = default_gsm8k_config()
        assert omi2.hf_validation_dataset is None
        assert omi2.hf_validation_proportion == 0.05
        assert omi2.do_validation is True
        assert omi2.do_test is False
        assert gsm8k.hf_validation_dataset is None
        assert gsm8k.hf_test_dataset.split == "test"

    def test_different_dataset_names(self):
        omi2 = default_openmathinstruct2_config()
        gsm8k = default_gsm8k_config()
        assert omi2.hf_dataset.dataset_name == "openmathinstruct2"
        assert gsm8k.hf_dataset.dataset_name == "gsm8k"

    def test_different_presets(self):
        omi2 = default_openmathinstruct2_config()
        gsm8k = default_gsm8k_config()
        assert omi2.hf_dataset.dataset_name != gsm8k.hf_dataset.dataset_name

    def test_gsm8k_has_subset_omi2_has_split(self):
        omi2 = default_openmathinstruct2_config()
        gsm8k = default_gsm8k_config()
        assert gsm8k.hf_dataset.subset is None
        assert omi2.hf_dataset.split is None


@pytest.mark.unit
class TestDefaultOpenmathinstruct2ThinkingConfig:
    """Test cases for default_openmathinstruct2_thinking_packed_config."""

    def test_uses_thinking_preset(self):
        cfg = default_openmathinstruct2_thinking_packed_config(seq_length=4096, packed_sequence=True)
        assert isinstance(cfg, GPTSFTDatasetConfig)
        assert cfg.hf_dataset.dataset_name == "openmathinstruct2_thinking"
        assert cfg.hf_dataset.split is None
        assert cfg.hf_validation_proportion == 0.05
        assert cfg.enable_offline_packing is True
        assert cfg.offline_packing_specs is not None
