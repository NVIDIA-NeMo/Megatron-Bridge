# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import megatron.bridge.recipes.qwen_omni.h100.qwen3_omni as _qwen3_omni_module
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


class _FakeProvider:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.pipeline_dtype = None
        self.virtual_pipeline_model_parallel_size = None
        self.context_parallel_size = 1
        self.expert_model_parallel_size = 1
        self.expert_tensor_parallel_size = 1
        self.sequence_parallel = False
        self.seq_length = 64
        self.freeze_language_model = False
        self.freeze_vision_model = False
        self.freeze_audio_model = False
        self.vit_gradient_checkpointing = False
        self.multimodal_attn_impl = "auto"
        self.transformer_impl = "transformer_engine"
        self.cuda_graph_impl = "none"
        self.attention_backend = "auto"

    def finalize(self):
        return None


class _FakeAutoBridge:
    @staticmethod
    def from_hf_pretrained(_hf_path: str):
        return _FakeAutoBridge()

    def to_megatron_provider(self, load_weights: bool = False):
        assert load_weights is False
        return _FakeProvider()


def test_qwen3_omni_sft_recipe_builds_config(monkeypatch):
    patch_recipe_module_global(monkeypatch, _qwen3_omni_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_omni_module.qwen3_omni_30b_a3b_sft_8gpu_h100_bf16_config()

    assert cfg.model is not None
    assert cfg.train is not None
    assert cfg.optimizer is not None
    assert cfg.scheduler is not None
    assert cfg.dataset is not None
    assert cfg.tokenizer is not None
    assert cfg.checkpoint is not None
    assert cfg.peft is None

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.sequence_parallel is False
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_audio_model is False
    assert cfg.model.vit_gradient_checkpointing is False
    assert cfg.model.multimodal_attn_impl == "auto"

    assert cfg.dataset.seq_length == 4096
    assert cfg.dataset.enable_in_batch_packing is False
    assert cfg.dataset.skip_getting_attention_mask_from_dataset is False
    assert cfg.dataset.hf_processor_path == "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    assert cfg.optimizer.lr == 5e-6


def test_qwen3_omni_hf_json_recipe_uses_direct_hf_source(monkeypatch):
    from megatron.bridge.data.builders import DirectHFSFTDatasetConfig, HFDatasetSourceConfig

    patch_recipe_module_global(monkeypatch, _qwen3_omni_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen3_omni_module.qwen3_omni_30b_a3b_sft_8gpu_h100_bf16_hf_json_config()

    assert isinstance(cfg.dataset, DirectHFSFTDatasetConfig)
    assert isinstance(cfg.dataset.source, HFDatasetSourceConfig)
    assert isinstance(cfg.dataset.validation_source, HFDatasetSourceConfig)
    assert isinstance(cfg.dataset.test_source, HFDatasetSourceConfig)
    assert cfg.dataset.source.path_or_dataset == "json"
    assert cfg.dataset.source.load_kwargs == {"data_files": {"train": None}}
    assert cfg.dataset.validation_source.load_kwargs == {"data_files": {"validation": None}}
    assert cfg.dataset.test_source.load_kwargs == {"data_files": {"test": None}}
    assert cfg.dataset.hf_processor_path == "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    assert cfg.dataset.enable_in_batch_packing is False
    assert cfg.dataset.skip_getting_attention_mask_from_dataset is False
