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

from types import SimpleNamespace

import pytest

from megatron.bridge.recipes.qwen.gb200 import qwen35 as qwen35_recipe
from megatron.bridge.training.config import ConfigContainer


class _FakeProvider(SimpleNamespace):
    def finalize(self) -> None:
        return None


class _FakeBridge:
    def __init__(self, provider: _FakeProvider):
        self.provider = provider

    def to_megatron_provider(self, load_weights: bool = False) -> _FakeProvider:
        assert load_weights is False
        return self.provider


def test_qwen35_text_recipe_uses_nested_language_model_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The text recipe must select the causal-LM bridge, not the VLM bridge."""
    text_config = SimpleNamespace(architectures=None)
    provider = _FakeProvider(num_moe_experts=None)
    captured = {}

    def fake_from_pretrained(model_id: str):
        captured["model_id"] = model_id
        return SimpleNamespace(text_config=text_config)

    def fake_from_hf_config(config):
        captured["text_config"] = config
        return _FakeBridge(provider)

    monkeypatch.setattr(qwen35_recipe.AutoConfig, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(qwen35_recipe.AutoBridge, "from_hf_config", fake_from_hf_config)

    cfg = qwen35_recipe.qwen35_35b_a3b_pretrain_8gpu_gb200_bf16_config()

    assert isinstance(cfg, ConfigContainer)
    assert captured["model_id"] == "Qwen/Qwen3.5-35B-A3B-Base"
    assert captured["text_config"] is text_config
    assert text_config.architectures == ["Qwen3_5MoeForCausalLM"]
    assert cfg.model is provider
    assert cfg.tokenizer.tokenizer_model == "Qwen/Qwen3.5-35B-A3B-Base"
    assert cfg.dataset.seq_length == 4096
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.train.global_batch_size == 512
    assert cfg.train.micro_batch_size == 1
    assert cfg.model.recompute_granularity is None
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_flex_dispatcher_num_sms == 32
    assert cfg.model.moe_router_force_load_balancing is False
    assert cfg.model.cuda_graph_impl == "transformer_engine"
    assert cfg.model.cuda_graph_scope is None
    assert cfg.model.cuda_graph_modules == ["attn", "moe_router", "moe_preprocess"]
    assert cfg.model.use_te_rng_tracker is True
    assert cfg.rng.te_rng_tracker is True
    assert cfg.model.apply_rope_fusion is True
    assert cfg.model.cross_entropy_fusion_impl == "native"
    assert cfg.ddp.grad_reduce_in_fp32 is True
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.dataset.mmap_bin_files is True
