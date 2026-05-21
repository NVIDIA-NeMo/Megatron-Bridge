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

"""
Unit tests for DeepSeek recipe configuration builders.

Patterned after Qwen recipe tests: import all exported helpers from
`megatron.bridge.recipes.deepseek`, monkeypatch `AutoBridge` to a lightweight
fake that returns a minimal provider, and assert a valid ConfigContainer is
built with small overrides.
"""

import importlib
from types import SimpleNamespace
from typing import Callable

import pytest

from megatron.bridge.models.deepseek.deepseek_v4_bridge import DeepSeekV4Bridge
from megatron.bridge.models.mla_provider import MLAModelProvider


_deepseek_module = importlib.import_module("megatron.bridge.recipes.deepseek")
_DEEPSEEK_RECIPE_FUNCS = [
    getattr(_deepseek_module, name)
    for name in getattr(_deepseek_module, "__all__", [])
    if callable(getattr(_deepseek_module, name, None))
]


class _FakeModelCfg:
    # Minimal provider to accept attribute assignments used in recipes
    def __init__(self):
        # Provide defaults for attributes that recipes might read
        self.rotary_base = 10000.0
        self.num_moe_experts = 0
        self.apply_rope_fusion = False
        self.seq_length = 4096

    def finalize(self):
        return None


class _FakeBridge:
    last_hf_path = None
    last_kwargs = None

    def __init__(self):
        pass

    def to_megatron_provider(self, load_weights: bool = False):
        return _FakeModelCfg()

    @classmethod
    def from_hf_pretrained(cls, hf_path: str, **kwargs):
        _FakeBridge.last_hf_path = hf_path
        _FakeBridge.last_kwargs = kwargs
        return cls()


class _RealMLABridge(_FakeBridge):
    def to_megatron_provider(self, load_weights: bool = False):
        return MLAModelProvider(
            num_layers=1,
            hidden_size=512,
            num_attention_heads=8,
            num_query_groups=1,
            ffn_hidden_size=2048,
        )


def _toy_deepseek_v4_hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        attention_bias=False,
        attention_dropout=0.0,
        compress_ratios=[0, 4, 128, 4],
        first_k_dense_replace=0,
        head_dim=16,
        hidden_act="silu",
        hidden_dropout=0.0,
        hidden_size=512,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        index_head_dim=128,
        index_n_heads=64,
        index_topk=512,
        initializer_range=0.006,
        intermediate_size=2048,
        max_position_embeddings=1024,
        mlp_bias=False,
        moe_intermediate_size=512,
        n_routed_experts=8,
        n_shared_experts=1,
        norm_topk_prob=True,
        num_attention_heads=8,
        num_experts_per_tok=1,
        num_hash_layers=1,
        num_hidden_layers=4,
        num_key_value_heads=1,
        num_nextn_predict_layers=1,
        o_groups=8,
        o_lora_rank=192,
        q_lora_rank=192,
        qk_rope_head_dim=8,
        rms_norm_eps=1e-6,
        rope_scaling={
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 1,
            "original_max_position_embeddings": 1024,
            "type": "yarn",
        },
        rope_theta=10000.0,
        routed_scaling_factor=1.0,
        scoring_func="sqrtsoftplus",
        sliding_window=128,
        swiglu_limit=10.0,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        use_qk_norm=True,
        vocab_size=32000,
    )


class _DeepSeekV4BridgeFromToyConfig(_FakeBridge):
    def to_megatron_provider(self, load_weights: bool = False):
        return DeepSeekV4Bridge().provider_bridge(SimpleNamespace(config=_toy_deepseek_v4_hf_config()))


def _assert_basic_config(cfg):
    from megatron.bridge.training.config import ConfigContainer

    assert isinstance(cfg, ConfigContainer)
    assert cfg.model is not None
    assert cfg.train is not None
    assert cfg.optimizer is not None
    assert cfg.scheduler is not None
    assert cfg.dataset is not None
    assert cfg.logger is not None
    assert cfg.tokenizer is not None
    assert cfg.checkpoint is not None
    assert cfg.rng is not None
    assert cfg.train.global_batch_size >= 1
    assert cfg.train.micro_batch_size >= 1
    assert cfg.dataset.seq_length >= 1


@pytest.mark.parametrize("recipe_func", _DEEPSEEK_RECIPE_FUNCS)
def test_each_deepseek_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    # Monkeypatch AutoBridge in the specific module where the recipe function is defined
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    # DeepSeek recipes are all pretrain configs - call without parameters
    cfg = recipe_func()

    _assert_basic_config(cfg)

    # Ensure tokenizer is properly configured
    # DeepSeek pretrain recipes use either NullTokenizer or HuggingFaceTokenizer
    if cfg.tokenizer.tokenizer_type == "NullTokenizer":
        assert cfg.tokenizer.vocab_size is not None
    else:
        assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert cfg.tokenizer.tokenizer_model is not None

    # Parallelism and shaping
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1


def _build_deepseek_recipe(recipe_name: str, monkeypatch: pytest.MonkeyPatch):
    _FakeBridge.last_hf_path = None
    _FakeBridge.last_kwargs = None
    recipe_func = getattr(_deepseek_module, recipe_name)
    mod = importlib.import_module(recipe_func.__module__)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)
    return recipe_func()


def _build_deepseek_recipe_with_real_mla(recipe_name: str, monkeypatch: pytest.MonkeyPatch):
    recipe_func = getattr(_deepseek_module, recipe_name)
    mod = importlib.import_module(recipe_func.__module__)
    monkeypatch.setattr(mod, "AutoBridge", _RealMLABridge)
    return recipe_func()


def _build_deepseek_recipe_with_real_dsv4_bridge(recipe_name: str, monkeypatch: pytest.MonkeyPatch):
    recipe_func = getattr(_deepseek_module, recipe_name)
    mod = importlib.import_module(recipe_func.__module__)
    monkeypatch.setattr(mod, "AutoBridge", _DeepSeekV4BridgeFromToyConfig)
    return recipe_func()


def test_deepseek_v4_tiny_recipe_is_attention_only(monkeypatch: pytest.MonkeyPatch):
    cfg = _build_deepseek_recipe("deepseek_v4_tiny_pretrain_config", monkeypatch)

    assert _FakeBridge.last_kwargs == {"trust_remote_code": True}
    assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
    assert cfg.dataset.seq_length == cfg.model.seq_length
    assert cfg.model.experimental_attention_variant == "dsv4_hybrid"
    assert cfg.model.multi_latent_attention is True
    assert cfg.model.num_layers == 1
    assert cfg.model.csa_compress_ratios == [0]
    assert cfg.model.csa_window_size == 128
    assert cfg.model.dsa_indexer_n_heads == 64
    assert cfg.model.dsa_indexer_head_dim == 128
    assert cfg.model.dsa_indexer_topk == 512
    assert cfg.model.mtp_num_layers is None
    assert cfg.model.enable_hyper_connections is False
    assert cfg.model.use_fused_mhc is False
    assert cfg.model.num_moe_experts is None
    assert cfg.model.moe_shared_expert_intermediate_size is None
    assert cfg.model.moe_n_hash_layers == 0
    assert cfg.model.activation_func_clamp_value is None
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.model.pipeline_model_parallel_size == 1


def test_deepseek_v4_proxy_recipe_exercises_mhc_and_hash_moe(monkeypatch: pytest.MonkeyPatch):
    cfg = _build_deepseek_recipe("deepseek_v4_flash_proxy_pretrain_config", monkeypatch)

    assert _FakeBridge.last_kwargs == {"trust_remote_code": True}
    assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
    assert cfg.dataset.seq_length == cfg.model.seq_length
    assert cfg.model.experimental_attention_variant == "dsv4_hybrid"
    assert cfg.model.multi_latent_attention is True
    assert cfg.model.num_layers == 4
    assert cfg.model.csa_compress_ratios == [0, 4, 128, 4]
    assert cfg.model.csa_window_size == 128
    assert cfg.model.dsa_indexer_n_heads == 64
    assert cfg.model.dsa_indexer_head_dim == 128
    assert cfg.model.dsa_indexer_topk == 512
    assert cfg.model.mtp_num_layers is None
    assert len(cfg.model.csa_compress_ratios) == cfg.model.num_layers
    assert cfg.model.enable_hyper_connections is True
    assert cfg.model.num_residual_streams == 4
    assert cfg.model.mhc_sinkhorn_iterations == 20
    assert cfg.model.use_fused_mhc is False
    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.recompute_modules == ["mhc"]
    assert cfg.model.num_moe_experts == 8
    assert cfg.model.moe_shared_expert_intermediate_size == 512
    assert cfg.model.moe_n_hash_layers == 1
    assert cfg.model.actual_vocab_size == cfg.tokenizer.vocab_size
    assert cfg.model.activation_func_clamp_value == 10.0
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_layout == [
        ["embedding", "decoder", "decoder"],
        ["decoder", "decoder", "loss"],
    ]


def test_deepseek_v4_mtp_proxy_recipe_exercises_mhc_and_mtp(monkeypatch: pytest.MonkeyPatch):
    cfg = _build_deepseek_recipe("deepseek_v4_flash_mtp_proxy_pretrain_config", monkeypatch)

    assert _FakeBridge.last_kwargs == {"trust_remote_code": True}
    assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
    assert cfg.dataset.seq_length == cfg.model.seq_length
    assert cfg.model.experimental_attention_variant == "dsv4_hybrid"
    assert cfg.model.multi_latent_attention is True
    assert cfg.model.num_layers == 4
    assert cfg.model.mtp_num_layers == 1
    assert cfg.model.mtp_loss_scaling_factor == 0.1
    assert cfg.model.csa_compress_ratios == [0, 4, 128, 4, 0]
    assert len(cfg.model.csa_compress_ratios) == cfg.model.num_layers + cfg.model.mtp_num_layers
    assert cfg.model.enable_hyper_connections is True
    assert cfg.model.num_residual_streams == 4
    assert cfg.model.mhc_sinkhorn_iterations == 20
    assert cfg.model.use_fused_mhc is False
    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.recompute_modules == ["mhc"]
    assert cfg.model.num_moe_experts is None
    assert cfg.model.moe_shared_expert_intermediate_size is None
    assert cfg.model.moe_n_hash_layers == 0
    assert cfg.model.actual_vocab_size is None
    assert cfg.model.activation_func_clamp_value is None
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_layout == [
        ["embedding", "decoder", "decoder"],
        ["decoder", "decoder", "mtp", "loss"],
    ]


@pytest.mark.parametrize(
    "recipe_name,expected_qk_head_dim,expected_kv_lora_rank",
    [
        ("deepseek_v4_tiny_pretrain_config", 8, 8),
        ("deepseek_v4_flash_proxy_pretrain_config", 8, 8),
        ("deepseek_v4_flash_mtp_proxy_pretrain_config", 8, 8),
    ],
)
def test_deepseek_v4_recipe_provider_finalize(
    recipe_name: str,
    expected_qk_head_dim: int,
    expected_kv_lora_rank: int,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = _build_deepseek_recipe_with_real_mla(recipe_name, monkeypatch)

    cfg.model.finalize()

    assert cfg.model.experimental_attention_variant == "dsv4_hybrid"
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.qk_head_dim == expected_qk_head_dim
    assert cfg.model.kv_lora_rank == expected_kv_lora_rank
    if cfg.model.moe_n_hash_layers > 0:
        assert cfg.model.actual_vocab_size == cfg.tokenizer.vocab_size


@pytest.mark.parametrize(
    "recipe_name,expected_mhc,expected_mtp_layers,expected_shared_expert_size",
    [
        ("deepseek_v4_tiny_pretrain_config", False, None, None),
        ("deepseek_v4_flash_proxy_pretrain_config", True, None, 512),
        ("deepseek_v4_flash_mtp_proxy_pretrain_config", True, 1, None),
    ],
)
def test_deepseek_v4_recipe_overrides_real_bridge_defaults(
    recipe_name: str,
    expected_mhc: bool,
    expected_mtp_layers: int | None,
    expected_shared_expert_size: int | None,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = _build_deepseek_recipe_with_real_dsv4_bridge(recipe_name, monkeypatch)

    assert cfg.model.experimental_attention_variant == "dsv4_hybrid"
    assert cfg.model.enable_hyper_connections is expected_mhc
    assert cfg.model.use_fused_mhc is False
    assert cfg.model.mtp_num_layers == expected_mtp_layers
    assert cfg.model.moe_shared_expert_intermediate_size == expected_shared_expert_size
