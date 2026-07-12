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
import torch
from transformers import PretrainedConfig

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.ernie.h100.ernie45 import ernie45_21b_a3b_pretrain_8gpu_h100_bf16_config
from megatron.bridge.recipes.gemma.h100.gemma_text import (
    gemma4_26b_a4b_pretrain_8gpu_h100_bf16_config,
    gemma4_31b_pretrain_8gpu_h100_bf16_config,
    gemma_2b_pretrain_1gpu_h100_bf16_config,
)
from megatron.bridge.recipes.glm.h100.glm47 import glm47_flash_31b_pretrain_8gpu_h100_bf16_config
from megatron.bridge.recipes.llama.h100.llama2 import llama2_7b_pretrain_2gpu_h100_bf16_config
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama3_8b_pretrain_2gpu_h100_bf16_config,
    llama31_8b_pretrain_2gpu_h100_bf16_config,
    llama31_70b_pretrain_32gpu_h100_bf16_config,
    llama33_70b_pretrain_32gpu_h100_bf16_config,
)
from megatron.bridge.recipes.mimo_v2_flash.h100.mimo_v2_flash import (
    mimo_v2_flash_310b_pretrain_16gpu_h100_bf16_config,
)
from megatron.bridge.recipes.nemotronh.h100.nemotron_3_super import (
    nemotron_3_super_pretrain_8gpu_h100_bf16_config,
)
from megatron.bridge.recipes.qwen.h100.qwen3_next import qwen3_next_80b_a3b_pretrain_32gpu_h100_bf16_config
from megatron.bridge.recipes.qwen.h100.qwen35 import qwen35_35b_a3b_pretrain_8gpu_h100_bf16_config
from megatron.bridge.recipes.sarvam.h100.sarvam import sarvam_30b_pretrain_8gpu_h100_bf16_config


pytestmark = pytest.mark.unit


def test_generic_mimo_config_dispatches_through_auto_bridge():
    hf_config = PretrainedConfig(
        name_or_path="XiaomiMiMo/MiMo-7B-Base",
        architectures=["MiMoForCausalLM"],
        attention_bias=True,
        attention_dropout=0.0,
        hidden_size=4096,
        initializer_range=0.02,
        intermediate_size=11008,
        max_position_embeddings=32768,
        model_type="mimo",
        num_attention_heads=32,
        num_hidden_layers=36,
        num_key_value_heads=8,
        num_nextn_predict_layers=1,
        rms_norm_eps=1e-5,
        rope_theta=640000.0,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        vocab_size=151680,
    )

    provider = AutoBridge.from_hf_config(hf_config).to_megatron_provider(load_weights=False)

    assert provider.num_layers == 36
    assert provider.hidden_size == 4096
    assert provider.ffn_hidden_size == 11008
    assert provider.num_query_groups == 8
    assert provider.mtp_num_layers == 1
    assert provider.mtp_loss_scaling_factor == 0.1


def test_mimo_v2_flash_pretrain_uses_pinned_config_as_source_of_truth(monkeypatch):
    calls = []
    hf_config = SimpleNamespace()
    provider = SimpleNamespace()

    class _Bridge:
        def to_megatron_provider(self, *, load_weights):
            assert load_weights is False
            return provider

    def _from_pretrained(model_id, **kwargs):
        calls.append(("config", model_id, kwargs))
        return hf_config

    def _from_hf_config(config):
        calls.append(("bridge", config))
        return _Bridge()

    monkeypatch.setattr(
        "megatron.bridge.recipes.mimo_v2_flash.h100.mimo_v2_flash.PretrainedConfig.from_pretrained",
        _from_pretrained,
    )
    monkeypatch.setattr(AutoBridge, "from_hf_config", _from_hf_config)

    config = mimo_v2_flash_310b_pretrain_16gpu_h100_bf16_config()

    assert calls == [
        (
            "config",
            "XiaomiMiMo/MiMo-V2-Flash",
            {"revision": "1afd314a2406c282e0956375c34a676501c78649"},  # pragma: allowlist secret
        ),
        ("bridge", hf_config),
    ]
    assert hf_config.name_or_path == "XiaomiMiMo/MiMo-V2-Flash"
    assert config.model.tensor_model_parallel_size == 1
    assert config.model.pipeline_model_parallel_size == 1
    assert config.model.expert_model_parallel_size == 16


def test_ernie45_pretrain_shards_logits_with_tensor_parallelism(monkeypatch):
    calls = []
    provider = SimpleNamespace()

    class _Bridge:
        def to_megatron_provider(self, *, load_weights):
            assert load_weights is False
            return provider

    def _from_hf_pretrained(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return _Bridge()

    monkeypatch.setattr(AutoBridge, "from_hf_pretrained", _from_hf_pretrained)

    config = ernie45_21b_a3b_pretrain_8gpu_h100_bf16_config()

    assert calls == [
        (
            "baidu/ERNIE-4.5-21B-A3B-PT",
            {
                "revision": "87db95487941cb39592ee0abca3b9155a6d19c5c",  # pragma: allowlist secret
            },
        )
    ]
    assert config.model.tensor_model_parallel_size == 2
    assert config.model.recompute_granularity == "selective"
    assert config.model.recompute_modules == ["core_attn"]


def test_gemma4_moe_pretrain_uses_tp_for_rmsnorm_headroom(monkeypatch):
    provider = SimpleNamespace()

    class _Bridge:
        def to_megatron_provider(self, *, load_weights):
            assert load_weights is False
            return provider

    monkeypatch.setattr(
        AutoBridge,
        "from_hf_pretrained",
        lambda *_args, **_kwargs: _Bridge(),
    )

    config = gemma4_26b_a4b_pretrain_8gpu_h100_bf16_config()

    assert config.model.tensor_model_parallel_size == 2
    assert config.model.recompute_granularity == "selective"
    assert config.model.recompute_modules == ["core_attn"]


def test_gemma4_dense_pretrain_avoids_unsupported_pipeline_parallelism(monkeypatch):
    provider = SimpleNamespace()

    class _Bridge:
        def to_megatron_provider(self, *, load_weights):
            assert load_weights is False
            return provider

    monkeypatch.setattr(
        AutoBridge,
        "from_hf_pretrained",
        lambda *_args, **_kwargs: _Bridge(),
    )

    config = gemma4_31b_pretrain_8gpu_h100_bf16_config()

    assert config.model.tensor_model_parallel_size == 8
    assert config.model.pipeline_model_parallel_size == 1


def test_glm47_flash_pretrain_uses_full_recompute_for_nccl_headroom(monkeypatch):
    provider = SimpleNamespace()

    class _Bridge:
        def to_megatron_provider(self, *, load_weights):
            assert load_weights is False
            return provider

    monkeypatch.setattr(
        AutoBridge,
        "from_hf_pretrained",
        lambda *_args, **_kwargs: _Bridge(),
    )

    config = glm47_flash_31b_pretrain_8gpu_h100_bf16_config()

    assert config.model.recompute_granularity == "full"
    assert config.model.recompute_method == "uniform"
    assert config.model.recompute_num_layers == 1
    assert config.model.recompute_modules is None


def test_sarvam_pretrain_uses_full_recompute_for_dispatch_headroom(monkeypatch):
    hf_config = SimpleNamespace()
    provider = SimpleNamespace()

    class _Bridge:
        def to_megatron_provider(self, *, load_weights):
            assert load_weights is False
            return provider

    monkeypatch.setattr(
        "megatron.bridge.recipes.sarvam.h100.sarvam.PretrainedConfig.from_pretrained",
        lambda *_args, **_kwargs: hf_config,
    )
    monkeypatch.setattr(AutoBridge, "from_hf_config", lambda config: _Bridge())

    config = sarvam_30b_pretrain_8gpu_h100_bf16_config()

    assert hf_config.name_or_path == "sarvamai/sarvam-30b"
    assert config.model.recompute_granularity == "full"
    assert config.model.recompute_method == "uniform"
    assert config.model.recompute_num_layers == 1
    assert config.model.recompute_modules is None


def test_qwen35_moe_pretrain_uses_full_recompute_for_hybrid_headroom(monkeypatch):
    text_config = SimpleNamespace()
    provider = SimpleNamespace()

    class _Bridge:
        def to_megatron_provider(self, *, load_weights):
            assert load_weights is False
            return provider

    monkeypatch.setattr(
        "megatron.bridge.recipes.qwen.h100.qwen35.AutoConfig.from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(text_config=text_config),
    )
    monkeypatch.setattr(AutoBridge, "from_hf_config", lambda config: _Bridge())

    config = qwen35_35b_a3b_pretrain_8gpu_h100_bf16_config()

    assert text_config.architectures == ["Qwen3_5MoeForCausalLM"]
    assert text_config.name_or_path == "Qwen/Qwen3.5-35B-A3B"
    assert config.model.tensor_model_parallel_size == 2
    assert config.model.recompute_granularity == "full"
    assert config.model.recompute_method == "uniform"
    assert config.model.recompute_num_layers == 1
    assert config.model.recompute_modules is None


def test_qwen3_next_pretrain_uses_bf16_optimizer_state(monkeypatch):
    provider = SimpleNamespace()

    class _Bridge:
        def to_megatron_provider(self, *, load_weights):
            assert load_weights is False
            return provider

    monkeypatch.setattr(
        AutoBridge,
        "from_hf_pretrained",
        lambda *_args, **_kwargs: _Bridge(),
    )

    config = qwen3_next_80b_a3b_pretrain_32gpu_h100_bf16_config()

    assert config.optimizer.use_precision_aware_optimizer is True
    assert config.optimizer.main_params_dtype == torch.float16
    assert config.optimizer.main_grads_dtype == torch.bfloat16
    assert config.optimizer.exp_avg_dtype == torch.bfloat16
    assert config.optimizer.exp_avg_sq_dtype == torch.bfloat16


def test_nemotron3_super_h100_pretrain_uses_hopper_fp8(monkeypatch):
    provider = SimpleNamespace()

    class _Bridge:
        def to_megatron_provider(self, *, load_weights):
            assert load_weights is False
            return provider

    monkeypatch.setattr(
        "megatron.bridge.recipes.nemotronh.h100.nemotron_3_super.AutoBridge.from_hf_pretrained",
        lambda *_args, **_kwargs: _Bridge(),
    )

    config = nemotron_3_super_pretrain_8gpu_h100_bf16_config()

    assert config.mixed_precision.fp8 == "hybrid"
    assert config.mixed_precision.fp8_recipe == "tensorwise"
    assert config.mixed_precision.grad_reduce_in_fp32 is False


def test_gemma_pretrain_uses_pinned_hf_config_as_source_of_truth(monkeypatch):
    calls = []
    provider = SimpleNamespace()

    class _Bridge:
        def to_megatron_provider(self, *, load_weights):
            assert load_weights is False
            return provider

    def _from_hf_pretrained(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return _Bridge()

    monkeypatch.setattr(AutoBridge, "from_hf_pretrained", _from_hf_pretrained)

    config = gemma_2b_pretrain_1gpu_h100_bf16_config()

    assert calls == [
        (
            "google/gemma-2b",
            {"revision": "9cf48e52b224239de00d483ec8eb84fb8d0f3a3a"},  # pragma: allowlist secret
        )
    ]
    assert config.model is provider


def test_llama2_pretrain_uses_hf_config_as_source_of_truth(monkeypatch):
    calls = []
    provider = SimpleNamespace()

    class _Bridge:
        def to_megatron_provider(self, *, load_weights):
            assert load_weights is False
            return provider

    def _from_hf_pretrained(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return _Bridge()

    monkeypatch.setattr(AutoBridge, "from_hf_pretrained", _from_hf_pretrained)

    config = llama2_7b_pretrain_2gpu_h100_bf16_config()

    assert calls == [("meta-llama/Llama-2-7b-hf", {})]
    assert config.model is provider


def test_llama3_pretrain_configs_use_their_own_hf_sources(monkeypatch):
    calls = []

    class _Bridge:
        def to_megatron_provider(self, *, load_weights):
            assert load_weights is False
            return SimpleNamespace()

    def _from_hf_pretrained(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return _Bridge()

    monkeypatch.setattr(AutoBridge, "from_hf_pretrained", _from_hf_pretrained)

    llama3_8b_pretrain_2gpu_h100_bf16_config()
    llama31_8b_pretrain_2gpu_h100_bf16_config()
    llama31_70b_pretrain_32gpu_h100_bf16_config()
    llama33 = llama33_70b_pretrain_32gpu_h100_bf16_config()

    assert calls == [
        ("meta-llama/Meta-Llama-3-8B", {}),
        ("meta-llama/Meta-Llama-3.1-8B", {}),
        ("meta-llama/Meta-Llama-3.1-70B", {}),
        (
            "meta-llama/Llama-3.3-70B-Instruct",
            {"revision": "6f6073b423013f6a7d4d9f39144961bfbfbc386b"},  # pragma: allowlist secret
        ),
    ]
    assert llama33.optimizer.use_precision_aware_optimizer is True
    assert llama33.optimizer.main_params_dtype == torch.float16
    assert llama33.optimizer.main_grads_dtype == torch.bfloat16
    assert llama33.optimizer.exp_avg_dtype == torch.bfloat16
    assert llama33.optimizer.exp_avg_sq_dtype == torch.bfloat16
