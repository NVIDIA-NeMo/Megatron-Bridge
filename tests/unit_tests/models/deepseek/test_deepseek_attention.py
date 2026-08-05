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

"""Unit tests for the DeepSeek MLA attention spec helpers."""

import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.deepseek.attention import (
    MLASelfAttentionWithoutQueryNorm,
    get_deepseek_decoder_block_spec,
)
from megatron.bridge.models.deepseek.deepseek_v2_bridge import DeepSeekV2Bridge
from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge


def _mla_submodules():
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=8, moe_grouped_gemm=False, qk_layernorm=True, multi_latent_attention=True
    )
    return spec.submodules.self_attention.submodules


def _config(q_lora_rank):
    return SimpleNamespace(
        q_lora_rank=q_lora_rank,
        qk_layernorm=True,
        qk_l2_norm=False,
        multi_latent_attention=True,
        experimental_attention_variant=None,
        normalization="RMSNorm",
        transformer_impl="transformer_engine",
    )


def _resolve(q_lora_rank):
    # `_resolve_qk_norm_config` only reads `self.config`, so bypass __init__ (which would
    # need process groups) but keep a real instance so its zero-arg `super()` resolves.
    attention = MLASelfAttentionWithoutQueryNorm.__new__(MLASelfAttentionWithoutQueryNorm)
    object.__setattr__(attention, "config", _config(q_lora_rank))
    return attention._resolve_qk_norm_config(_mla_submodules())


class TestMLASelfAttentionWithoutQueryNorm:
    """DeepSeek MLA must not gain a query norm the HF architecture does not define."""

    def test_no_query_lora_drops_the_fused_query_norm(self):
        """With q_lora_rank=None, HF has no query norm, so linear_q_proj must stay unfused."""
        resolved = _resolve(q_lora_rank=None)
        assert "LayerNorm" not in resolved["linear_q_proj"].__name__

    def test_no_query_lora_keeps_the_kv_norm(self):
        """kv_a_layernorm exists in every DeepSeek checkpoint and must still be built."""
        resolved = _resolve(q_lora_rank=None)
        assert "LayerNorm" in resolved["linear_kv_up_proj"].__name__

    def test_query_lora_is_untouched(self):
        """With a query LoRA the norm belongs on linear_q_up_proj and maps to q_a_layernorm."""
        resolved = _resolve(q_lora_rank=1536)
        assert "LayerNorm" in resolved["linear_q_up_proj"].__name__
        assert "LayerNorm" in resolved["linear_kv_up_proj"].__name__

    @pytest.mark.parametrize("q_lora_rank", [None, 1536])
    def test_standalone_norms_stay_disabled(self, q_lora_rank):
        """Norms stay fused into the projections; no standalone q/kv norm modules appear."""
        resolved = _resolve(q_lora_rank)
        assert resolved["q_layernorm"].__name__ == "IdentityOp"
        assert resolved["kv_layernorm"].__name__ == "IdentityOp"


class TestDeepSeekBridgesUseTheSpecHelper:
    """The bridges behind the affected models must route through the corrected spec builder.

    `deepseek-ai/DeepSeek-V2-Lite` goes through DeepSeekV2Bridge and
    `kakaocorp/kanana-2-30b-a3b-thinking` through DeepSeekV3Bridge, and both ship
    `q_lora_rank: null`.
    """

    @pytest.mark.parametrize("bridge_cls", [DeepSeekV2Bridge, DeepSeekV3Bridge])
    def test_provider_bridge_installs_the_spec_helper(self, bridge_cls, monkeypatch):
        """Both bridges must build their decoder block through get_deepseek_decoder_block_spec."""
        provider = SimpleNamespace()
        monkeypatch.setattr(
            MegatronModelBridge,
            "provider_bridge",
            lambda self, hf_pretrained: provider,
        )
        hf_pretrained = Mock()
        hf_pretrained.config = SimpleNamespace(
            first_k_dense_replace=1,
            num_hidden_layers=4,
            moe_intermediate_size=128,
            n_shared_experts=1,
            q_lora_rank=None,
        )

        bridge_cls().provider_bridge(hf_pretrained)

        assert provider.qk_layernorm is True, "the KV norm still has to be requested"
        assert provider.transformer_layer_spec.func is get_deepseek_decoder_block_spec

    def test_spec_helper_is_a_drop_in_for_the_mcore_builder(self):
        """The helper must keep the signature the provider calls it with."""
        parameters = inspect.signature(get_deepseek_decoder_block_spec).parameters
        assert "config" in parameters
        assert parameters["use_transformer_engine"].kind is inspect.Parameter.KEYWORD_ONLY
