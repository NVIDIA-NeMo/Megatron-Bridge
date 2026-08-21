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
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.multi_latent_attention import MLASelfAttention

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.deepseek import attention as attention_module
from megatron.bridge.models.deepseek.attention import (
    MLASelfAttentionWithoutQueryNorm,
    get_deepseek_decoder_block_spec,
    replace_mla_self_attention,
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


def _local_mla_submodules(qk_layernorm=True):
    """MLA submodules as the non-Transformer-Engine backend builds them.

    This backend differs from the TE one in exactly the way that matters here: it puts a
    real module in ``q_layernorm`` instead of folding the norm into the projection.
    """
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    spec = get_gpt_layer_local_spec(
        num_experts=8, moe_grouped_gemm=False, qk_layernorm=qk_layernorm, multi_latent_attention=True
    )
    return spec.submodules.self_attention.submodules


class TestSpecBuilderContract:
    """The helper stands in for `get_gpt_decoder_block_spec`, so it must accept its arguments."""

    def test_signature_matches_the_mcore_builder(self):
        """Every parameter MCore's builder takes has to be taken here too.

        `GPTModelProvider.provide()` decides whether to pass `vp_stage` by inspecting this
        signature. A missing parameter is not a type error at the call site; it silently
        routes interleaved pipeline parallelism into MCore's layer-offset helper without a
        virtual stage, which asserts.
        """
        ours = inspect.signature(get_deepseek_decoder_block_spec).parameters
        theirs = inspect.signature(get_gpt_decoder_block_spec).parameters
        assert list(ours) == list(theirs)
        for name, their_param in theirs.items():
            assert ours[name].default == their_param.default, name

    def test_virtual_stage_arguments_reach_mcore(self, monkeypatch):
        """vp_stage and pp_rank must be forwarded, not dropped."""
        seen = {}

        def _capture(config, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(layer_specs=[])

        monkeypatch.setattr(attention_module, "get_gpt_decoder_block_spec", _capture)
        get_deepseek_decoder_block_spec(_config(q_lora_rank=None), use_transformer_engine=True, vp_stage=1, pp_rank=2)

        assert seen["vp_stage"] == 1
        assert seen["pp_rank"] == 2

    def test_the_partial_the_bridges_build_still_declares_vp_stage(self):
        """The provider inspects the partial, so binding use_transformer_engine must not hide it."""
        bound = partial(get_deepseek_decoder_block_spec, use_transformer_engine=True)
        assert "vp_stage" in inspect.signature(bound).parameters


class TestLocalBackend:
    """The local backend cannot express this architecture; say so, and say it clearly."""

    def test_local_backend_is_rejected_by_name(self):
        """Without a query LoRA the local backend has no viable MLA spec at all.

        MCore builds `linear_q_proj` from the backend's fused norm+linear implementation
        whenever `qk_layernorm` is on, and the local backend has none. Keeping the query
        norm instead trips `_raise_unused_q_norm`, so both branches are closed, and
        DeepSeek needs `qk_layernorm` on for `kv_a_layernorm`. Left to MCore this surfaces
        as `RuntimeError: qk_layernorm requires TransformerEngine ...` from deep inside the
        resolver, which does not say which model setting is at fault.
        """
        attention = MLASelfAttentionWithoutQueryNorm.__new__(MLASelfAttentionWithoutQueryNorm)
        config = _config(q_lora_rank=None)
        config.transformer_impl = "local"
        object.__setattr__(attention, "config", config)

        with pytest.raises(ValueError, match="transformer_engine"):
            attention._resolve_qk_norm_config(_local_mla_submodules())

    def test_local_spec_with_a_query_lora_is_untouched(self):
        """The query-LoRA path must resolve exactly as it did before."""
        attention = MLASelfAttentionWithoutQueryNorm.__new__(MLASelfAttentionWithoutQueryNorm)
        config = _config(q_lora_rank=1536)
        config.transformer_impl = "local"
        object.__setattr__(attention, "config", config)

        resolved = attention._resolve_qk_norm_config(_local_mla_submodules())

        assert resolved is not None


class TestStandaloneMTPStage:
    """A standalone MTP stage owns no decoder layers and re-derives its spec from MCore."""

    def test_transform_replaces_attention_on_a_bare_layer_spec(self):
        """The MTP fallback hands over one layer spec, not a block, so both must work."""
        layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=8, moe_grouped_gemm=False, qk_layernorm=True, multi_latent_attention=True
        )
        assert layer_spec.submodules.self_attention.module is MLASelfAttention

        replace_mla_self_attention(_config(q_lora_rank=None), layer_spec)

        assert layer_spec.submodules.self_attention.module is MLASelfAttentionWithoutQueryNorm

    def test_transform_leaves_the_query_lora_path_alone(self):
        """With a query LoRA the stock attention is correct and must stay."""
        layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=8, moe_grouped_gemm=False, qk_layernorm=True, multi_latent_attention=True
        )

        replace_mla_self_attention(_config(q_lora_rank=1536), layer_spec)

        assert layer_spec.submodules.self_attention.module is MLASelfAttention

    @pytest.mark.parametrize("bridge_cls", [DeepSeekV2Bridge, DeepSeekV3Bridge])
    def test_bridges_register_the_mtp_transform(self, bridge_cls, monkeypatch):
        """Without this the MTP layer on a standalone stage regains the query norm."""
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

        assert provider.mtp_layer_spec_transform is replace_mla_self_attention
