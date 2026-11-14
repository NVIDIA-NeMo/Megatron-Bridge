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

"""The Shensi bridge registers, maps the new architecture, and validates the provider."""

from dataclasses import fields
from types import SimpleNamespace

import pytest

from megatron.bridge.models.conversion.model_bridge import get_model_bridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.shensi import ShensiBridge, ShensiModelProvider
from megatron.bridge.models.shensi.configuration_shensi import ShensiTransformerConfig


def _hf_config():
    return SimpleNamespace(
        num_hidden_layers=4,
        mlp_layer_types=["dense", "dense", "sparse", "sparse"],
        first_k_dense_replace=2,
        attn_res_block_size=2,
        attn_res_block_summary_rank=32,
        indexer_types=["full", "shared", "full", "shared"],
        index_topk_freq=2,
        n_routed_experts=8,
        hidden_size=128,
        head_dim=64,
        qk_nope_head_dim=64,
        qk_rope_head_dim=0,
        v_head_dim=64,
        num_attention_heads=4,
        q_lora_rank=32,
        latent_size=32,
        deepemb_kv_rank=16,
        kv_emb_rank=16,
        deepemb_rank=8,
        index_n_heads=8,
        index_head_dim=16,
        index_topk=32,
        index_topk_blocks=4,
        index_block_size=4,
        index_latent_rank=4,
        linear_num_heads=4,
        linear_head_dim=16,
        linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0,
        hc_mult=4,
        hc_active_streams=2,
        hc_adapter_rank=4,
        hc_fixed_streams=1,
        hc_conv_kernels=[4],
        hc_stream_rank=32,
        hc_eps=1e-6,
        attn_res_rank=8,
        rosa_embed_dim=128,
        rosa_layer_ids=[],
        rosa_reset_token_id=None,
        moe_intermediate_size=32,
        erc_loss_alpha=0.5,
        erc_loss_coef=1.0,
        vocab_size=128,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        num_experts_per_tok=2,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        attention_dropout=0.0,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )


def _registry():
    return ShensiBridge.mapping_registry(SimpleNamespace(hf_config=_hf_config()))


def _by_megatron(registry):
    return {m.megatron_param: m for m in registry.get_all_mappings()}


class TestShensiDispatch:
    def test_register_bridge_resolves(self):
        got = get_model_bridge("ShensiForCausalLM")
        assert got is ShensiBridge or isinstance(got, ShensiBridge), type(got)

    def test_provider_is_mcore_shensi_config(self):
        assert issubclass(ShensiModelProvider, ShensiTransformerConfig)
        names = {f.name for f in fields(ShensiModelProvider)}
        for key in ("attn_res_block_size", "hc_mult", "linear_num_heads", "index_block_size"):
            assert key in names, key

    def test_provider_implements_provide(self):
        assert callable(ShensiModelProvider.provide)

    def _provider(self):
        return ShensiModelProvider()

    def test_finalize_rejects_mcore_hyper_connections(self):
        provider = self._provider()
        provider.enable_mhc_connections = True
        with pytest.raises(ValueError, match="enable_mhc_connections"):
            provider.finalize()

    def test_finalize_rejects_mcore_wide_residual(self):
        from megatron.core.transformer import WideResidualConfig

        provider = self._provider()
        provider.wide_residual = WideResidualConfig(num_streams=4)
        with pytest.raises(ValueError, match="wide_residual"):
            provider.finalize()

    def test_finalize_rejects_inconsistent_stream_kernels(self):
        provider = self._provider()
        provider.hc_conv_kernels = [4]
        with pytest.raises(ValueError, match=r"hc_mult .*hc_conv_kernels"):
            provider.finalize()

    def test_finalize_rejects_invalid_active_streams(self):
        provider = self._provider()
        provider.hc_conv_kernels = [4, 8, 12]
        provider.hc_active_streams = 5
        with pytest.raises(ValueError, match="hc_fixed_streams"):
            provider.finalize()


class TestShensiMappings:
    def test_top_level(self):
        reg = _by_megatron(_registry())
        assert isinstance(reg["embedding.word_embeddings.weight"], AutoMapping)
        assert isinstance(reg["norm.weight"], ReplicatedMapping)
        assert isinstance(reg["hc_head.hc_fn"], ReplicatedMapping)
        assert isinstance(reg["attn_bank.basis"], ReplicatedMapping)
        assert isinstance(reg["output_attn_res.g_proj.bias"], ReplicatedMapping)

    def test_attention_residual_is_same_named(self):
        reg = _by_megatron(_registry())
        for layer in (0, 1):
            for slot in ("self_attention_res", "mlp_res"):
                for suffix in ("q_proj.weight", "k_proj.weight", "g_proj.weight", "g_proj.bias", "scale"):
                    assert f"decoder.layers.{layer}.{slot}.{suffix}" in reg

    def test_conditional_hyper_connection_mappings(self):
        reg = _by_megatron(_registry())
        assert "decoder.layers.0.attn_hc.fn" in reg
        assert "decoder.layers.0.attn_hc.adapters.0.0.weight" in reg
        assert "decoder.layers.0.attn_hc.adapters.0.1.weight" in reg
        assert "decoder.layers.0.attn_hc.decay_logit" in reg
        assert "decoder.layers.0.attn_hc.res.weight" not in reg

    def test_dense_layer_maps_separate_mlp_projections(self):
        reg = _by_megatron(_registry())
        assert "decoder.layers.0.mlp.linear_fc1.weight" in reg
        assert "decoder.layers.0.mlp.linear_fc2.weight" in reg
        assert "decoder.layers.0.mlp.deepemb.weight" in reg

    def test_moe_layer_uses_fused_expert_mappings(self):
        reg = _by_megatron(_registry())
        assert isinstance(reg["decoder.layers.3.mlp.experts.linear_fc1.weight*"], FusedGatedExpertMapping)
        assert isinstance(reg["decoder.layers.3.mlp.experts.linear_fc2.weight*"], FusedExpertMapping)
        assert isinstance(reg["decoder.layers.3.mlp.router.weight"], ReplicatedMapping)

    def test_indexed_layer_has_hisa_indexer(self):
        reg = _by_megatron(_registry())
        for key in (
            "decoder.layers.0.self_attention.indexer.linear_wq_b.weight",
            "decoder.layers.0.self_attention.indexer.linear_key.weight",
            "decoder.layers.0.self_attention.indexer.linear_key.weight",
            "decoder.layers.0.self_attention.indexer.linear_key_up.weight",
            "decoder.layers.0.self_attention.indexer.linear_key_emb.weight",
            "decoder.layers.0.self_attention.kv.weight",
        ):
            assert key in reg, key

    def test_linear_layer_has_kda_projections(self):
        reg = _by_megatron(_registry())
        for key in (
            "decoder.layers.1.self_attention.in_proj.weight",
            "decoder.layers.1.self_attention.conv1d.weight",
            "decoder.layers.1.self_attention.beta_proj.weight",
            "decoder.layers.1.self_attention.f_a_proj.weight",
            "decoder.layers.1.self_attention.g_b_proj.weight",
            "decoder.layers.1.self_attention.out_norm.weight",
        ):
            assert key in reg, key
        assert "decoder.layers.1.self_attention.indexer.linear_wq_b.weight" not in reg

    def test_stream_token_table_uses_the_padded_vocabulary(self):
        """The vocabulary-parallel stream table pads the reference rows and crops them on export."""
        import torch
        from torch import nn

        from megatron.bridge.models.shensi.shensi_bridge import _StreamTokenEmbeddingMapping

        mapping = _by_megatron(_registry())["stream_embeddings.token_embeddings.weight"]
        assert isinstance(mapping, _StreamTokenEmbeddingMapping)

        holder = nn.Module()
        holder.stream_embeddings = nn.Module()
        holder.stream_embeddings.token_embeddings = nn.Embedding(16, 6)
        mapping = _StreamTokenEmbeddingMapping(
            "stream_embeddings.token_embeddings.weight",
            "model.stream_embeddings.token_embeddings.weight",
            hf_vocab_size=12,
        )
        reference = torch.arange(12 * 6, dtype=torch.float32).view(12, 6)
        loaded = mapping.hf_to_megatron(reference, holder)
        assert loaded.shape == (16, 6)
        torch.testing.assert_close(loaded[:12], reference)
        assert not loaded[12:].any()
        with torch.no_grad():
            holder.stream_embeddings.token_embeddings.weight.copy_(loaded)
        exported = mapping.megatron_to_hf(holder.stream_embeddings.token_embeddings.weight.detach(), holder)
        torch.testing.assert_close(exported["model.stream_embeddings.token_embeddings.weight"], reference)
