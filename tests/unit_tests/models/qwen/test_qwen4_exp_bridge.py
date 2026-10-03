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

"""Unit tests for the Qwen4-Exp (Qwen3.8-Flash-Next) bridge."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen.qwen4_exp_bridge import (
    PLENGramEmbeddingMapping,
    Qwen4ExpBridge,
    Qwen4ExpTextBridge,
    get_qwen4_exp_hf_lm_prefix,
    linear_attention_pattern_from_hf,
    ple_local_rows_from_shards,
)


def _text_config_dict(num_layers=8):
    layer_types = ["qwen_sparse_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(num_layers)]
    return {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 248044,
        "eos_token_id": 248044,
        "hc_count": 4,
        "hc_lowrank": 320,
        "head_dim": 256,
        "heads_per_ngram": 8,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "indexer_head_dim": 128,
        "indexer_kv_heads": 1,
        "indexer_n_heads": 4,
        "initializer_range": 0.02,
        "layer_types": layer_types,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "make_ngram_vocab_size_divisible_by": 128,
        "max_position_embeddings": 262144,
        "model_type": "qwen4_exp_text",
        "moe_intermediate_size": 640,
        "mtp_num_hidden_layers": 1,
        "ngram_size": 3,
        "ngram_vocab_size_base": 20000000,
        "num_attention_heads": 24,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "num_hidden_layers": num_layers,
        "num_key_value_heads": 2,
        "output_gate_type": "sigmoid",
        "ple_conv_kernel_size": 4,
        "ple_embed_dim": 2560,
        "ple_layer_ids": [2],
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "router_aux_loss_coef": 0.001,
        "seed": 1234,
        "shared_expert_intermediate_size": 640,
        "split_ngram_parts": 128,
        "tie_word_embeddings": False,
        "vocab_size": 248320,
    }


def _namespace(d):
    return SimpleNamespace(**d)


@pytest.fixture
def vl_config():
    return _namespace(
        {
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "tie_word_embeddings": False,
            "text_config": _namespace(_text_config_dict()),
            "vision_config": _namespace({"depth": 27}),
        }
    )


@pytest.fixture
def mock_pretrained(vl_config):
    pretrained = Mock(spec=PreTrainedCausalLM)
    pretrained.config = vl_config
    pretrained.model = Mock()
    pretrained.model.dtype = torch.bfloat16
    # Checkpoint keys: fused experts and 128 PLE shards, like the released checkpoint.
    source = Mock()
    source.get_all_keys.return_value = [
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
    ] + [f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}.weight" for i in range(128)]
    pretrained.state = Mock()
    pretrained.state.source = source
    return pretrained


class TestQwen4ExpProvider:
    def test_registration(self):
        assert issubclass(Qwen4ExpBridge, MegatronModelBridge)
        assert issubclass(Qwen4ExpTextBridge, Qwen4ExpBridge)

    def test_layer_pattern_accepts_qsa_layer_type(self, vl_config):
        pattern = linear_attention_pattern_from_hf(vl_config.text_config)
        assert pattern == [1, 1, 1, 0, 1, 1, 1, 0]
        with pytest.raises(ValueError):
            linear_attention_pattern_from_hf(_namespace({"layer_types": ["bogus"]}))

    def test_prefixes(self, vl_config):
        assert get_qwen4_exp_hf_lm_prefix(vl_config) == "model.language_model."
        assert get_qwen4_exp_hf_lm_prefix(_namespace({"text_config": None})) == "model."

    def test_provider_bridge(self, mock_pretrained, vl_config):
        provider = Qwen4ExpBridge().provider_bridge(mock_pretrained)
        text = vl_config.text_config

        assert isinstance(provider, GPTModelProvider)
        assert provider.num_layers == text.num_hidden_layers
        assert provider.hidden_size == 2560
        assert provider.num_attention_heads == 24 and provider.num_query_groups == 2
        assert provider.kv_channels == 256
        assert provider.rotary_base == 10000000 and provider.rotary_percent == 0.25
        assert provider.vocab_size == 248320 and not provider.share_embeddings_and_output_weights

        # Qwen3.5-MoE base
        assert provider.experimental_attention_variant == "gdn"
        assert provider.linear_attention_freq == [1, 1, 1, 0, 1, 1, 1, 0]
        assert provider.linear_num_value_heads == 48 and provider.linear_num_key_heads == 16
        assert provider.linear_attention_output_gate_activation == "sigmoid"
        assert provider.attention_output_gate and provider.qk_layernorm and provider.layernorm_zero_centered_gamma
        assert provider.num_moe_experts == 512 and provider.moe_router_topk == 10
        assert provider.moe_ffn_hidden_size == 640 and provider.moe_shared_expert_intermediate_size == 640
        assert provider.moe_shared_expert_gate and provider.moe_grouped_gemm

        # Qwen4-Exp additions
        assert provider.enable_mhc_connections and provider.mhc_variant == "gated_residual"
        assert provider.mhc_num_residual_streams == 4 and provider.mhc_gated_residual_rank == 320
        assert provider.qsa_indexer_n_heads == 4 and provider.qsa_indexer_head_dim == 128
        assert provider.qsa_indexer_budget == 2048 and provider.qsa_indexer_compress_ratio == 4
        assert provider.ple_layer_ids == [2] and provider.ple_eos_token_id == 248044
        assert provider.ple_unigram_vocab_size == 248320 and provider.ple_ngram_vocab_size_base == 20000000
        assert provider.mtp_num_layers is None

    def test_provider_finalizes(self, mock_pretrained):
        """The provider must pass megatron-core's config validation once parallelism is set."""
        provider = Qwen4ExpBridge().provider_bridge(mock_pretrained)
        provider.tensor_model_parallel_size = 1
        provider.pipeline_model_parallel_size = 1
        provider.finalize()
        assert provider.ple_layer_ids == [2]


class TestQwen4ExpMappings:
    def test_mapping_registry_covers_all_components(self, mock_pretrained, vl_config):
        bridge = Qwen4ExpBridge()
        bridge.hf_pretrained = mock_pretrained
        registry = bridge.mapping_registry()
        hf = "model.language_model."

        expectations = {
            "embedding.word_embeddings.weight": f"{hf}embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.hc_norm.weight": f"{hf}hyper_connection_mixer.hc_norm.weight",
            "decoder.final_layernorm.input_mix_weight_down.weight": f"{hf}hyper_connection_mixer.input_mix_weight_down.weight",
            "decoder.layers.3.self_attention_hyper_connection.block_inject_weight.weight": f"{hf}layers.3.attn_hyper_connection.block_inject_weight.weight",
            "decoder.layers.3.mlp_hyper_connection.input_mix_weight_up.weight": f"{hf}layers.3.mlp_hyper_connection.input_mix_weight_up.weight",
            "decoder.layers.3.self_attention.indexer.index_qk_proj.weight": f"{hf}layers.3.self_attn.indexer.index_qk_proj.weight",
            "decoder.layers.3.self_attention.indexer.k_layernorm.weight": f"{hf}layers.3.self_attn.indexer.k_layernorm.weight",
            "decoder.layers.3.self_attention.q_layernorm.weight": f"{hf}layers.3.self_attn.q_norm.weight",
            "decoder.layers.3.self_attention.linear_proj.weight": f"{hf}layers.3.self_attn.o_proj.weight",
            "decoder.layers.0.self_attention.out_norm.weight": f"{hf}layers.0.linear_attn.norm.weight",
            "decoder.layers.0.self_attention.A_log": f"{hf}layers.0.linear_attn.A_log",
            "decoder.layers.0.mlp.router.weight": f"{hf}layers.0.mlp.gate.weight",
            "decoder.layers.0.mlp.shared_experts.gate_weight": f"{hf}layers.0.mlp.shared_expert_gate.weight",
            "decoder.layers.1.per_layer_embedding.key_proj.weight": f"{hf}layers.1.ple.key_proj.weight",
            "decoder.layers.1.per_layer_embedding.conv1d.weight": f"{hf}layers.1.ple.conv1d.weight",
        }
        for megatron_name, hf_name in expectations.items():
            mapping = registry.megatron_to_hf_lookup(megatron_name)
            assert mapping is not None, megatron_name
            assert mapping.hf_param == hf_name, (megatron_name, mapping.hf_param)

        qkv = registry.megatron_to_hf_lookup("decoder.layers.3.self_attention.linear_qkv.weight")
        assert qkv.hf_param == {
            "q": f"{hf}layers.3.self_attn.q_proj.weight",
            "k": f"{hf}layers.3.self_attn.k_proj.weight",
            "v": f"{hf}layers.3.self_attn.v_proj.weight",
        }
        in_proj = registry.megatron_to_hf_lookup("decoder.layers.0.self_attention.in_proj.weight")
        assert set(in_proj.hf_param) == {"qkv", "z", "b", "a"}
        experts = registry.megatron_to_hf_lookup("decoder.layers.0.mlp.experts.linear_fc1.weight5")
        assert experts.hf_param == f"{hf}layers.0.mlp.experts.gate_up_proj"

        ple = registry.megatron_to_hf_lookup(
            "decoder.layers.1.per_layer_embedding.ple_embedding.ngram_embedding.weight"
        )
        assert isinstance(ple, PLENGramEmbeddingMapping)
        assert ple.num_shards == 128
        assert ple.hf_param["shard_0"] == f"{hf}layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
        assert ple.hf_param["shard_127"].endswith("shard_127.weight")

    def test_num_ple_shards_from_checkpoint_keys(self, mock_pretrained, vl_config):
        mock_pretrained.state.source.get_all_keys.return_value = [
            f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}.weight" for i in range(16)
        ]
        bridge = Qwen4ExpBridge()
        bridge.hf_pretrained = mock_pretrained
        assert bridge._num_ple_shards(vl_config.text_config) == 16


class TestPLEShardLoading:
    @pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
    def test_local_rows_cover_rank_range(self, tp_size):
        num_shards, shard_rows, dim = 16, 6, 3
        table = torch.arange(num_shards * shard_rows * dim, dtype=torch.float32).view(-1, dim)
        loaded = []

        def load_shard(i):
            loaded.append(i)
            return table[i * shard_rows : (i + 1) * shard_rows]

        rows_per_rank = table.shape[0] // tp_size
        for rank in range(tp_size):
            loaded.clear()
            rows = ple_local_rows_from_shards(load_shard, num_shards, table.shape[0], shard_rows, rank, tp_size)
            torch.testing.assert_close(rows, table[rank * rows_per_rank : (rank + 1) * rows_per_rank])
            # Only the overlapping shards are read.
            assert len(loaded) <= -(-rows_per_rank // shard_rows) + 1

    def test_rejects_indivisible_tables(self):
        with pytest.raises(ValueError):
            ple_local_rows_from_shards(lambda i: torch.zeros(5, 2), 3, 15, 5, 0, 2)

    def test_export_resplits_shards(self):
        mapping = PLENGramEmbeddingMapping("x.weight", "hf.shard_{}.weight", num_shards=4)
        mapping.broadcast_from_pp_rank = lambda t, cache_key=None: t
        full = torch.arange(24.0).view(12, 2)
        out = mapping.megatron_to_hf(full, None)
        assert list(out) == [f"hf.shard_{i}.weight" for i in range(4)]
        torch.testing.assert_close(torch.cat(list(out.values()), dim=0), full)

    @pytest.mark.parametrize("tp_size,num_shards", [(2, 4), (4, 4), (4, 6), (3, 5)])
    def test_export_streams_one_shard_at_a_time_across_tp(self, monkeypatch, tp_size, num_shards):
        """Each shard is assembled from the ranks' row slices (one collective per shard)."""
        dim = 3
        rows_per_rank = 12
        full = torch.arange(float(tp_size * rows_per_rank * dim)).view(-1, dim)
        shard_rows = -(-full.shape[0] // num_shards)

        # Stand in for the TP group: one mapping per rank (the TP coordinates are class
        # properties), run every rank's contribution for the same shard and sum them, which is
        # what the all_reduce does.
        mappings = []
        for rank in range(tp_size):
            rank_cls = type(
                f"RankMapping{rank}",
                (PLENGramEmbeddingMapping,),
                {
                    "tp_size": property(lambda self, n=tp_size: n),
                    "tp_rank": property(lambda self, r=rank: r),
                    "tp_group": property(lambda self: None),
                },
            )
            mappings.append(rank_cls("x.weight", "hf.shard_{}.weight", num_shards=num_shards))
        contributions = []
        monkeypatch.setattr(torch.distributed, "all_reduce", lambda t, group=None: contributions.append(t.clone()))

        for i in range(num_shards):
            contributions.clear()
            for rank, m in enumerate(mappings):
                local = full[rank * rows_per_rank : (rank + 1) * rows_per_rank]
                m.gather_shard(local, i)
            assert len(contributions) == tp_size
            shard = torch.stack(contributions).sum(0)
            torch.testing.assert_close(shard, full[i * shard_rows : (i + 1) * shard_rows])

        # The export mapping is lazy: building and listing it gathers nothing.
        contributions.clear()
        mappings[0].broadcast_from_pp_rank = lambda t, cache_key=None: t
        out = mappings[0].megatron_to_hf(full[:rows_per_rank], None)
        assert len(out) == num_shards and list(out) == [f"hf.shard_{i}.weight" for i in range(num_shards)]
        assert contributions == []
