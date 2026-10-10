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

"""Provider coverage for Shensi: the HF config must land on the matching Megatron fields."""

import pytest

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.shensi import ShensiModelProvider

from . import TINY_SHENSI_CONFIG, make_tiny_config


def test_provider_maps_hf_config_fields():
    cfg = make_tiny_config()
    provider = AutoBridge.from_hf_config(cfg).to_megatron_provider(load_weights=False)

    assert isinstance(provider, ShensiModelProvider)
    assert provider.num_layers == cfg.num_hidden_layers
    assert provider.hidden_size == cfg.hidden_size
    assert provider.num_attention_heads == cfg.num_attention_heads
    assert provider.num_query_groups == 1
    assert provider.vocab_size == cfg.vocab_size
    assert provider.seq_length == cfg.max_position_embeddings
    assert provider.layernorm_epsilon == cfg.rms_norm_eps
    assert provider.init_method_std == cfg.initializer_range

    # MLA-style indexed attention.
    assert provider.multi_latent_attention is True
    assert provider.q_lora_rank == cfg.q_lora_rank
    assert provider.v_head_dim == cfg.v_head_dim
    assert provider.kv_lora_rank == cfg.deepemb_kv_rank
    assert provider.qk_pos_emb_head_dim == 0

    # HISA indexer.
    assert provider.index_n_heads == cfg.index_n_heads
    assert provider.index_head_dim == cfg.index_head_dim
    assert provider.index_topk == cfg.index_topk
    assert provider.index_block_size == cfg.index_block_size
    assert provider.index_topk_blocks == cfg.index_topk_blocks
    assert provider.index_topk_freq == cfg.index_topk_freq
    assert provider.index_latent_rank == cfg.index_latent_rank

    # KDA linear attention.
    assert provider.linear_num_key_heads == cfg.linear_num_heads
    assert provider.linear_num_value_heads == cfg.linear_num_heads
    assert provider.linear_key_head_dim == cfg.linear_head_dim
    assert provider.linear_conv_kernel_dim == cfg.linear_conv_kernel_dim
    assert provider.kda_two_stage_gates is True

    # MoE / hyper-connections / attention residuals.
    assert provider.num_moe_experts == cfg.n_routed_experts
    assert provider.moe_router_topk == cfg.num_experts_per_tok
    assert provider.moe_router_score_function == cfg.scoring_func
    assert provider.moe_router_topk_scaling_factor == cfg.routed_scaling_factor
    assert provider.moe_latent_size == cfg.latent_size
    assert provider.moe_ffn_hidden_size == cfg.moe_intermediate_size
    assert provider.moe_aux_loss_coeff == cfg.router_aux_loss_coef
    # HF only applies the router load-balancing loss when router logits are requested.
    assert provider.moe_router_load_balancing_type == "none"
    assert provider.hc_mult == cfg.hc_mult
    assert provider.hc_active_streams == cfg.hc_active_streams
    assert provider.hc_adapter_rank == cfg.hc_adapter_rank
    assert provider.attn_res_rank == cfg.attn_res_rank
    assert provider.attn_res_block_size == cfg.attn_res_block_size
    assert provider.attn_res_block_summary_rank == cfg.attn_res_block_summary_rank


def test_provider_layer_layout_matches_config():
    cfg = make_tiny_config()
    provider = AutoBridge.from_hf_config(cfg).to_megatron_provider(load_weights=False)

    assert list(provider.mlp_layer_types) == list(cfg.mlp_layer_types)
    assert list(provider.layer_types) == [
        "indexed_attention" if block == "block_write_layer" else "linear_attention"
        for block in cfg.attn_res_block_layer_types
    ]
    assert list(provider.indexer_types) == list(cfg.indexer_types)
    assert provider.activation_func_clamp_value == cfg.swiglu_limit


@pytest.mark.parametrize("mtp_layers", [0, 1])
def test_mtp_configuration_follows_hf(mtp_layers):
    cfg_dict = dict(TINY_SHENSI_CONFIG)
    cfg_dict["num_mtp_layers"] = mtp_layers
    cfg = type(make_tiny_config())(**cfg_dict)

    provider = AutoBridge.from_hf_config(cfg).to_megatron_provider(load_weights=False)
    assert (provider.mtp_num_layers or 0) == mtp_layers


def test_output_router_logits_enables_the_aux_loss():
    """HF's `output_router_logits` is the switch that adds the router aux loss to training."""
    cfg_dict = dict(TINY_SHENSI_CONFIG)
    cfg_dict["output_router_logits"] = True
    cfg = type(make_tiny_config())(**cfg_dict)

    provider = AutoBridge.from_hf_config(cfg).to_megatron_provider(load_weights=False)
    assert provider.moe_router_load_balancing_type == "aux_loss"
    assert provider.moe_aux_loss_coeff == cfg.router_aux_loss_coef


def test_tensor_parallel_configuration_is_accepted():
    """TP shards the attention heads and dense/MoE linears and enables the required SP path."""
    cfg = make_tiny_config()
    provider = AutoBridge.from_hf_config(cfg).to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = 2
    provider.finalize()
    assert provider.tensor_model_parallel_size == 2
    assert provider.sequence_parallel is True
    assert cfg.num_attention_heads % provider.tensor_model_parallel_size == 0


def test_pipeline_parallel_configuration_sets_variable_seq_lengths():
    """The stage payload carries a per-microbatch stream state, so PP needs variable lengths."""
    cfg = make_tiny_config()
    provider = AutoBridge.from_hf_config(cfg).to_megatron_provider(load_weights=False)
    provider.pipeline_model_parallel_size = 2
    provider.finalize()
    assert provider.variable_seq_lengths is True
    assert provider.pipeline_model_parallel_size == 2
    assert provider.moe_token_dispatcher_type == "alltoall"
    assert provider.hybrid_layer_pattern.count("|") == 1


def test_context_parallel_is_rejected():
    cfg = make_tiny_config()
    provider = AutoBridge.from_hf_config(cfg).to_megatron_provider(load_weights=False)
    provider.context_parallel_size = 2
    with pytest.raises(NotImplementedError, match="context parallelism"):
        provider.finalize()


def test_tensor_parallel_head_divisibility_is_validated():
    cfg = make_tiny_config()
    provider = AutoBridge.from_hf_config(cfg).to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = 3
    with pytest.raises(ValueError, match="num_attention_heads"):
        provider.finalize()
