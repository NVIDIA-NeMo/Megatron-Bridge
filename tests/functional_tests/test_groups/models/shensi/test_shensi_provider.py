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
    # MLA shares one compressed KV latent across heads, so query groups follow the head count;
    # the HF config's ``num_key_value_heads=1`` shows up as ``kv_lora_rank`` instead.
    assert provider.num_query_groups == cfg.num_attention_heads
    assert provider.vocab_size == cfg.vocab_size
    assert provider.seq_length == cfg.max_position_embeddings
    assert provider.layernorm_epsilon == cfg.rms_norm_eps
    assert provider.init_method_std == cfg.initializer_range
    assert provider.rotary_base == cfg.rope_theta
    # MoE / MLA / shensi-specific fields
    assert provider.num_moe_experts == cfg.n_routed_experts
    assert provider.moe_router_topk == cfg.num_experts_per_tok
    assert provider.moe_router_score_function == cfg.scoring_func
    assert provider.moe_router_topk_scaling_factor == cfg.routed_scaling_factor
    assert provider.moe_grouped_gemm is True
    assert provider.multi_latent_attention is True
    assert provider.q_lora_rank == cfg.q_lora_rank
    assert provider.v_head_dim == cfg.head_dim
    # Shensi-specific fields (mHC streams, attention residuals, sparse-attention indexer) are
    assert provider.experimental_attention_variant == "dsv4_hybrid"
    assert provider.enable_hyper_connections is True
    assert provider.num_residual_streams == cfg.hc_mult
    assert provider.hc_active_streams == cfg.hc_active_streams
    assert provider.hc_fixed_streams == cfg.hc_fixed_streams
    assert list(provider.hc_conv_kernels) == list(cfg.hc_conv_kernels)
    assert provider.attn_res_block_size == cfg.attn_res_block_size
    assert provider.csa_window_size == cfg.sliding_window
    assert provider.csa_compress_rotary_base == cfg.compress_rope_theta
    assert provider.dsa_indexer_n_heads == cfg.index_n_heads
    assert provider.dsa_indexer_head_dim == cfg.index_head_dim
    assert provider.dsa_indexer_topk == cfg.index_topk
    assert provider.moe_latent_size == cfg.routed_expert_hidden_size
    assert provider.moe_ffn_hidden_size == cfg.moe_intermediate_size


def test_provider_layer_layout_matches_config():
    cfg = make_tiny_config()
    provider = AutoBridge.from_hf_config(cfg).to_megatron_provider(load_weights=False)

    assert list(provider.csa_compress_ratios) == [
        cfg.compress_rates[layer_type] for layer_type in TINY_SHENSI_CONFIG["layer_types"]
    ]
    assert provider.moe_layer_freq == [1] * cfg.num_hidden_layers
    assert provider.activation_func_clamp_value == cfg.swiglu_limit


@pytest.mark.parametrize("mtp_layers", [0, 1])
def test_mtp_configuration_follows_hf(mtp_layers):
    cfg_dict = dict(TINY_SHENSI_CONFIG)
    cfg_dict["num_nextn_predict_layers"] = mtp_layers
    cfg = type(make_tiny_config())(**cfg_dict)

    provider = AutoBridge.from_hf_config(cfg).to_megatron_provider(load_weights=False)
    assert (provider.mtp_num_layers or 0) == mtp_layers
