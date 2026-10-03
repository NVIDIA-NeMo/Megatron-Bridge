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

"""Bridge between the Shensi HuggingFace model and its Megatron implementation."""

from typing import Dict

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    DirectMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.shensi.configuration_shensi import (
    resolve_csa_compress_ratios,
    resolve_moe_n_hash_layers,
)
from megatron.bridge.models.shensi.modeling_shensi import ShensiModel
from megatron.bridge.models.shensi.shensi_provider import (
    SHENSI_HARD_DEFAULTS,
    ShensiModelProvider,
    shensi_derived_field_values,
)


def _layer_mappings(
    megatron_prefix: str,
    hf_prefix: str,
    *,
    compress_ratio: int,
    is_hash: bool,
    conv_kernels: tuple,
) -> list:
    """Mappings of one decoder-shaped layer.

    The decoder stack and every MTP depth carry the same submodules, so both go through here
    with their own prefixes: an MTP depth nests a decoder layer inside the MTP wrapper and
    follows the MLP type of the last decoder layer.
    """
    fp, hp = megatron_prefix, hf_prefix
    out = []
    out += [
        AutoMapping(fp + "input_layernorm.weight", hp + "input_layernorm.weight"),
        AutoMapping(fp + "pre_mlp_layernorm.weight", hp + "post_attention_layernorm.weight"),
    ]
    for hc in ("attn_hc", "ffn_hc"):
        for suffix in (
            "pre_fn",
            "pre_base",
            "pre_scale",
            "route_norm.weight",
            "route_norm.bias",
            "route_fn",
            "route_base",
            "route_scale",
            "post_fn",
            "post_base",
            "post_scale",
        ):
            out.append(ReplicatedMapping(fp + f"{hc}.{suffix}", hp + f"{hc}.{suffix}"))
    for k in range(len(conv_kernels)):
        out.append(
            ReplicatedMapping(
                fp + f"ffn_hc.temporal_convs.{k}.weight",
                hp + f"ffn_hc.temporal_convs.{k}.weight",
            )
        )
    for slot in ("self_attention_attn_res", "mlp_attn_res"):
        out += [
            DirectMapping(fp + f"{slot}.g_scale", hp + f"{slot}.g_scale"),
            DirectMapping(fp + f"{slot}.t", hp + f"{slot}.t"),
            DirectMapping(fp + f"{slot}.q_a_proj.weight", hp + f"{slot}.q_a_proj.weight"),
            DirectMapping(fp + f"{slot}.q_b_proj.weight", hp + f"{slot}.q_b_proj.weight"),
            DirectMapping(fp + f"{slot}.k_a_proj.weight", hp + f"{slot}.k_a_proj.weight"),
            DirectMapping(fp + f"{slot}.k_b_proj.weight", hp + f"{slot}.k_b_proj.weight"),
            DirectMapping(fp + f"{slot}.g_a_proj.weight", hp + f"{slot}.g_a_proj.weight"),
            DirectMapping(fp + f"{slot}.g_a_proj.bias", hp + f"{slot}.g_a_proj.bias"),
            DirectMapping(fp + f"{slot}.g_b_proj.weight", hp + f"{slot}.g_b_proj.weight"),
            DirectMapping(fp + f"{slot}.g_b_proj.bias", hp + f"{slot}.g_b_proj.bias"),
        ]
    out += [
        AutoMapping(fp + "self_attention.linear_q_down_proj.weight", hp + "self_attn.q_a_proj.weight"),
        AutoMapping(fp + "self_attention.q_layernorm.weight", hp + "self_attn.q_a_norm.weight"),
        AutoMapping(fp + "self_attention.linear_q_up_proj.weight", hp + "self_attn.q_b_proj.weight"),
        AutoMapping(fp + "self_attention.linear_kv_proj.weight", hp + "self_attn.kv_proj.weight"),
        AutoMapping(fp + "self_attention.kv_layernorm.weight", hp + "self_attn.kv_norm.weight"),
        ReplicatedMapping(fp + "self_attention.linear_o_group_proj", hp + "self_attn.o_a_proj.weight"),
        AutoMapping(fp + "self_attention.linear_proj.weight", hp + "self_attn.o_b_proj.weight"),
        ReplicatedMapping(fp + "self_attention.core_attention.attn_sink", hp + "self_attn.sinks"),
    ]
    if compress_ratio:
        out += [
            AutoMapping(
                fp + "self_attention.core_attention.compressor.linear_wkv.weight",
                hp + "self_attn.compressor.kv_proj.weight",
            ),
            AutoMapping(
                fp + "self_attention.core_attention.compressor.linear_wgate.weight",
                hp + "self_attn.compressor.gate_proj.weight",
            ),
            ReplicatedMapping(
                fp + "self_attention.core_attention.compressor.ape",
                hp + "self_attn.compressor.position_bias",
            ),
            ReplicatedMapping(
                fp + "self_attention.core_attention.compressor.norm.weight",
                hp + "self_attn.compressor.kv_norm.weight",
            ),
        ]
    if compress_ratio == 4:
        out += [
            AutoMapping(
                fp + "self_attention.core_attention.indexer.linear_wq_b.weight",
                hp + "self_attn.compressor.indexer.q_b_proj.weight",
            ),
            AutoMapping(
                fp + "self_attention.core_attention.indexer.linear_weights_proj.weight",
                hp + "self_attn.compressor.indexer.scorer.weights_proj.weight",
            ),
            AutoMapping(
                fp + "self_attention.core_attention.indexer.compressor.linear_wkv.weight",
                hp + "self_attn.compressor.indexer.kv_proj.weight",
            ),
            AutoMapping(
                fp + "self_attention.core_attention.indexer.compressor.linear_wgate.weight",
                hp + "self_attn.compressor.indexer.gate_proj.weight",
            ),
            ReplicatedMapping(
                fp + "self_attention.core_attention.indexer.compressor.ape",
                hp + "self_attn.compressor.indexer.position_bias",
            ),
            ReplicatedMapping(
                fp + "self_attention.core_attention.indexer.compressor.norm.weight",
                hp + "self_attn.compressor.indexer.kv_norm.weight",
            ),
        ]
    if is_hash:
        out += [
            ReplicatedMapping(fp + "mlp.deepemb.weight", hp + "mlp.deepemb.weight"),
            GatedMLPMapping(
                fp + "mlp.linear_fc1.weight",
                hp + "mlp.gate_proj.weight",
                hp + "mlp.up_proj.weight",
            ),
            AutoMapping(fp + "mlp.linear_fc2.weight", hp + "mlp.down_proj.weight"),
        ]
    else:
        out += [
            ReplicatedMapping(fp + "mlp.router.weight", hp + "mlp.gate.weight"),
            ReplicatedMapping(fp + "mlp.routed_expert_norm.weight", hp + "mlp.routed_expert_norm.weight"),
            AutoMapping(fp + "mlp.fc1_latent_proj.weight", hp + "mlp.routed_expert_down_proj.weight"),
            AutoMapping(fp + "mlp.fc2_latent_proj.weight", hp + "mlp.routed_expert_up_proj.weight"),
        ]
        out += [
            FusedGatedExpertMapping(
                fp + "mlp.experts.linear_fc1.weight*",
                hp + "mlp.experts.gate_up_proj",
            ),
            FusedExpertMapping(
                fp + "mlp.experts.linear_fc2.weight*",
                hp + "mlp.experts.down_proj",
            ),
        ]
    return out


@MegatronModelBridge.register_bridge(
    source="ShensiForCausalLM",
    target=ShensiModel,
    provider=ShensiModelProvider,
    model_type="shensi",
)
class ShensiBridge(MegatronModelBridge):
    """Maps a Shensi HF checkpoint onto a `ShensiModelProvider` and its parameters.

    Three groups of parameters need a mapping of their own:

    * the mHC stream mixers and the attention-residual readers exist once per layer and are
      replicated across tensor-parallel ranks (they act on the full hidden dimension);
    * the CSA/HCA layers carry a compressor (and, for CSA, a Lightning Indexer) whose projections
      exist only on layers whose compress ratio is non-zero;
    * the leading hash-MoE layers have a dense gated MLP plus a token-embedding gate instead of
      routed experts, so they map onto `linear_fc1/linear_fc2` rather than the expert tensors.
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> ShensiModelProvider:
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config
        mtp = int(getattr(hf_config, "num_nextn_predict_layers", 0) or 0)
        for key, value in shensi_derived_field_values(hf_config, mtp_num_layers=mtp).items():
            setattr(provider, key, value)
        for key, value in SHENSI_HARD_DEFAULTS.items():
            setattr(provider, key, value)
        provider.transformer_impl = "transformer_engine"
        provider.gated_linear_unit = True
        provider.moe_grouped_gemm = True
        provider.moe_router_pre_softmax = False
        provider.moe_router_score_function = hf_config.scoring_func
        provider.moe_router_topk = hf_config.num_experts_per_tok
        provider.moe_router_topk_scaling_factor = hf_config.routed_scaling_factor
        provider.moe_token_dispatcher_type = "allgather"
        provider.moe_router_dtype = "fp32"
        provider.moe_layer_freq = [1] * hf_config.num_hidden_layers
        provider.activation_func_clamp_value = hf_config.swiglu_limit
        provider.layernorm_epsilon = hf_config.rms_norm_eps
        provider.init_method_std = hf_config.initializer_range
        provider.attention_dropout = hf_config.attention_dropout
        provider.hidden_dropout = 0.0
        provider.masked_softmax_fusion = False
        provider.persist_layer_norm = False
        provider.hidden_size = hf_config.hidden_size
        provider.num_layers = hf_config.num_hidden_layers
        provider.vocab_size = hf_config.vocab_size
        provider.seq_length = int(getattr(hf_config, "max_position_embeddings", 4096))
        provider.share_embeddings_and_output_weights = bool(getattr(hf_config, "tie_word_embeddings", False))
        provider.rotary_percent = 1.0
        provider.num_query_groups = int(hf_config.num_attention_heads)
        provider.rope_type = "rope"
        provider.original_max_position_embeddings = 4096
        provider.qk_head_dim = provider.v_head_dim - provider.qk_pos_emb_head_dim
        provider.kv_lora_rank = provider.v_head_dim - provider.qk_pos_emb_head_dim
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        hf_config = self.hf_config
        mappings = [
            AutoMapping("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
            AutoMapping("output_layer.weight", "lm_head.weight"),
            AutoMapping("output_norm.weight", "model.norm.weight"),
            ReplicatedMapping("hc_head.hc_fn", "model.hc_head.hc_fn"),
            ReplicatedMapping("hc_head.hc_base", "model.hc_head.hc_base"),
            ReplicatedMapping("hc_head.hc_scale", "model.hc_head.hc_scale"),
            DirectMapping("output_attn_res.g_scale", "model.output_attn_res.g_scale"),
            DirectMapping("output_attn_res.t", "model.output_attn_res.t"),
            DirectMapping("output_attn_res.q_a_proj.weight", "model.output_attn_res.q_a_proj.weight"),
            DirectMapping("output_attn_res.q_b_proj.weight", "model.output_attn_res.q_b_proj.weight"),
            DirectMapping("output_attn_res.k_a_proj.weight", "model.output_attn_res.k_a_proj.weight"),
            DirectMapping("output_attn_res.k_b_proj.weight", "model.output_attn_res.k_b_proj.weight"),
            DirectMapping("output_attn_res.g_a_proj.weight", "model.output_attn_res.g_a_proj.weight"),
            DirectMapping("output_attn_res.g_a_proj.bias", "model.output_attn_res.g_a_proj.bias"),
            DirectMapping("output_attn_res.g_b_proj.weight", "model.output_attn_res.g_b_proj.weight"),
            DirectMapping("output_attn_res.g_b_proj.bias", "model.output_attn_res.g_b_proj.bias"),
        ]
        n_layers = int(getattr(hf_config, "num_hidden_layers", 0))
        n_hash = resolve_moe_n_hash_layers(list(hf_config.mlp_layer_types))
        compress = resolve_csa_compress_ratios(list(hf_config.layer_types))
        conv_kernels = tuple(int(k) for k in getattr(hf_config, "hc_conv_kernels", ()) or ())
        for i in range(n_layers):
            mappings += _layer_mappings(
                f"decoder.layers.{i}.",
                f"model.layers.{i}.",
                compress_ratio=(compress[i] if i < len(compress) else 0),
                is_hash=i < n_hash,
                conv_kernels=conv_kernels,
            )
        # MTP depths: the checkpoint keeps them under ``mtp.{k}`` and mcore stacks them under
        # ``mtp.layers.{k}``. Each depth owns its embed/hidden norms, the fused embedding-hidden
        # projection and its final norm, plus a decoder-shaped layer holding the last decoder
        # layer's submodules.
        for depth_index in range(int(getattr(hf_config, "num_nextn_predict_layers", 0) or 0)):
            ck = f"mtp.{depth_index}"
            mg = f"mtp.layers.{depth_index}"
            mappings += [
                AutoMapping(f"{mg}.enorm.weight", f"{ck}.enorm.weight"),
                AutoMapping(f"{mg}.hnorm.weight", f"{ck}.hnorm.weight"),
                AutoMapping(f"{mg}.eh_proj.weight", f"{ck}.eh_proj.weight"),
                AutoMapping(f"{mg}.final_layernorm.weight", f"{ck}.norm.weight"),
            ]
            mappings += _layer_mappings(
                f"{mg}.mtp_model_layer.",
                f"{ck}.",
                compress_ratio=128,
                is_hash=(n_layers - 1) < n_hash,
                conv_kernels=conv_kernels,
            )
        return MegatronMappingRegistry(*mappings)

    @classmethod
    def megatron_to_hf_config(cls, provider: ShensiModelProvider) -> Dict:
        hf = super().megatron_to_hf_config(provider)
        hf["model_type"] = "shensi"
        hf["architectures"] = ["ShensiForCausalLM"]
        hf["hidden_size"] = provider.hidden_size
        hf["num_hidden_layers"] = provider.num_layers
        # The MTP depths travel as their own checkpoint subtree, so the exported config has to carry
        # their count for the reverse direction to rebuild them.
        hf["num_nextn_predict_layers"] = int(getattr(provider, "mtp_num_layers", 0) or 0)
        hf["hc_mult"] = getattr(provider, "num_residual_streams", 16)
        hf["attn_res_block_size"] = getattr(provider, "attn_res_block_size", 4)
        hf["attn_res_read_heads"] = getattr(provider, "attn_res_read_heads", 8)
        return hf
