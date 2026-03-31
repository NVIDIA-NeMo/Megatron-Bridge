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

import torch
from megatron.core.models.mamba import MambaModel
from transformers import Qwen3NextForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (  # noqa: F401
    AutoMapping,
    GatedMLPMapping,
    GDNConv1dMapping,
    GDNLinearMapping,
    QKVMapping,
    ReplicatedMapping,
    RMSNorm2ZeroCenteredRMSNormMapping,
)
from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider


@MegatronModelBridge.register_bridge(
    source=Qwen3NextForCausalLM, target=MambaModel, provider=MambaModelProvider, model_type="qwen3_next"
)
class Qwen3NextBridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen3-Next Causal LM.

    This bridge handles the conversion between HuggingFace Qwen3NextForCausalLM
    and Megatron-Core MambaModel formats. Qwen3-Next uses a hybrid architecture
    combining gated delta net linear attention with standard softmax attention,
    mixture of experts with shared experts, and zero-centered RMSNorm.

    In MambaModel, attention and MLP are separate physical layers. Each HuggingFace
    logical layer N maps to two physical layers:
      - Physical layer 2*N:   attention (GDN or standard softmax)
      - Physical layer 2*N+1: MoE FFN

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained):
        """Convert HuggingFace Qwen3-Next config to MambaModelProvider."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # Architecture
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.position_embedding_type = "rope"
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.hidden_dropout = 0.0
        provider.qk_layernorm = True
        provider.autocast_dtype = torch.bfloat16
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)

        # MoE settings
        provider.moe_grouped_gemm = True
        provider.moe_router_load_balancing_type = "global_aux_loss"
        provider.moe_aux_loss_coeff = 1e-3
        provider.moe_router_pre_softmax = False
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True
        provider.moe_shared_expert_gate = True
        provider.moe_router_dtype = "fp32"
        provider.moe_shared_expert_intermediate_size = hf_config.shared_expert_intermediate_size

        # Qwen3-Next specific
        provider.layernorm_zero_centered_gamma = True
        provider.attention_output_gate = True

        # GDN linear attention parameters
        provider.linear_conv_kernel_dim = hf_config.linear_conv_kernel_dim
        provider.linear_key_head_dim = hf_config.linear_key_head_dim
        provider.linear_value_head_dim = hf_config.linear_value_head_dim
        provider.linear_num_key_heads = hf_config.linear_num_key_heads
        provider.linear_num_value_heads = hf_config.linear_num_value_heads

        # Build hybrid_override_pattern from HF config.
        # Each HF layer becomes 2 physical layers: attention (G or *) + MLP (E).
        num_hf_layers = hf_config.num_hidden_layers
        full_attn_interval = hf_config.full_attention_interval
        pattern_chars = []
        for n in range(num_hf_layers):
            pattern_chars.append("*" if (n + 1) % full_attn_interval == 0 else "G")
            pattern_chars.append("E")
        main_pattern = "".join(pattern_chars)

        provider.hybrid_override_pattern = main_pattern
        provider.num_layers = num_hf_layers * 2

        # MTP: Qwen3-Next MTP uses standard attention (not GDN), pattern "*E".
        # The HF config doesn't expose num_mtp_layers, so we detect from the
        # safetensors index (no model load / GPU memory needed).
        if self._hf_model_has_mtp(hf_pretrained):
            provider.mtp_num_layers = 1
            provider.mtp_hybrid_override_pattern = "*E"

        # Heterogeneous checkpointing for mixed attention layers
        provider.hetereogenous_dist_checkpoint = True

        return provider

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hf_model_has_mtp(hf_pretrained) -> bool:
        """Detect MTP by checking the safetensors index without loading model weights."""
        import json
        from pathlib import Path

        from huggingface_hub import hf_hub_download

        model_id = getattr(hf_pretrained, 'model_name_or_path', None)
        if model_id is None:
            config = getattr(hf_pretrained, 'config', hf_pretrained)
            model_id = getattr(config, 'name_or_path', None)
        if not model_id:
            return False

        # Try local path first, then download from hub
        try:
            local_path = Path(str(model_id)) / "model.safetensors.index.json"
            if local_path.exists():
                index_path = local_path
            else:
                index_path = Path(hf_hub_download(str(model_id), "model.safetensors.index.json"))

            with open(index_path) as f:
                weight_map = json.load(f).get("weight_map", {})
            return any(k.startswith("mtp.") for k in weight_map)
        except Exception:
            return False

    @staticmethod
    def _gdn_attention_mappings(p, h):
        """GDN (Gated Delta Net) linear attention mappings.

        Args:
            p: Megatron layer prefix (e.g. "decoder.layers.4").
            h: HuggingFace layer prefix (e.g. "model.layers.2").
        """
        return [
            AutoMapping(f"{p}.self_attention.in_proj.layer_norm_weight", f"{h}.input_layernorm.weight"),
            GDNLinearMapping(
                f"{p}.self_attention.in_proj.weight",
                qkvz=f"{h}.linear_attn.in_proj_qkvz.weight",
                ba=f"{h}.linear_attn.in_proj_ba.weight",
            ),
            GDNConv1dMapping(f"{p}.self_attention.conv1d.weight", f"{h}.linear_attn.conv1d.weight"),
            AutoMapping(f"{p}.self_attention.out_proj.weight", f"{h}.linear_attn.out_proj.weight"),
            AutoMapping(f"{p}.self_attention.A_log", f"{h}.linear_attn.A_log"),
            AutoMapping(f"{p}.self_attention.dt_bias", f"{h}.linear_attn.dt_bias"),
            RMSNorm2ZeroCenteredRMSNormMapping(
                f"{p}.self_attention.out_norm.weight", f"{h}.linear_attn.norm.weight"
            ),
        ]

    @staticmethod
    def _standard_attention_mappings(p, h):
        """Standard softmax attention mappings.

        Args:
            p: Megatron layer prefix (e.g. "decoder.layers.6").
            h: HuggingFace layer prefix (e.g. "model.layers.3").
        """
        return [
            AutoMapping(f"{p}.self_attention.linear_qkv.layer_norm_weight", f"{h}.input_layernorm.weight"),
            QKVMapping(
                f"{p}.self_attention.linear_qkv.weight",
                q=f"{h}.self_attn.q_proj.weight",
                k=f"{h}.self_attn.k_proj.weight",
                v=f"{h}.self_attn.v_proj.weight",
            ),
            AutoMapping(f"{p}.self_attention.linear_proj.weight", f"{h}.self_attn.o_proj.weight"),
            AutoMapping(f"{p}.self_attention.q_layernorm.weight", f"{h}.self_attn.q_norm.weight"),
            AutoMapping(f"{p}.self_attention.k_layernorm.weight", f"{h}.self_attn.k_norm.weight"),
        ]

    @staticmethod
    def _moe_mappings(p, h):
        """MoE FFN mappings (router, routed experts, shared expert, gate).

        Args:
            p: Megatron layer prefix (e.g. "decoder.layers.5").
            h: HuggingFace layer prefix (e.g. "model.layers.2").
        """
        return [
            AutoMapping(f"{p}.mlp.router.weight", f"{h}.mlp.gate.weight"),
            AutoMapping(f"{p}.pre_mlp_layernorm.weight", f"{h}.post_attention_layernorm.weight"),
            GatedMLPMapping(
                f"{p}.mlp.experts.linear_fc1.weight*",
                gate=f"{h}.mlp.experts.*.gate_proj.weight",
                up=f"{h}.mlp.experts.*.up_proj.weight",
            ),
            AutoMapping(
                megatron_param=f"{p}.mlp.experts.linear_fc2.weight*",
                hf_param=f"{h}.mlp.experts.*.down_proj.weight",
            ),
            GatedMLPMapping(
                f"{p}.mlp.shared_experts.linear_fc1.weight",
                gate=f"{h}.mlp.shared_expert.gate_proj.weight",
                up=f"{h}.mlp.shared_expert.up_proj.weight",
            ),
            AutoMapping(
                megatron_param=f"{p}.mlp.shared_experts.linear_fc2.weight",
                hf_param=f"{h}.mlp.shared_expert.down_proj.weight",
            ),
            ReplicatedMapping(
                megatron_param=f"{p}.mlp.shared_experts.gate_weight",
                hf_param=f"{h}.mlp.shared_expert_gate.weight",
            ),
        ]

    # ------------------------------------------------------------------
    # Mapping registry
    # ------------------------------------------------------------------

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Build parameter mappings with MambaModel physical layer indexing.

        HuggingFace logical layer N maps to:
          - Physical layer 2*N:   attention (GDN or standard)
          - Physical layer 2*N+1: MoE FFN
        """
        num_hf_layers = self.hf_config.num_hidden_layers
        full_attn_interval = self.hf_config.full_attention_interval

        mapping_list = [
            AutoMapping("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
            AutoMapping("output_layer.weight", "lm_head.weight"),
            AutoMapping("decoder.final_norm.weight", "model.norm.weight"),
        ]

        AutoMapping.register_module_type("SharedExpertMLP", "column")
        AutoMapping.register_module_type("GatedDeltaNet", "column")

        # ---- Per-layer mappings ----
        for n in range(num_hf_layers):
            attn_phys = 2 * n       # physical attention layer
            mlp_phys = 2 * n + 1    # physical MLP layer
            is_gdn = (n + 1) % full_attn_interval != 0

            p_attn = f"decoder.layers.{attn_phys}"
            p_mlp = f"decoder.layers.{mlp_phys}"
            h = f"model.layers.{n}"

            if is_gdn:
                mapping_list.extend(self._gdn_attention_mappings(p_attn, h))
            else:
                mapping_list.extend(self._standard_attention_mappings(p_attn, h))

            mapping_list.extend(self._moe_mappings(p_mlp, h))

        # ---- MTP mappings ----
        # MTP inner MambaStack: standard attention at physical layer 0, MoE at physical layer 1.
        # In MambaModel MTP, the inner stack is stored as "mtp_model_layer".
        mtp = "mtp.layers.0"
        mtp_inner = f"{mtp}.mtp_model_layer.layers"

        mapping_list.extend([
            AutoMapping(f"{mtp}.eh_proj.weight", "mtp.fc.weight"),
            AutoMapping(f"{mtp}.enorm.weight", "mtp.pre_fc_norm_embedding.weight"),
            AutoMapping(f"{mtp}.hnorm.weight", "mtp.pre_fc_norm_hidden.weight"),
            AutoMapping(f"{mtp}.final_layernorm.weight", "mtp.norm.weight"),
        ])
        mapping_list.extend(self._standard_attention_mappings(f"{mtp_inner}.0", mtp))
        mapping_list.extend(self._moe_mappings(f"{mtp_inner}.1", mtp))

        return MegatronMappingRegistry(*mapping_list)
