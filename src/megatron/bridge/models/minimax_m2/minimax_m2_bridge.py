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

from collections.abc import Mapping
from typing import Dict, Optional

import torch
import torch.nn as nn
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    QKVMapping,
)
from megatron.bridge.models.minimax_m2.minimax_m2_provider import minimax_m2_layer_spec


_FP8_BLOCK_SIZE = 128


def _dequant_fp8_blockwise(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Block-wise FP8 dequantization: out = fp8_val * scale_inv per 128x128 block."""
    M, N = weight.shape
    B = _FP8_BLOCK_SIZE
    w = weight.float()
    out = torch.empty_like(w)
    sM, sN = scale_inv.shape
    for bi in range(sM):
        for bj in range(sN):
            r0, r1 = bi * B, min((bi + 1) * B, M)
            c0, c1 = bj * B, min((bj + 1) * B, N)
            out[r0:r1, c0:c1] = w[r0:r1, c0:c1] * scale_inv[bi, bj]
    return out.to(torch.bfloat16)


class _FullDimQKNormMapping(MegatronParamMapping[torch.Tensor]):
    """TP-sharded mapping for full-dimension QK norm weights.

    HF weight shape: ``[num_heads * head_dim]``
    Megatron weight shape per rank: ``[num_heads_per_partition * head_dim]``

    Uses broadcast-then-slice instead of ``scatter_to_tp_ranks`` because the
    ``_FullDimRMSNorm`` module may reside on CPU / meta device where NCCL
    scatter is not available.
    """

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        target_param = megatron_module.weight
        shard_size = target_param.shape[0]

        if self.tp_size == 1:
            return hf_weights.to(device=target_param.device, dtype=target_param.dtype)

        device = torch.device("cuda", torch.cuda.current_device())
        hf_weights = hf_weights.to(device=device, dtype=target_param.dtype)

        if self.tp_rank > 0:
            hf_weights = torch.empty_like(hf_weights)

        full_weight = self.broadcast_tensor_to_tp_ranks(hf_weights, src_rank=0)
        start = self.tp_rank * shard_size
        return full_weight[start : start + shard_size]

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights, cache_key=str(self.hf_param))
        if megatron_weights is None:
            return {}
        megatron_weights = self.maybe_dequantize(megatron_weights)
        if self.tp_size == 1:
            return {str(self.hf_param): megatron_weights}
        gathered = self.gather_from_tp_ranks(megatron_weights)
        return {str(self.hf_param): torch.cat(gathered, dim=0)}


@MegatronModelBridge.register_bridge(
    source="MiniMaxM2ForCausalLM",
    target=GPTModel,
    model_type="minimax_m2",
)
class MiniMaxM2Bridge(MegatronModelBridge):
    """
    Megatron Bridge for MiniMax-M2 MoE Causal LM.

    MiniMax-M2 is a sparse MoE model (256 experts, top-8 routing with sigmoid
    scoring and expert bias correction). Two checkpoint layouts are supported
    and auto-detected at load time:

    - **Native transformers** (transformers >= 5.0, toy models): ``mlp`` prefix
      with stacked ``gate_up_proj`` / ``down_proj`` tensors.
    - **Legacy HF checkpoint** (``MiniMaxAI/MiniMax-M2`` with
      ``trust_remote_code=True``): ``block_sparse_moe`` prefix with per-expert
      ``w1`` / ``w3`` (gate/up) and ``w2`` (down) weights.

    QK normalization:
        MiniMax-M2 applies full-dimension RMSNorm to Q/K (weight shape =
        num_heads * head_dim) before splitting into heads. Megatron's built-in
        QK norm is per-head (weight shape = head_dim). This bridge uses a custom
        layer spec (``minimax_m2_layer_spec``) with ``FullDimQNorm``/``FullDimKNorm``
        that normalizes over the full partition dimension. With TP > 1 the
        sum-of-squares is all-reduced across TP ranks so the RMS denominator
        matches the single-GPU case.

    Known limitations:
        - MTP (Multi-Token Prediction) modules are not mapped.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("MiniMaxAI/MiniMax-M2")
        >>> provider = bridge.to_megatron_provider()
    """

    def build_conversion_tasks(self, hf_pretrained, megatron_model):
        """Store HF state keys before mapping_registry is called (for format detection)."""
        self._hf_state_source = hf_pretrained.state.source
        self._hf_keys = list(self._hf_state_source.get_all_keys())
        return super().build_conversion_tasks(hf_pretrained, megatron_model)

    def _uses_native_transformers_format(self) -> bool:
        """Return True if the HF checkpoint uses the native transformers layout.

        The native transformers layout (transformers >= 5.0) uses:
          - ``mlp`` attribute name for the MoE block
          - stacked ``mlp.experts.gate_up_proj`` / ``mlp.experts.down_proj``

        The original HuggingFace checkpoint (``MiniMaxAI/MiniMax-M2``) uses the
        pre-merge custom code layout:
          - ``block_sparse_moe`` attribute name
          - per-expert ``block_sparse_moe.experts.*.w1/w3/w2.weight``

        Detection falls back to native format (True) when no HF keys are available.
        """
        hf_keys = getattr(self, "_hf_keys", None)
        if hf_keys:
            return any("mlp.experts.gate_up_proj" in k for k in hf_keys) or not any(
                "block_sparse_moe" in k for k in hf_keys
            )
        return True

    def provider_bridge(self, hf_pretrained):
        """Convert HuggingFace MiniMax-M2 config to GPTModelProvider."""
        provider = super().provider_bridge(hf_pretrained)

        hf_config = hf_pretrained.config

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.position_embedding_type = "rope"
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.hidden_dropout = 0.0
        provider.autocast_dtype = torch.bfloat16

        # MiniMax-M2 uses rotary_dim instead of partial_rotary_factor
        rotary_dim = getattr(hf_config, "rotary_dim", None)
        head_dim = getattr(hf_config, "head_dim", None)
        if rotary_dim is not None and head_dim is not None:
            provider.rotary_percent = rotary_dim / head_dim

        # Full-dimension QK norm via custom layer spec (see minimax_m2_provider.py).
        # qk_layernorm stays False to avoid the default per-head TENorm; our custom
        # spec injects FullDimQNorm/FullDimKNorm directly into SelfAttention.
        provider.qk_layernorm = False
        provider.transformer_layer_spec = minimax_m2_layer_spec

        # MoE settings — sigmoid routing with expert bias (same pattern as DeepSeek V3)
        provider.moe_grouped_gemm = True
        provider.moe_router_pre_softmax = False
        provider.moe_router_load_balancing_type = "aux_loss"
        provider.moe_aux_loss_coeff = 1e-3
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_enable_expert_bias = True

        return provider

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Load HF weights with FP8 block-wise dequantization when needed.

        MiniMax-M2 stores linear weights as float8_e4m3fn with per-block scale
        factors in ``<key>_scale_inv`` tensors (128x128 blocks).
        """
        if isinstance(hf_param, dict):
            return {k: self._load_and_dequant(v, hf_state_dict) for k, v in hf_param.items()}
        return self._load_and_dequant(hf_param, hf_state_dict)

    def _load_and_dequant(self, key: str, hf_state_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        w = hf_state_dict[key]
        if w.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            return w
        sinv_key = key + "_scale_inv"
        if w.ndim == 2 and sinv_key in hf_state_dict:
            return _dequant_fp8_blockwise(w, hf_state_dict[sinv_key])
        return w.float().to(torch.bfloat16)

    def mapping_registry(self) -> MegatronMappingRegistry:
        native = self._uses_native_transformers_format()
        # Two checkpoint layouts exist for MiniMax-M2:
        #   native (transformers >= 5.0): "mlp" prefix, stacked gate_up_proj/down_proj
        #   legacy (MiniMaxAI/MiniMax-M2 HF repo, trust_remote_code=True): "block_sparse_moe"
        #     prefix, per-expert w1/w3/w2 weights
        moe_prefix = "mlp" if native else "block_sparse_moe"

        param_mappings = {
            # Global weights
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            # Per-layer layernorms (TE backend)
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            # Attention o_proj
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            # MoE router and expert bias
            "decoder.layers.*.mlp.router.weight": f"model.layers.*.{moe_prefix}.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": f"model.layers.*.{moe_prefix}.e_score_correction_bias",
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # QK norm — FullDimQNorm/FullDimKNorm weight is [num_heads_per_partition * head_dim],
        # which is a TP shard of the HF [num_heads * head_dim] weight.
        mapping_list.append(
            _FullDimQKNormMapping(
                megatron_param="decoder.layers.*.self_attention.q_layernorm.weight",
                hf_param="model.layers.*.self_attn.q_norm.weight",
            )
        )
        mapping_list.append(
            _FullDimQKNormMapping(
                megatron_param="decoder.layers.*.self_attention.k_layernorm.weight",
                hf_param="model.layers.*.self_attn.k_norm.weight",
            )
        )

        # QKV
        mapping_list.append(
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            )
        )

        # MoE expert weights — two formats depending on the checkpoint:
        if native:
            # Native transformers layout: stacked tensors per layer
            #   gate_up_proj: [num_experts, 2*intermediate, hidden]
            #   down_proj:    [num_experts, hidden, intermediate]
            mapping_list.extend(
                [
                    FusedGatedExpertMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                        hf_param="model.layers.*.mlp.experts.gate_up_proj",
                    ),
                    FusedExpertMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                        hf_param="model.layers.*.mlp.experts.down_proj",
                    ),
                ]
            )
        else:
            # Legacy HF checkpoint layout: per-expert w1/w3 (gate/up) and w2 (down)
            mapping_list.extend(
                [
                    GatedMLPMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                        gate="model.layers.*.block_sparse_moe.experts.*.w1.weight",
                        up="model.layers.*.block_sparse_moe.experts.*.w3.weight",
                    ),
                    AutoMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                        hf_param="model.layers.*.block_sparse_moe.experts.*.w2.weight",
                    ),
                ]
            )

        return MegatronMappingRegistry(*mapping_list)
