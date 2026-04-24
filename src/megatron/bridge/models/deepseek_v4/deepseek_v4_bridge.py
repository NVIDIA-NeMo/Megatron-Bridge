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

"""Megatron Bridge for DeepSeek-V4-Flash.

Follows the standard `@MegatronModelBridge.register_bridge` convention.

Caveats specific to DeepSeek-V4-Flash:

* The published HF checkpoint ships in *inference format* — keys have no
  `model.` prefix and use `attn`/`ffn` instead of `self_attn`/`mlp`. There is
  no `modeling_deepseek.py` in the repo and no `auto_map`. The `AutoBridge`
  dispatch here keys off the class name string `DeepseekV4ForCausalLM` from
  `config.architectures`, which still matches even though the class is not
  importable from `transformers`.

* Block-level Hyper-Connection parameters map to mcore
  `HyperConnectionModule`-compatible names (`hc_attn`, `hc_ffn`) with a fused
  `mapping_proj.weight`, `bias`, and a single `alpha [3]` tensor. Forward code
  that instantiates the real mcore module splits `alpha` into three scalar
  parameters at construction time.

* Head-level HC (`hc_head_fn / base / scale`) has no mcore analog — the
  inference head uses a smaller n-row form. It is carried through as raw
  parameters under `decoder.hc_head_*` and `mtp.layers.*.hc_head_*`.

* V4 introduces **CSA** (full attention: low-rank Q, single KV, grouped HCA O,
  optional Compressor + Indexer) under `self_attention.*` and **HCA**
  (grouped low-rank O under `self_attention.o_head_grouped.*`). Hash-routed
  MoE layers carry `mlp.router.tid2eid` instead of `mlp.router.bias`.

* FP8/FP4 quantized weights are dequantized to BF16 on load via
  :meth:`maybe_modify_loaded_hf_weight`; the corresponding `.scale` sidecar
  keys are absorbed there and never appear in the mapping registry.
"""

from __future__ import annotations

import logging
from typing import Mapping, Union

import torch
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mla_provider import MLAModelProvider


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dequantization helpers (block-scaled FP8 E4M3 as used by DSV4-Flash)
# ---------------------------------------------------------------------------

_FP8_BLOCK = 128  # per config.quantization_config.weight_block_size = [128, 128]


def _dequantize_block_scaled_fp8(
    weight: torch.Tensor, scale: torch.Tensor, *, block_size: int = _FP8_BLOCK
) -> torch.Tensor:
    """Dequantize FP8 E4M3 block-scaled weights to BF16.

    `weight`: [O, I] in `float8_e4m3fn`.
    `scale` : [ceil(O/B), ceil(I/B)] in float32 or float8_e8m0fnu (ue8m0).

    Returns a BF16 tensor of shape [O, I]. Scale is broadcast over `block_size`
    along both dimensions (the `.to(bf16)` before the multiply keeps the
    product in BF16 and avoids FP32 intermediates that blow memory up).
    """
    out_dim, in_dim = weight.shape
    w_f32 = weight.to(torch.float32)
    s_f32 = scale.to(torch.float32)
    # Expand scale to per-element via repeat_interleave along each dim; clip to
    # the matrix size in case the last block is partial.
    s_expanded = s_f32.repeat_interleave(block_size, dim=0)[:out_dim].repeat_interleave(block_size, dim=1)[:, :in_dim]
    return (w_f32 * s_expanded).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


@MegatronModelBridge.register_bridge(
    source="DeepseekV4ForCausalLM",
    target=GPTModel,
    provider=MLAModelProvider,
    model_type="deepseek_v4",
)
class DeepSeekV4Bridge(MegatronModelBridge):
    """Megatron Bridge for DeepSeek-V4-Flash.

    Target is registered as `GPTModel` with `MLAModelProvider` for dispatch
    bookkeeping; the actual V4 Megatron model will subclass from there once
    HC/CSA/HCA forward paths are wired up.
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MLAModelProvider:
        """Configure MLAModelProvider with V4-specific attributes."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.share_embeddings_and_output_weights = False
        provider.qk_layernorm = True
        provider.multi_latent_attention = True

        # V4 attention specifics — no kv_lora_rank (single KV proj), grouped O.
        provider.q_lora_rank = hf_config.q_lora_rank
        provider.qk_head_dim = hf_config.head_dim - hf_config.qk_rope_head_dim
        provider.qk_pos_emb_head_dim = hf_config.qk_rope_head_dim
        provider.v_head_dim = hf_config.head_dim
        # These do not exist on MLATransformerConfig — stashed as attrs for the
        # future V4-aware provider to consume.
        provider.o_groups = hf_config.o_groups
        provider.o_lora_rank = hf_config.o_lora_rank
        provider.window_size = hf_config.sliding_window
        provider.compress_ratios = tuple(hf_config.compress_ratios)
        provider.index_n_heads = hf_config.index_n_heads
        provider.index_head_dim = hf_config.index_head_dim
        provider.index_topk = hf_config.index_topk

        # MoE + routing
        provider.moe_grouped_gemm = True
        provider.moe_router_pre_softmax = False  # sqrtsoftplus handled in forward
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_router_load_balancing_type = "none"
        provider.moe_shared_expert_overlap = True
        provider.moe_router_score_function = "sqrtsoftplus"  # V4-specific; may need provider ext
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_dtype = "fp32"
        provider.moe_aux_loss_coeff = 0.0
        provider.n_hash_layers = hf_config.num_hash_layers
        provider.hc_mult = hf_config.hc_mult
        provider.hc_sinkhorn_iters = hf_config.hc_sinkhorn_iters
        provider.hc_eps = hf_config.hc_eps
        provider.swiglu_limit = hf_config.swiglu_limit
        provider.routed_scaling_factor = hf_config.routed_scaling_factor

        provider.moe_ffn_hidden_size = hf_config.moe_intermediate_size
        provider.moe_shared_expert_intermediate_size = hf_config.moe_intermediate_size * hf_config.n_shared_experts

        # All V4 layers are MoE (no dense replace) — no moe_layer_freq needed.
        provider.mtp_num_layers = getattr(hf_config, "num_nextn_predict_layers", 0) or None

        provider.apply_rope_fusion = False
        provider.hidden_dropout = 0.0
        provider.attention_softmax_in_fp32 = False
        provider.make_vocab_size_divisible_by = 1280

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Enumerate all HF↔Megatron weight name mappings for DeepSeek-V4-Flash."""
        hf_config = self.hf_config
        num_transformer_layers = hf_config.num_hidden_layers
        num_mtp_layers = getattr(hf_config, "num_nextn_predict_layers", 0) or 0
        num_experts = hf_config.n_routed_experts
        num_hash_layers = getattr(hf_config, "num_hash_layers", 0)

        mapping_list: list = []

        # ------------------------------------------------------------------
        # Root
        # ------------------------------------------------------------------
        mapping_list.extend(
            [
                AutoMapping(megatron_param="embedding.word_embeddings.weight", hf_param="embed.weight"),
                AutoMapping(megatron_param="decoder.final_layernorm.weight", hf_param="norm.weight"),
                AutoMapping(megatron_param="output_layer.weight", hf_param="head.weight"),
                AutoMapping(megatron_param="decoder.hc_head_fn", hf_param="hc_head_fn"),
                AutoMapping(megatron_param="decoder.hc_head_base", hf_param="hc_head_base"),
                AutoMapping(megatron_param="decoder.hc_head_scale", hf_param="hc_head_scale"),
            ]
        )

        # ------------------------------------------------------------------
        # Per-layer mappings — wildcarded
        # ------------------------------------------------------------------
        mapping_list.extend(_layer_mappings(num_experts=num_experts))

        # ------------------------------------------------------------------
        # MTP layers — explicit, same pattern but different prefix.
        # ------------------------------------------------------------------
        for mtp_layer in range(num_mtp_layers):
            hf_idx = num_transformer_layers + mtp_layer
            mapping_list.extend(_mtp_layer_mappings(mtp_layer, hf_idx, num_experts))

        # Cache which layers are hash-routed so maybe_modify_loaded_hf_weight can
        # route `gate.tid2eid` vs `gate.bias` correctly.
        self._num_hash_layers = num_hash_layers

        return MegatronMappingRegistry(*mapping_list)

    # ------------------------------------------------------------------
    # Weight-modification hooks
    # ------------------------------------------------------------------

    def maybe_modify_loaded_hf_weight(
        self,
        hf_param: Union[str, dict[str, str]],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        """Dequantize FP8/FP4 weights to BF16 using `.scale` sidecars.

        The real DSV4-Flash checkpoint stores:
          * attention / shared_experts: `float8_e4m3fn` with a float32
            `.scale` block tensor (128×128 blocks).
          * routed experts: FP4 packed as `int8` with a `float8_e8m0fnu`
            `.scale` block tensor.

        We dequantize everything to BF16 here so that downstream conversion
        sees plain BF16 tensors and the Megatron side never carries `.scale`.
        """
        hf_weights = super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)
        if isinstance(hf_weights, dict):
            return {
                key: self._dequantize_one(tensor, hf_param[key], hf_state_dict) for key, tensor in hf_weights.items()
            }
        return self._dequantize_one(hf_weights, hf_param, hf_state_dict)

    @staticmethod
    def _dequantize_one(
        weight: torch.Tensor,
        param_name: str,
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Dequantize a single tensor if it has a `.scale` sidecar."""
        if not param_name.endswith(".weight"):
            return weight

        scale_key = param_name[: -len(".weight")] + ".scale"
        if scale_key not in hf_state_dict:
            return weight

        scale = hf_state_dict[scale_key]

        # FP8 E4M3 block-scaled
        if weight.dtype == torch.float8_e4m3fn:
            return _dequantize_block_scaled_fp8(weight, scale)

        # FP4 packed experts (int8, two fp4 vals per byte). Expansion/dequant
        # via DS's reference FP4_TABLE is implemented in `_fp4_to_bf16`.
        if weight.dtype == torch.int8:
            return _fp4_to_bf16(weight, scale)

        return weight


# ---------------------------------------------------------------------------
# Mapping-list builders
# ---------------------------------------------------------------------------

# Block-relative name pairs (Megatron-side → HF-side) that are 1:1 renames.
_SIMPLE_BLOCK_MAPPINGS: dict[str, str] = {
    # Norms + HC
    "input_layernorm.weight": "attn_norm.weight",
    "pre_mlp_layernorm.weight": "ffn_norm.weight",
    "hc_attn.mapping_proj.weight": "hc_attn_fn",
    "hc_attn.bias": "hc_attn_base",
    "hc_attn.alpha": "hc_attn_scale",
    "hc_ffn.mapping_proj.weight": "hc_ffn_fn",
    "hc_ffn.bias": "hc_ffn_base",
    "hc_ffn.alpha": "hc_ffn_scale",
    # Attention — CSA
    "self_attention.attn_sink": "attn.attn_sink",
    "self_attention.linear_q_down_proj.weight": "attn.wq_a.weight",
    "self_attention.linear_q_up_proj.weight": "attn.wq_b.weight",
    "self_attention.q_layernorm.weight": "attn.q_norm.weight",
    "self_attention.linear_kv_proj.weight": "attn.wkv.weight",
    "self_attention.kv_layernorm.weight": "attn.kv_norm.weight",
    # Attention — HCA (grouped low-rank O)
    "self_attention.o_head_grouped.linear_o_down_proj.weight": "attn.wo_a.weight",
    "self_attention.o_head_grouped.linear_o_up_proj.weight": "attn.wo_b.weight",
    # Attention — per-layer Compressor (gated by compress_ratios!=0)
    "self_attention.compressor.ape": "attn.compressor.ape",
    "self_attention.compressor.wkv.weight": "attn.compressor.wkv.weight",
    "self_attention.compressor.wgate.weight": "attn.compressor.wgate.weight",
    "self_attention.compressor.norm.weight": "attn.compressor.norm.weight",
    # Attention — per-layer Indexer (gated by compress_ratios==4)
    "self_attention.indexer.linear_q_up_proj.weight": "attn.indexer.wq_b.weight",
    "self_attention.indexer.weights_proj.weight": "attn.indexer.weights_proj.weight",
    "self_attention.indexer.compressor.ape": "attn.indexer.compressor.ape",
    "self_attention.indexer.compressor.wkv.weight": "attn.indexer.compressor.wkv.weight",
    "self_attention.indexer.compressor.wgate.weight": "attn.indexer.compressor.wgate.weight",
    "self_attention.indexer.compressor.norm.weight": "attn.indexer.compressor.norm.weight",
    # MoE — router + shared expert down-projection
    "mlp.router.weight": "ffn.gate.weight",
    "mlp.router.bias": "ffn.gate.bias",
    "mlp.router.tid2eid": "ffn.gate.tid2eid",
    "mlp.experts.linear_fc2.weight*": "ffn.experts.*.w2.weight",
    "mlp.shared_experts.linear_fc2.weight": "ffn.shared_experts.w2.weight",
}


def _layer_mappings(*, num_experts: int) -> list:
    """Return the wildcarded per-transformer-layer mapping list."""
    mappings: list = []
    for mg_rel, hf_rel in _SIMPLE_BLOCK_MAPPINGS.items():
        mappings.append(
            AutoMapping(
                megatron_param=f"decoder.layers.*.{mg_rel}",
                hf_param=f"layers.*.{hf_rel}",
            )
        )
    # Gated-MLP fusions: Megatron's single linear_fc1 ↔ HF's separate w1/w3.
    mappings.extend(
        [
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                gate="layers.*.ffn.experts.*.w1.weight",
                up="layers.*.ffn.experts.*.w3.weight",
            ),
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                gate="layers.*.ffn.shared_experts.w1.weight",
                up="layers.*.ffn.shared_experts.w3.weight",
            ),
        ]
    )
    return mappings


def _mtp_layer_mappings(mtp_layer: int, hf_idx: int, num_experts: int) -> list:
    """Return the explicit mapping list for a single MTP layer."""
    prefix_mg = f"mtp.layers.{mtp_layer}"
    inner_mg = f"{prefix_mg}.mtp_model_layer"
    prefix_hf = f"mtp.{mtp_layer}"

    mappings: list = [
        # MTP extras (not present in regular layers).
        AutoMapping(megatron_param=f"{prefix_mg}.e_proj.weight", hf_param=f"{prefix_hf}.e_proj.weight"),
        AutoMapping(megatron_param=f"{prefix_mg}.h_proj.weight", hf_param=f"{prefix_hf}.h_proj.weight"),
        AutoMapping(megatron_param=f"{prefix_mg}.enorm.weight", hf_param=f"{prefix_hf}.enorm.weight"),
        AutoMapping(megatron_param=f"{prefix_mg}.hnorm.weight", hf_param=f"{prefix_hf}.hnorm.weight"),
        AutoMapping(megatron_param=f"{prefix_mg}.final_layernorm.weight", hf_param=f"{prefix_hf}.norm.weight"),
        AutoMapping(megatron_param=f"{prefix_mg}.hc_head_fn", hf_param=f"{prefix_hf}.hc_head_fn"),
        AutoMapping(megatron_param=f"{prefix_mg}.hc_head_base", hf_param=f"{prefix_hf}.hc_head_base"),
        AutoMapping(megatron_param=f"{prefix_mg}.hc_head_scale", hf_param=f"{prefix_hf}.hc_head_scale"),
    ]

    # Same block pattern as transformer layers — but with explicit indices.
    for mg_rel, hf_rel in _SIMPLE_BLOCK_MAPPINGS.items():
        mappings.append(
            AutoMapping(
                megatron_param=f"{inner_mg}.{mg_rel}",
                hf_param=f"{prefix_hf}.{hf_rel}",
            )
        )

    mappings.extend(
        [
            GatedMLPMapping(
                megatron_param=f"{inner_mg}.mlp.experts.linear_fc1.weight*",
                gate=f"{prefix_hf}.ffn.experts.*.w1.weight",
                up=f"{prefix_hf}.ffn.experts.*.w3.weight",
            ),
            GatedMLPMapping(
                megatron_param=f"{inner_mg}.mlp.shared_experts.linear_fc1.weight",
                gate=f"{prefix_hf}.ffn.shared_experts.w1.weight",
                up=f"{prefix_hf}.ffn.shared_experts.w3.weight",
            ),
        ]
    )
    return mappings


# ---------------------------------------------------------------------------
# FP4 dequant (reference, not fused)
# ---------------------------------------------------------------------------


_FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

_FP4_BLOCK = 32  # per inference/convert.py


def _fp4_to_bf16(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Unpack FP4 (two nibbles per int8) to BF16 using DS's FP4_TABLE + ue8m0 scale.

    See `inference/convert.py:cast_e2m1fn_to_e4m3fn` for the intended layout:
    `weight: [O, I/2]` int8 with two FP4 values per byte; `scale: [O, I/32]`
    in float8_e8m0fnu. We expand to [O, I] and multiply by scale per
    32-wide block.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D FP4 weight, got shape {tuple(weight.shape)}")
    out_dim, packed_in = weight.shape
    in_dim = packed_in * 2
    w_u8 = weight.view(torch.uint8)
    low = w_u8 & 0x0F
    high = (w_u8 >> 4) & 0x0F
    table = _FP4_TABLE.to(weight.device)
    w = torch.stack([table[low.long()], table[high.long()]], dim=-1).flatten(1)
    # scale is per (out, in/block) — expand to per-element along in_dim.
    s_bf16 = scale.to(torch.bfloat16)
    s_exp = s_bf16.repeat_interleave(_FP4_BLOCK, dim=1)[:, :in_dim]
    return (w.to(torch.bfloat16) * s_exp).to(torch.bfloat16)
