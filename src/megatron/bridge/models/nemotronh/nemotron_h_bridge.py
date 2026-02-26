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

import logging
from typing import Dict, Optional, Tuple

import torch
from megatron.core.activations import squared_relu
from megatron.core.models.mamba import MambaModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    MambaConv1dMapping,
    MambaInProjMapping,
    QKVMapping,
    RowParallelMapping,
    MegatronParamMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider


logger = logging.getLogger(__name__)


def _count_wildcards(pattern: str) -> int:
    # Mirrors MegatronParamMapping._count_wildcard_groups (private there).
    count = 0
    remaining = pattern
    while "**" in remaining:
        count += 1
        remaining = remaining.replace("**", "", 1)
    count += remaining.count("*")
    return count


def _replace_wildcards(pattern: str, captures: Tuple[str, ...]) -> str:
    """Replace ** then * sequentially with captures."""
    out = pattern
    capture_index = 0
    while "**" in out and capture_index < len(captures):
        out = out.replace("**", captures[capture_index], 1)
        capture_index += 1
    while "*" in out and capture_index < len(captures):
        out = out.replace("*", captures[capture_index], 1)
        capture_index += 1
    return out


class _MTPFlatteningMapping(MegatronParamMapping[torch.Tensor]):
    """
    Pattern mapping that flattens Megatron's two-level MTP indices:

      megatron: mtp.layers.{outer}.mtp_model_layer.layers.{inner}.<...>
      hf:       mtp.layers.{outer * L + inner}.<...>

    Also supports an optional `inner_override` for parameters that live outside
    `mtp_model_layer.layers.*` in Megatron but should attach to a specific inner
    layer in HF (e.g., eh_proj on inner=0, final_layernorm on inner=L-1).
    """

    def __init__(
        self,
        megatron_param: str,
        hf_param: str,
        *,
        mtp_layers_per_block: int,
        inner_override: Optional[int] = None,
    ):
        # NOTE: We intentionally bypass wildcard validation because Megatron has
        # 2+N wildcards while HF has 1+N in the flattened scheme.
        self.megatron_param = megatron_param
        self.hf_param = hf_param
        self._mtp_layers_per_block = int(mtp_layers_per_block)
        self._inner_override = inner_override

        if self._mtp_layers_per_block <= 0:
            raise ValueError(
                "mtp_layers_per_block must be > 0 for MTP flattening mappings. "
                "This should come from len(hf_config.mtp_hybrid_override_pattern)."
            )

        # minimal init for PP/TP groups (copied pattern from MegatronParamMapping.__init__)
        self._broadcast_obj_cache = {}
        self._tensor_spec_output_cache = {}
        if torch.distributed.is_available():
            # Groups are populated in MegatronParamMapping when mpu is initialized.
            # Here we don't need them because this class resolves into AutoMapping/QKVMapping
            # (which will handle TP/PP), and we never run conversions on this instance.
            self.pp_group = None
            self.ep_group = None
            self._tp_group = None
            self._etp_group = None

        # Precompute wildcard counts to disambiguate resolve() calls coming from
        # megatron_to_hf_lookup vs hf_to_megatron_lookup.
        self._megatron_wc = _count_wildcards(self.megatron_param)
        self._hf_wc = _count_wildcards(self.hf_param) if isinstance(self.hf_param, str) else 0

    def resolve(self, captures: Tuple[str, ...]) -> MegatronParamMapping:
        # We primarily expect captures from Megatron pattern matching.
        # If called from HF reverse lookup, captures length will typically match hf wildcards.
        treat_as_hf = (len(captures) == self._hf_wc) and (self._hf_wc != self._megatron_wc)

        if treat_as_hf:
            # captures: (flat_layer_idx, [maybe more like expert idx...])
            flat = int(captures[0])
            outer = flat // self._mtp_layers_per_block
            inner = flat % self._mtp_layers_per_block
            # For outer-only megatron params (eh_proj, etc.) we only need `outer`.
            # If the megatron pattern contains an inner wildcard, use computed `inner`.
            # Build a synthetic capture tuple matching the megatron pattern.
            # We replace the first two megatron wildcards with outer/inner unless inner_override is set.
            inner_eff = self._inner_override if self._inner_override is not None else inner
            # Preserve any remaining captures (e.g., expert id) after the flat idx.
            remaining = captures[1:]
            megatron_captures = (str(outer), str(inner_eff), *remaining)
            resolved_megatron = _replace_wildcards(self.megatron_param, megatron_captures)
            resolved_hf = _replace_wildcards(self.hf_param, captures)
            return AutoMapping(megatron_param=resolved_megatron, hf_param=resolved_hf)

        # Treat captures as coming from the Megatron pattern.
        # captures: (outer, inner, [maybe expert idx...]) for nested params
        # or (outer,) for outer-only params like eh_proj.
        if len(captures) < 1:
            raise ValueError(f"Expected at least 1 capture for MTP mapping, got {captures}")

        outer = int(captures[0])

        # Determine the effective inner index
        if self._inner_override is not None:
            inner_eff = self._inner_override
            remaining = captures[1:]
        else:
            if len(captures) < 2:
                raise ValueError(
                    f"Expected 2 captures (outer, inner) for MTP nested mapping, got {captures}"
                )
            inner_eff = int(captures[1])
            remaining = captures[2:]

        flat = outer * self._mtp_layers_per_block + inner_eff

        resolved_megatron = _replace_wildcards(self.megatron_param, captures)

        # Resolve HF: first wildcard is the flattened idx, remaining wildcards pass through.
        hf_captures = (str(flat), *remaining)
        resolved_hf = _replace_wildcards(self.hf_param, hf_captures)

        return AutoMapping(megatron_param=resolved_megatron, hf_param=resolved_hf)

    # These are never called: this mapping always resolves into an AutoMapping.
    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: torch.nn.Module) -> torch.Tensor:
        raise NotImplementedError

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[torch.nn.Module]
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class _MTPFlatteningQKVMapping(MegatronParamMapping[Dict[str, torch.Tensor]]):
    """Resolve-time wrapper that flattens MTP indices then delegates to QKVMapping."""

    def __init__(
        self,
        megatron_param: str,
        *,
        q: str,
        k: str,
        v: str,
        mtp_layers_per_block: int,
    ):
        # Bypass wildcard validation (2 wildcards on Megatron side vs 1 on HF q/k/v).
        self.megatron_param = megatron_param
        self.hf_param = {"q": q, "k": k, "v": v}
        self._mtp_layers_per_block = int(mtp_layers_per_block)
        if self._mtp_layers_per_block <= 0:
            raise ValueError("mtp_layers_per_block must be > 0 for MTP flattening QKV mapping.")
        self._megatron_wc = _count_wildcards(self.megatron_param)
        self._hf_wc = _count_wildcards(q)  # q/k/v share pattern structure here

    def resolve(self, captures: Tuple[str, ...]) -> MegatronParamMapping:
        # Expect captures from Megatron lookup: (outer, inner)
        if len(captures) < 2:
            raise ValueError(f"Expected (outer, inner) captures for MTP QKV mapping, got {captures}")
        outer = int(captures[0])
        inner = int(captures[1])
        flat = outer * self._mtp_layers_per_block + inner

        resolved_megatron = _replace_wildcards(self.megatron_param, captures)
        resolved_q = _replace_wildcards(self.hf_param["q"], (str(flat),))
        resolved_k = _replace_wildcards(self.hf_param["k"], (str(flat),))
        resolved_v = _replace_wildcards(self.hf_param["v"], (str(flat),))

        return QKVMapping(megatron_param=resolved_megatron, q=resolved_q, k=resolved_k, v=resolved_v)

    def hf_to_megatron(self, hf_weights: Dict[str, torch.Tensor], megatron_module: torch.nn.Module) -> torch.Tensor:
        raise NotImplementedError

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[torch.nn.Module]
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


@MegatronModelBridge.register_bridge(
    source="NemotronHForCausalLM",
    target=MambaModel,
    provider=MambaModelProvider,
    model_type="nemotron_h",
)
class NemotronHBridge(MegatronModelBridge):
    """
    Megatron Bridge for Nemotron-H Causal LM.

    This bridge handles the conversion between HuggingFace NemotronHForCausalLM
    and Megatron-Core MambaModel formats, including weight mappings and
    configuration translation.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("nvidia/Nemotron-H-8B-Base-8K", trust_remote_code=True)
        >>> provider = bridge.to_megatron_provider()
    """
    CONFIG_MAPPING = MegatronModelBridge.CONFIG_MAPPING + [
        # Mamba-specific fields
        ("mamba_head_dim", "mamba_head_dim"),
        ("mamba_num_heads", "mamba_num_heads"),
        ("n_groups", "mamba_num_groups"),
        ("ssm_state_size", "mamba_state_dim"),
        ("hybrid_override_pattern", "hybrid_override_pattern"),
        ("residual_in_fp32", "fp32_residual_connection"),
        ("use_bias", "add_bias_linear"),
        ("layer_norm_epsilon", "layernorm_epsilon"),
        # MoE-specific fields (already in base but with different HF names)
        ("moe_shared_expert_intermediate_size", "moe_shared_expert_intermediate_size"),
    ]

    # Additional files to copy during HF export (reasoning parser utilities)
    ADDITIONAL_FILE_PATTERNS = ["*reasoning_parser.py"]

    def __init__(self):
        super().__init__()
        self._mtp_layers_per_block: Optional[int] = None

    def build_conversion_tasks(self, hf_pretrained: PreTrainedCausalLM, megatron_model):
        # Cache MTP block depth (len of mtp_hybrid_override_pattern) so mapping_registry()
        # can compute the flattened HF layer indices deterministically.
        mtp_pattern = getattr(getattr(hf_pretrained, "config", None), "mtp_hybrid_override_pattern", None)
        if mtp_pattern is not None:
            self._mtp_layers_per_block = len(mtp_pattern)
        else:
            self._mtp_layers_per_block = 0

        return super().build_conversion_tasks(hf_pretrained, megatron_model)

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MambaModelProvider:
        """Convert HuggingFace Nemotron-H config to MambaModelProvider."""
        # Use base class for common config conversion
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # Nemotron-H specific defaults
        provider.activation_func = squared_relu
        provider.masked_softmax_fusion = True
        provider.apply_query_key_layer_scaling = False
        provider.persist_layer_norm = True
        provider.attention_softmax_in_fp32 = False
        provider.first_last_layers_bf16 = True
        provider.is_hybrid_model = True

        # MoE-specific defaults (only if MoE is enabled)
        if hasattr(hf_config, "n_routed_experts") and hf_config.n_routed_experts > 0:
            provider.moe_aux_loss_coeff = 0.0001
            provider.moe_router_score_function = "sigmoid"
            provider.moe_router_enable_expert_bias = True
            provider.moe_router_load_balancing_type = "seq_aux_loss"
            provider.moe_router_dtype = "fp32"
            provider.moe_grouped_gemm = True
            provider.moe_token_dispatcher_type = "alltoall"
            provider.moe_permute_fusion = True
            provider.moe_shared_expert_overlap = True

        
        if hasattr(hf_config, "moe_latent_size"):
            provider.moe_latent_size = hf_config.moe_latent_size
        if hasattr(hf_config, "moe_shared_expert_overlap"):
            provider.moe_shared_expert_overlap = hf_config.moe_shared_expert_overlap
        if hasattr(hf_config, "num_nextn_predict_layers"):
            provider.mtp_num_layers = hf_config.num_nextn_predict_layers
        if hasattr(hf_config, "mtp_hybrid_override_pattern"):
            provider.mtp_hybrid_override_pattern = hf_config.mtp_hybrid_override_pattern
        if hasattr(hf_config, "keep_mtp_spec_in_bf16"):
            provider.keep_mtp_spec_in_bf16 = hf_config.keep_mtp_spec_in_bf16

        return provider


    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        # Dictionary maps Megatron parameter names -> HF parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            "decoder.layers.*.mlp.linear_fc1.weight": "backbone.layers.*.mixer.up_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "backbone.layers.*.mixer.down_proj.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "backbone.layers.*.mixer.o_proj.weight",
            "decoder.final_norm.weight": "backbone.norm_f.weight",
            # Fused TE layer norm weights (when using TELayerNormColumnParallelLinear)
            # if the megatron key does not exist for a given layer it will be ignored,
            # so only one of these will be used per layer
            "decoder.layers.*.mixer.in_proj.layer_norm_weight": "backbone.layers.*.norm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "backbone.layers.*.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "backbone.layers.*.norm.weight",
            # Separate Norm layer weights (when using Norm for quantization)
            # These are used when quantization spec uses Norm instead of TENorm
            "decoder.layers.*.norm.weight": "backbone.layers.*.norm.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "backbone.layers.*.norm.weight",
            "decoder.layers.*.input_layernorm.weight": "backbone.layers.*.norm.weight",
            # TODO (@maanug): need to find a way to prune the vocab padding from the vocab dimension for these params
            "embedding.word_embeddings.weight": "backbone.embeddings.weight",
            "output_layer.weight": "lm_head.weight",
            # MoE layers
            "decoder.layers.*.mlp.router.weight": "backbone.layers.*.mixer.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "backbone.layers.*.mixer.gate.e_score_correction_bias",
            "decoder.layers.*.mlp.experts.linear_fc1.weight*": "backbone.layers.*.mixer.experts.*.up_proj.weight",
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "backbone.layers.*.mixer.experts.*.down_proj.weight",
            # TODO(liding): check compatibility: if model does not have `fc1_latent_proj` and `fc2_latent_proj` layer
            # in megatron model: nano v3 does not have these layers, but super v3 does
            # in hf model: both variants have these layers (nano v3 has nn.Identity, super v3 has nn.Linear)
            "decoder.layers.*.mlp.fc1_latent_proj.weight": "backbone.layers.*.mixer.fc1_latent_proj.weight",
            "decoder.layers.*.mlp.fc2_latent_proj.weight": "backbone.layers.*.mixer.fc2_latent_proj.weight",
            "decoder.layers.*.mlp.shared_experts.linear_fc1.weight": "backbone.layers.*.mixer.shared_experts.up_proj.weight",
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "backbone.layers.*.mixer.shared_experts.down_proj.weight",
        }

        mtp_layers_per_block = int(self._mtp_layers_per_block or 0)

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # -----------------------------
        # MTP (flattened HF structure)
        # -----------------------------
        #
        # Megatron stores MTP in a nested form:
        #   mtp.layers.{outer}.mtp_model_layer.layers.{inner}.<...>
        # while HF stores it as:
        #   mtp.layers.{outer * L + inner}.<...>
        #
        # Additionally, some MTP parameters live outside mtp_model_layer in Megatron
        # (eh_proj/enorm/hnorm/final_layernorm) but attach to the first/last inner layer
        # in HF.
        if mtp_layers_per_block > 0:
            # First-inner (inner=0) params
            for p in ["eh_proj.weight", "enorm.weight", "hnorm.weight"]:
                mapping_list.append(
                    _MTPFlatteningMapping(
                        megatron_param=f"mtp.layers.*.{p}",
                        hf_param=f"mtp.layers.*.{p}",
                        mtp_layers_per_block=mtp_layers_per_block,
                        inner_override=0,
                    )
                )

            # Last-inner (inner=L-1) params
            mapping_list.append(
                _MTPFlatteningMapping(
                    megatron_param="mtp.layers.*.final_layernorm.weight",
                    hf_param="mtp.layers.*.final_layernorm.weight",
                    mtp_layers_per_block=mtp_layers_per_block,
                    inner_override=mtp_layers_per_block - 1,
                )
            )

            # Nested MTP transformer block params
            mtp_nested = {
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.linear_fc1.weight": "mtp.layers.*.mixer.up_proj.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.linear_fc2.weight": "mtp.layers.*.mixer.down_proj.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.self_attention.linear_proj.weight": "mtp.layers.*.mixer.o_proj.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.mixer.in_proj.layer_norm_weight": "mtp.layers.*.norm.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.linear_fc1.layer_norm_weight": "mtp.layers.*.norm.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.self_attention.linear_qkv.layer_norm_weight": "mtp.layers.*.norm.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.router.weight": "mtp.layers.*.mixer.gate.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.router.expert_bias": "mtp.layers.*.mixer.gate.e_score_correction_bias",
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.experts.linear_fc1.weight*": "mtp.layers.*.mixer.experts.*.up_proj.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.experts.linear_fc2.weight*": "mtp.layers.*.mixer.experts.*.down_proj.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.fc1_latent_proj.weight": "mtp.layers.*.mixer.fc1_latent_proj.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.fc2_latent_proj.weight": "mtp.layers.*.mixer.fc2_latent_proj.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.shared_experts.linear_fc1.weight": "mtp.layers.*.mixer.shared_experts.up_proj.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.mlp.shared_experts.linear_fc2.weight": "mtp.layers.*.mixer.shared_experts.down_proj.weight",
                "mtp.layers.*.mtp_model_layer.layers.*.pre_mlp_layernorm.weight": "mtp.layers.*.norm.weight",
            }

            for megatron_param, hf_param in mtp_nested.items():
                mapping_list.append(
                    _MTPFlatteningMapping(
                        megatron_param=megatron_param,
                        hf_param=hf_param,
                        mtp_layers_per_block=mtp_layers_per_block,
                        inner_override=None,
                    )
                )
        else:
            logger.warning(
                "mtp_layers_per_block is not set (or 0). Skipping MTP flattening mappings. "
                "If you are converting a model with MTP enabled, ensure hf_pretrained.config.mtp_hybrid_override_pattern is present."
            )

        # Handling Mamba Mixer submodules separately for more clarity
        # Special Handling for InProj and Conv1d due to specific TP logic
        for mixer_sub_module in ["A_log", "D", "dt_bias", "norm.weight"]:
            mapping_list.extend(
                [
                    ColumnParallelMapping(
                        megatron_param=rf"decoder.layers.*.mixer.{mixer_sub_module}",
                        hf_param=rf"backbone.layers.*.mixer.{mixer_sub_module}",
                    ),
                ]
            )
        mapping_list.extend(
            [
                RowParallelMapping(
                    megatron_param="decoder.layers.*.mixer.out_proj.weight",
                    hf_param="backbone.layers.*.mixer.out_proj.weight",
                ),
            ]
        )
        mapping_list.extend(
            [
                MambaInProjMapping(
                    megatron_param="decoder.layers.*.mixer.in_proj.weight",
                    hf_param="backbone.layers.*.mixer.in_proj.weight",
                ),
            ]
        )
        for conv1d_sub_module in ["weight", "bias"]:
            mapping_list.extend(
                [
                    MambaConv1dMapping(
                        megatron_param=rf"decoder.layers.*.mixer.conv1d.{conv1d_sub_module}",
                        hf_param=rf"backbone.layers.*.mixer.conv1d.{conv1d_sub_module}",
                    ),
                ]
            )
        # Add special mappings that require parameter concatenation/transformation, pruning, etc.
        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="backbone.layers.*.mixer.q_proj.weight",
                    k="backbone.layers.*.mixer.k_proj.weight",
                    v="backbone.layers.*.mixer.v_proj.weight",
                ),
            ]
        )

        if mtp_layers_per_block > 0:
            mapping_list.extend(
                [
                    _MTPFlatteningQKVMapping(
                        megatron_param="mtp.layers.*.mtp_model_layer.layers.*.self_attention.linear_qkv.weight",
                        q="mtp.layers.*.mixer.q_proj.weight",
                        k="mtp.layers.*.mixer.k_proj.weight",
                        v="mtp.layers.*.mixer.v_proj.weight",
                        mtp_layers_per_block=mtp_layers_per_block,
                    ),
                ]
            )

        return MegatronMappingRegistry(*mapping_list)
