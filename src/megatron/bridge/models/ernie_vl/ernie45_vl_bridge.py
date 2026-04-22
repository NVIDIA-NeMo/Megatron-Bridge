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

"""
Megatron Bridge for ERNIE 4.5 VL MoE (Vision-Language with Mixture of Experts).

This bridge handles conversion between HuggingFace Ernie4_5_VLMoeForConditionalGeneration
and Megatron-Core Ernie45VLModel formats, including:

- Language model weights with heterogeneous dual-pool MoE:
  * text_moe: 64 experts with intermediate_size=1536 (text tokens)
    -> mapped to ErnieMultiTypeMoE.text_moe_layer (standard MoELayer)
  * vision_moe: 64 experts with intermediate_size=512 (vision tokens)
    -> mapped to ErnieMultiTypeMoE.vision_moe_layer (standard MoELayer)
  * shared_experts: 2 shared experts with intermediate_size=3072
    -> mapped to ErnieMultiTypeMoE.shared_experts
- Vision encoder weights:
  * HF ViT (use_mg_vit=False): replicated across TP ranks via ReplicatedMapping
  * MG ViT (use_mg_vit=True): TP-sharded with ConcatenatedQKVMapping for fused QKV
- Resampler / projector weights (replicated across TP ranks)
- 3D M-RoPE position embedding configuration

The ErnieMultiTypeMoE module contains two separate MoELayer instances (one per
expert pool), each with its own router and SequentialMLP experts. This gives both
pools full TP support through standard Megatron-Core infrastructure.

HF on-disk (safetensors) keys -- after ``_checkpoint_conversion_mapping`` reversal:
    model.layers.{i}.mlp.gate.weight                       (text router)
    model.layers.{i}.mlp.gate.weight_1                     (vision router)
    model.layers.{i}.mlp.moe_statics.e_score_correction_bias  (concat text+vision)
    model.layers.{i}.mlp.experts.{j}.gate_proj.weight       (j=0..N-1 text, j=N..2N-1 vision)
    model.layers.{i}.mlp.experts.{j}.up_proj.weight
    model.layers.{i}.mlp.experts.{j}.down_proj.weight
    model.layers.{i}.mlp.shared_experts.{gate,up,down}_proj.weight
    model.vision_model.**

Megatron Weight Naming (per-expert SequentialMLP within ErnieMultiTypeMoE):
    language_model.decoder.layers.{i}.mlp.text_moe_layer.router.weight
    language_model.decoder.layers.{i}.mlp.text_moe_layer.router.expert_bias
    language_model.decoder.layers.{i}.mlp.text_moe_layer.experts.local_experts.{j}.linear_fc1.weight
    language_model.decoder.layers.{i}.mlp.text_moe_layer.experts.local_experts.{j}.linear_fc2.weight
    language_model.decoder.layers.{i}.mlp.vision_moe_layer.router.weight
    language_model.decoder.layers.{i}.mlp.vision_moe_layer.router.expert_bias
    language_model.decoder.layers.{i}.mlp.vision_moe_layer.experts.local_experts.{j}.linear_fc1.weight
    language_model.decoder.layers.{i}.mlp.vision_moe_layer.experts.local_experts.{j}.linear_fc2.weight
    language_model.decoder.layers.{i}.mlp.shared_experts.linear_fc1.weight
    language_model.decoder.layers.{i}.mlp.shared_experts.linear_fc2.weight

MG-native ViT Weight Naming (use_mg_vit=True, TP-sharded):
    vision_model.decoder.layers.{i}.self_attention.linear_qkv.weight     (fused QKV, ConcatenatedQKVMapping)
    vision_model.decoder.layers.{i}.self_attention.linear_qkv.bias
    vision_model.decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight  (fused norm1)
    vision_model.decoder.layers.{i}.self_attention.linear_qkv.layer_norm_bias
    vision_model.decoder.layers.{i}.self_attention.linear_proj.weight
    vision_model.decoder.layers.{i}.self_attention.linear_proj.bias
    vision_model.decoder.layers.{i}.mlp.linear_fc1.weight
    vision_model.decoder.layers.{i}.mlp.linear_fc1.bias
    vision_model.decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight     (fused norm2)
    vision_model.decoder.layers.{i}.mlp.linear_fc1.layer_norm_bias
    vision_model.decoder.layers.{i}.mlp.linear_fc2.weight
    vision_model.decoder.layers.{i}.mlp.linear_fc2.bias
    vision_model.patch_embed.proj.weight                                  (replicated)
    vision_model.decoder.final_layernorm.weight
    vision_model.decoder.final_layernorm.bias

Note on Expert Parallelism:
    EP>1 is supported for dual-pool MoE. The bridge handles the expert offset
    between text and vision pools correctly: text experts use indices 0..N-1 and
    vision experts use N..2N-1 in HF on-disk format. The framework's
    `_megatron_local_name_to_global` function handles SequentialMLP-style expert
    numbering, and `gather_from_ep_ranks` preserves pool offsets when
    reconstructing HF parameter names during export.
"""

import logging
import re
from typing import Dict, Optional, Tuple

import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ConcatenatedQKVMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.ernie_vl.ernie45_vl_provider import Ernie45VLModelProvider
from megatron.bridge.models.ernie_vl.modeling_ernie45_vl.model import Ernie45VLModel
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.utils.common_utils import extract_expert_number_from_param


logger = logging.getLogger(__name__)

# Use string-based registration since the HF model class may not be importable
# if transformers is an older version or the model isn't registered yet.
_ERNIE45_VL_MOE_HF_CLASS_NAME = "Ernie4_5_VLMoeForConditionalGeneration"


# ---------------------------------------------------------------------------
# Dual-pool expert EP export support
# ---------------------------------------------------------------------------
# In ERNIE VL's dual-pool MoE, vision expert j maps to HF flat expert
# (j + num_text_experts).  The _Offset*Mapping classes handle the HF→Megatron
# direction by shifting expert indices during resolve().  For the Megatron→HF
# export direction with EP > 1, gather_from_ep_ranks must reconstruct the
# pool-offset HF expert indices.  The mixin below overrides that method so
# the offset logic stays in this file instead of the base param_mapping.py.
# ---------------------------------------------------------------------------


class _DualPoolExpertMixin:
    """Mixin that adds pool-offset awareness to ``gather_from_ep_ranks``.

    For dual-pool MoE, the resolved HF param name already carries the offset
    (e.g. ``experts.67.weight`` for vision expert 3 with 64 text experts).
    The mixin computes ``pool_offset = hf_index - megatron_local_index`` and
    uses it when constructing the HF names for every EP rank.
    """

    def gather_from_ep_ranks(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module,
        hf_param_name: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.ep_size == 1:
            return {str(hf_param_name): megatron_weights}

        if megatron_module is None:
            num_experts_per_rank = self.broadcast_obj_from_pp_rank(None, "num_experts_per_rank")
        else:
            model_config = self._get_config(megatron_module)
            num_experts = model_config.num_moe_experts
            num_experts_per_rank = num_experts // self.ep_size
            num_experts_per_rank = self.broadcast_obj_from_pp_rank(num_experts_per_rank, "num_experts_per_rank")

        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        local_expert_number = global_expert_number % num_experts_per_rank

        # Compute pool offset from the resolved HF param name.
        hf_expert_match = re.search(r"experts\.(\d+)", str(hf_param_name))
        if hf_expert_match:
            hf_expert_number = int(hf_expert_match.group(1))
            pool_offset = hf_expert_number - local_expert_number
        else:
            pool_offset = 0

        gathered_expert_param_names = [
            re.sub(
                r"experts\.(\d+)",
                f"experts.{pool_offset + int(local_expert_number) + num_experts_per_rank * i}",
                str(hf_param_name),
            )
            for i in range(self.ep_size)
        ]
        assert str(hf_param_name) in gathered_expert_param_names

        gathered_weights = [torch.empty_like(megatron_weights) for _ in range(self.ep_size)]
        torch.distributed.all_gather(gathered_weights, megatron_weights, group=self.ep_group)

        weights_dict: Dict[str, torch.Tensor] = {}
        for i, param_name in enumerate(gathered_expert_param_names):
            if param_name in weights_dict:
                weights_dict[param_name] = torch.cat(
                    [weights_dict[param_name], gathered_weights[i].unsqueeze(0)], dim=0
                )
            else:
                weights_dict[param_name] = gathered_weights[i].unsqueeze(0)
        for param_name in weights_dict:
            weights_dict[param_name] = weights_dict[param_name].squeeze()
        return weights_dict


class _DualPoolGatedMLPMapping(_DualPoolExpertMixin, GatedMLPMapping):
    """GatedMLPMapping with pool-offset-aware EP export."""

    pass


class _DualPoolAutoMapping(_DualPoolExpertMixin, AutoMapping):
    """AutoMapping with pool-offset-aware EP export."""

    pass


class _OffsetGatedMLPMapping(GatedMLPMapping):
    """GatedMLPMapping that adds a fixed offset to the expert index wildcard.

    Used for vision MoE experts where Megatron expert index j maps to
    HF flat expert index (j + offset).  The offset is the number of text
    experts, so vision expert 0 maps to HF expert N, etc.

    The Megatron pattern has two ``*`` wildcards: layer index and expert index.
    The HF patterns also have two ``*`` wildcards.  During ``resolve()``, the
    second capture (expert index) is shifted by ``expert_offset`` **only** for
    HF patterns.  The Megatron side keeps the original (unshifted) expert index.
    """

    def __init__(self, megatron_param: str, gate: str, up: str, expert_offset: int = 0):
        super().__init__(megatron_param=megatron_param, gate=gate, up=up)
        self._expert_offset = expert_offset

    def resolve(self, captures: Tuple[str, ...]):
        """Override resolve to apply the expert index offset only to HF side."""
        if self._expert_offset and len(captures) >= 2:
            # Use original captures for Megatron, shifted for HF
            shifted_expert = str(int(captures[1]) + self._expert_offset)
            hf_captures = (captures[0], shifted_expert) + captures[2:]
        else:
            hf_captures = captures
        # Resolve Megatron side with original captures
        resolved_megatron_param = self.megatron_param
        idx = 0
        while "**" in resolved_megatron_param and idx < len(captures):
            resolved_megatron_param = resolved_megatron_param.replace("**", captures[idx], 1)
            idx += 1
        while "*" in resolved_megatron_param and idx < len(captures):
            resolved_megatron_param = resolved_megatron_param.replace("*", captures[idx], 1)
            idx += 1
        # Resolve HF side with shifted captures
        resolved_hf_param = {}
        for k, v in self.hf_param.items():
            resolved_v = v
            idx = 0
            while "**" in resolved_v and idx < len(hf_captures):
                resolved_v = resolved_v.replace("**", hf_captures[idx], 1)
                idx += 1
            while "*" in resolved_v and idx < len(hf_captures):
                resolved_v = resolved_v.replace("*", hf_captures[idx], 1)
                idx += 1
            resolved_hf_param[k] = resolved_v
        return _DualPoolGatedMLPMapping(
            megatron_param=resolved_megatron_param,
            gate=resolved_hf_param["gate"],
            up=resolved_hf_param["up"],
        )


class _OffsetAutoMapping(AutoMapping):
    """AutoMapping that adds a fixed offset to the expert index wildcard.

    Same concept as _OffsetGatedMLPMapping but for simple 1:1 mappings
    (e.g. down_proj).  The offset is applied only to HF side, not Megatron.
    """

    def __init__(self, megatron_param: str, hf_param: str, expert_offset: int = 0):
        super().__init__(megatron_param=megatron_param, hf_param=hf_param)
        self._expert_offset = expert_offset

    def resolve(self, captures: Tuple[str, ...]):
        """Override resolve to apply the expert index offset only to HF side."""
        if self._expert_offset and len(captures) >= 2:
            shifted_expert = str(int(captures[1]) + self._expert_offset)
            hf_captures = (captures[0], shifted_expert) + captures[2:]
        else:
            hf_captures = captures
        # Resolve Megatron side with original captures
        resolved_megatron_param = self.megatron_param
        idx = 0
        while "**" in resolved_megatron_param and idx < len(captures):
            resolved_megatron_param = resolved_megatron_param.replace("**", captures[idx], 1)
            idx += 1
        while "*" in resolved_megatron_param and idx < len(captures):
            resolved_megatron_param = resolved_megatron_param.replace("*", captures[idx], 1)
            idx += 1
        # Resolve HF side with shifted captures
        resolved_hf_param = self.hf_param
        idx = 0
        while "**" in resolved_hf_param and idx < len(hf_captures):
            resolved_hf_param = resolved_hf_param.replace("**", hf_captures[idx], 1)
            idx += 1
        while "*" in resolved_hf_param and idx < len(hf_captures):
            resolved_hf_param = resolved_hf_param.replace("*", hf_captures[idx], 1)
            idx += 1
        return _DualPoolAutoMapping(
            megatron_param=resolved_megatron_param,
            hf_param=resolved_hf_param,
        )


class _ConcatBiasMapping(AutoMapping):
    """Mapping for the concatenated text+vision expert bias tensor.

    The on-disk HF format stores a single ``moe_statics.e_score_correction_bias``
    tensor of shape ``[2, num_experts]`` where row 0 is the text pool bias and
    row 1 is the vision pool bias.  This mapping extracts the appropriate row
    based on ``slice_name``.

    For export (megatron_to_hf), the text mapping buffers its bias in a
    class-level dict keyed by resolved HF param name.  The vision mapping
    retrieves the buffered text bias, stacks them into ``[2, N]``, and
    returns the merged tensor.  This ensures only one entry per HF key.
    """

    # Class-level buffer: {resolved_hf_param: text_bias_tensor}
    _export_buffer: Dict[str, torch.Tensor] = {}

    @classmethod
    def clear_export_buffer(cls):
        """Remove any stale entries from the class-level export buffer."""
        cls._export_buffer.clear()

    def __init__(self, megatron_param: str, hf_param: str, slice_name: str, num_experts: int):
        super().__init__(megatron_param=megatron_param, hf_param=hf_param)
        self._slice_name = slice_name  # "text" or "vision"
        self._num_experts = num_experts
        self.allow_hf_name_mismatch = True

    def resolve(self, captures: Tuple[str, ...]):
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        result = _ConcatBiasMapping(
            megatron_param=resolved_megatron_param,
            hf_param=resolved_hf_param,
            slice_name=self._slice_name,
            num_experts=self._num_experts,
        )
        return result

    def hf_to_megatron(self, hf_weights, megatron_module):
        """Extract the text or vision slice from the concatenated bias.

        On-disk shape is [2, num_experts]: row 0 = text, row 1 = vision.
        """
        if self._slice_name == "text":
            sliced = hf_weights[0]
        else:
            sliced = hf_weights[1]
        return super().hf_to_megatron(sliced, megatron_module)

    def megatron_to_hf(self, megatron_weights, megatron_module):
        """Export text+vision expert bias as concatenated [2, N] tensor.

        The text mapping buffers its bias; the vision mapping retrieves it
        and stacks into [2, N].  If the text bias is not yet buffered
        (shouldn't happen in practice), falls back to exporting as-is.
        """
        result = super().megatron_to_hf(megatron_weights, megatron_module)
        if result is None:
            return result

        hf_key = str(self.hf_param)

        if self._slice_name == "text":
            # Buffer the text bias, return empty dict (don't emit yet)
            for _, tensor in result.items():
                _ConcatBiasMapping._export_buffer[hf_key] = tensor
            return {}
        else:
            # Vision: retrieve buffered text bias and merge
            text_bias = _ConcatBiasMapping._export_buffer.pop(hf_key, None)
            if text_bias is not None:
                for _, vision_bias in result.items():
                    merged = torch.stack([text_bias, vision_bias], dim=0)
                    return {hf_key: merged}
            # Fallback: no buffered text bias, just return vision as-is
            return result


@MegatronModelBridge.register_bridge(
    source=_ERNIE45_VL_MOE_HF_CLASS_NAME,
    target=Ernie45VLModel,
    provider=Ernie45VLModelProvider,
    model_type="ernie4_5_vl_moe",
)
class Ernie45VLBridge(MegatronModelBridge):
    """
    Megatron Bridge for ERNIE 4.5 VL MoE Conditional Generation.

    This bridge handles the conversion between HuggingFace Ernie4_5_VLMoeForConditionalGeneration
    and Megatron-Core Ernie45VLModel formats, including weight mappings and
    configuration translation for this vision-language MoE model.

    Key architectural features handled:
    - Heterogeneous dual-pool MoE via ErnieMultiTypeMoE:
      * text_moe_layer: standard Megatron MoELayer (TP support)
      * vision_moe_layer: standard Megatron MoELayer (TP support)
    - Shared experts across modalities
    - 3D Multimodal RoPE (M-RoPE)
    - Variable-resolution vision resampler (spatial + temporal merging)
    - GQA with configurable query/KV heads
    - HF on-disk per-expert weights <-> Megatron per-expert SequentialMLP weights

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("baidu/ERNIE-4.5-VL-28B-A3B-Instruct")
        >>> provider = bridge.to_megatron_provider()
    """

    @staticmethod
    def _get_text_config(hf_config):
        """Extract the text/language config from either nested or flat HF config.

        The transformers-builtin ``Ernie4_5_VLMoeConfig`` (model_type=ernie4_5_vl_moe)
        uses a nested ``text_config`` sub-object, while the custom auto_map config
        ``Ernie4_5_VLMoEConfig`` (model_type=ernie4_5_moe_vl, e.g. the Thinking model)
        uses a flat layout where all LLM fields live directly on the top-level config.

        Returns the appropriate config object (nested text_config or the config itself).
        """
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            return text_config
        # Flat config: LLM fields are on hf_config itself
        return hf_config

    @staticmethod
    def _get_num_experts(text_config) -> int:
        """Extract the per-pool number of experts as an int.

        The nested config stores ``moe_num_experts`` as a plain int (e.g. 4),
        while the flat/Thinking config stores it as a list ``[64, 64]``
        (text pool, vision pool -- both values are always equal).
        """
        raw = getattr(text_config, "moe_num_experts", 4)
        if isinstance(raw, (list, tuple)):
            return raw[0]
        return raw

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> Ernie45VLModelProvider:
        """
        Create an Ernie45VLModelProvider from a HuggingFace pretrained model.

        Maps HuggingFace Ernie4_5_VLMoeConfig fields to Megatron provider parameters,
        including vision config, MoE settings, M-RoPE sections, and token IDs.

        Supports both nested config (transformers builtin, model_type=ernie4_5_vl_moe)
        and flat config (auto_map custom, model_type=ernie4_5_moe_vl).

        Args:
            hf_pretrained: HuggingFace pretrained VLM model.

        Returns:
            Ernie45VLModelProvider configured with the HF model's parameters.
        """
        hf_config = hf_pretrained.config
        text_config = self._get_text_config(hf_config)

        # Extract common config fields via base class utility
        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)

        # ERNIE 4.5 VL has moe_intermediate_size=[1536, 512] (list of 2 values
        # for text/vision expert pools). CONFIG_MAPPING would auto-map this to
        # moe_ffn_hidden_size, but that field expects a single int. Pop it here
        # and set it explicitly below with the text expert size.
        provider_kwargs.pop("moe_ffn_hidden_size", None)

        # Similarly, the attribute_map on the HF config aliases num_experts ->
        # moe_num_experts, so CONFIG_MAPPING might double-set num_moe_experts.
        # Pop MoE fields that we will set explicitly.
        provider_kwargs.pop("num_moe_experts", None)
        provider_kwargs.pop("moe_router_topk", None)

        provider = Ernie45VLModelProvider(**provider_kwargs)

        # --- Common LLM settings ---
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_qkv_bias = False
        provider.add_bias_linear = False
        provider.hidden_dropout = 0.0
        # ERNIE 4.5 VL language model uses interleaved RoPE (pairs even/odd dims),
        # unlike the LLaMA-style first-half/second-half split.
        provider.rotary_interleaved = True

        # Extract rope_theta: nested config uses rope_parameters dict, flat config
        # may use rope_scaling.mrope_section or a top-level rope_theta attribute.
        rope_params = getattr(text_config, "rope_parameters", None) or {}
        if isinstance(rope_params, dict):
            provider.rotary_base = rope_params.get("rope_theta", getattr(text_config, "rope_theta", 500000.0))
        else:
            provider.rotary_base = getattr(text_config, "rope_theta", 500000.0)

        # For VLMs, tie_word_embeddings lives on the top-level config, not text_config
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", True)

        # --- MoE settings ---
        num_experts = self._get_num_experts(text_config)
        provider.moe_ffn_hidden_size = text_config.moe_intermediate_size[0]  # 1536 (text experts)
        provider.num_moe_experts = num_experts
        provider.moe_router_topk = text_config.moe_k  # 6
        provider.moe_router_pre_softmax = False
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_router_load_balancing_type = "aux_loss"
        provider.moe_aux_loss_coeff = getattr(text_config, "router_aux_loss_coef", 0.001)
        # ERNIE 4.5 MoE uses sigmoid gating with expert bias for
        # aux-free load balancing:
        #   1. scores = sigmoid(logits)  -- per-expert independent scores
        #   2. scores_ = scores + e_score_correction_bias  -- biased for top-k selection
        #   3. weights = gather(scores, topk_indices)  -- unbiased sigmoid scores
        #   4. weights = weights / sum(weights)  -- normalize to sum=1
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_dtype = "fp32"
        provider.gradient_accumulation_fusion = False

        # Dual-pool MoE intermediate sizes
        provider.moe_intermediate_size = tuple(text_config.moe_intermediate_size)  # (1536, 512)

        # Shared experts: intermediate_size = moe_intermediate_size[0] * moe_num_shared_experts
        # e.g. 1536 * 2 = 3072
        moe_num_shared_experts = getattr(text_config, "moe_num_shared_experts", 2)
        provider.moe_shared_expert_intermediate_size = text_config.moe_intermediate_size[0] * moe_num_shared_experts

        # Determine which layers are dense vs MoE.
        # Nested config (Instruct): mlp_layer_types = ["dense", "sparse", ...]
        # Flat config (Thinking): moe_layer_start_index = [1, 1],
        #                         moe_layer_end_index = [29, 28]
        mlp_layer_types = getattr(text_config, "mlp_layer_types", None)
        if mlp_layer_types is not None:
            provider.moe_layer_freq = [0 if t == "dense" else 1 for t in mlp_layer_types]
        else:
            num_layers = text_config.num_hidden_layers
            moe_start = getattr(text_config, "moe_layer_start_index", None)
            if moe_start is not None:
                # moe_layer_start_index can be a list (per-pool) or int.
                # Take the first value: this is the first MoE layer index.
                start = moe_start[0] if isinstance(moe_start, (list, tuple)) else moe_start
                provider.moe_layer_freq = [0] * start + [1] * (num_layers - start)
            else:
                # Default: layer 0 dense, rest MoE
                provider.moe_layer_freq = [0] + [1] * (num_layers - 1)

        # --- VL-specific overrides ---
        provider.position_embedding_type = "mrope"
        provider.vision_config = hf_config.vision_config
        provider.hf_config = hf_config

        # M-RoPE section: [height, width, temporal] frequency allocation
        # Nested config: rope_parameters.mrope_section
        # Flat config: rope_scaling.mrope_section
        mrope_section = None
        if isinstance(rope_params, dict):
            mrope_section = rope_params.get("mrope_section")
        if mrope_section is None:
            rope_scaling = getattr(text_config, "rope_scaling", None) or {}
            if isinstance(rope_scaling, dict):
                mrope_section = rope_scaling.get("mrope_section")
        provider.mrope_section = mrope_section or [22, 22, 20]

        # Token IDs -- these live on the top-level config in both formats
        provider.image_start_token_id = getattr(hf_config, "image_start_token_id", 101304)
        provider.image_end_token_id = getattr(hf_config, "image_end_token_id", 101305)
        provider.image_token_id = getattr(hf_config, "image_token_id", getattr(hf_config, "im_patch_id", 100295))
        provider.video_start_token_id = getattr(hf_config, "video_start_token_id", 101306)
        provider.video_end_token_id = getattr(hf_config, "video_end_token_id", 101307)
        provider.video_token_id = getattr(hf_config, "video_token_id", 103367)

        return provider

    def stream_weights_megatron_to_hf(self, *args, **kwargs):
        """Override to clear the _ConcatBiasMapping export buffer before each run."""
        _ConcatBiasMapping.clear_export_buffer()
        return super().stream_weights_megatron_to_hf(*args, **kwargs)

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Return MegatronMappingRegistry with parameter mappings for ERNIE 4.5 VL MoE.

        Uses the HF **on-disk (safetensors)** key format, which differs from the
        in-memory ``state_dict()`` format due to HuggingFace's ``_checkpoint_conversion_mapping``.

        On-disk format:
        - No ``language_model.`` prefix: ``model.layers.*`` not ``model.language_model.layers.*``
        - Per-expert flat-indexed weights: ``experts.{j}.gate_proj.weight``
        - Text experts indices 0..N-1, vision experts indices N..2N-1
        - Single ``gate.weight`` (text router) and ``gate.weight_1`` (vision router)
        - Concatenated ``moe_statics.e_score_correction_bias`` for text+vision
        - ``model.vision_model.**`` not ``model.vision_tower.**``
        - Resampler: ``spatial_linear.0/2/3`` not ``spatial_linear.fc1/fc2/ln``
          (same for ``temporal_linear``)

        Returns:
            MegatronMappingRegistry with all parameter mappings.
        """
        # Get num_experts from the HF config (injected by the bridge dispatch).
        # Falls back to 4 for toy model / direct instantiation.
        num_experts = 4
        is_flat_config = False
        use_mg_vit = False
        if hasattr(self, "hf_config"):
            text_config = self._get_text_config(self.hf_config)
            num_experts = self._get_num_experts(text_config)
            # Detect flat config (Thinking / auto_map) vs nested config (Instruct).
            #
            # Simple ``not hasattr(hf_config, 'text_config')`` is unreliable because
            # ``_normalize_hf_config()`` in modeling_ernie45_vl.py mutates the config
            # object to add ``text_config = hf_config`` (self-reference) so that the
            # HF resampler can access ``config.text_config.hidden_size``.  After this
            # mutation ``hasattr`` returns True even for flat configs.
            #
            # Instead, detect flat config by checking whether ``text_config`` is absent
            # OR is a self-reference (points back to hf_config itself).  A genuinely
            # nested config has ``text_config`` as a distinct sub-object.
            text_cfg_attr = getattr(self.hf_config, "text_config", None)
            is_flat_config = (text_cfg_attr is None) or (text_cfg_attr is self.hf_config)

        # Check use_mg_vit: set externally on the bridge instance (like hf_config).
        # When True, the Megatron model uses MG-native ViT (TP-sharded weights).
        # When False (default), it uses HF-wrapped ViT (replicated weights).
        use_mg_vit = getattr(self, "use_mg_vit", False)

        # Determine on-disk vision key prefix based on config format.
        # Flat config (Thinking/auto_map): "vision_model.**"
        # Nested config (Instruct/transformers-builtin): "model.vision_model.**"
        vision_hf_prefix = "vision_model.**" if is_flat_config else "model.vision_model.**"
        # Per-param prefix (without glob suffix) for MG ViT block-level mappings.
        vision_hf_block_prefix = "vision_model" if is_flat_config else "model.vision_model"

        # =====================================================================
        # Simple 1:1 parameter mappings (AutoMapping detects parallelism)
        # =====================================================================
        param_mappings = {
            # =================================================================
            # Language Model: Embeddings and output
            # =================================================================
            "language_model.embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "language_model.decoder.final_layernorm.weight": "model.norm.weight",
            # =================================================================
            # Language Model: Self-attention (all layers)
            # input_layernorm is fused into TELayerNormColumnParallelLinear
            # =================================================================
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": (
                "model.layers.*.input_layernorm.weight"
            ),
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": (
                "model.layers.*.self_attn.o_proj.weight"
            ),
            # =================================================================
            # Dense MLP (layer 0): post_attention_layernorm fused into
            # TELayerNormColumnParallelLinear (linear_fc1)
            # =================================================================
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": (
                "model.layers.*.post_attention_layernorm.weight"
            ),
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": ("model.layers.*.mlp.down_proj.weight"),
            # =================================================================
            # MoE layers: pre_mlp_layernorm (separate, not fused)
            # =================================================================
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight": (
                "model.layers.*.post_attention_layernorm.weight"
            ),
            # =================================================================
            # Shared experts down projection
            # =================================================================
            "language_model.decoder.layers.*.mlp.shared_experts.linear_fc2.weight": (
                "model.layers.*.mlp.shared_experts.down_proj.weight"
            ),
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # =====================================================================
        # Vision encoder mappings
        #
        # Two modes depending on use_mg_vit:
        # - HF ViT (use_mg_vit=False): single ReplicatedMapping for all
        #   vision_tower weights (replicated across TP ranks).
        # - MG ViT (use_mg_vit=True): per-parameter TP-sharded mappings
        #   using ConcatenatedQKVMapping for fused QKV and AutoMapping
        #   for other TP-auto-detected parameters.
        #
        # On-disk key prefix differs by config format:
        #   Flat (Thinking/auto_map): "vision_model.*"
        #   Nested (Instruct/transformers): "model.vision_model.*"
        # =====================================================================
        if use_mg_vit:
            # MG-native ViT: TP-sharded weight mappings
            # Megatron key prefix: "vision_model.decoder.layers.*..."
            # HF on-disk prefix: "{vision_hf_block_prefix}.blocks.*..."
            vit_param_mappings = {
                # Attention: proj weight/bias (TP-sharded via AutoMapping)
                "vision_model.decoder.layers.*.self_attention.linear_proj.weight": (
                    f"{vision_hf_block_prefix}.blocks.*.attn.proj.weight"
                ),
                "vision_model.decoder.layers.*.self_attention.linear_proj.bias": (
                    f"{vision_hf_block_prefix}.blocks.*.attn.proj.bias"
                ),
                # MLP: fc1 weight/bias (TP column-parallel via AutoMapping)
                "vision_model.decoder.layers.*.mlp.linear_fc1.weight": (
                    f"{vision_hf_block_prefix}.blocks.*.mlp.fc1.weight"
                ),
                "vision_model.decoder.layers.*.mlp.linear_fc1.bias": (
                    f"{vision_hf_block_prefix}.blocks.*.mlp.fc1.bias"
                ),
                # MLP: fc2 weight/bias (TP row-parallel via AutoMapping)
                "vision_model.decoder.layers.*.mlp.linear_fc2.weight": (
                    f"{vision_hf_block_prefix}.blocks.*.mlp.fc2.weight"
                ),
                "vision_model.decoder.layers.*.mlp.linear_fc2.bias": (
                    f"{vision_hf_block_prefix}.blocks.*.mlp.fc2.bias"
                ),
                # LayerNorm: norm1 fused into linear_qkv (TELayerNormColumnParallelLinear)
                "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": (
                    f"{vision_hf_block_prefix}.blocks.*.norm1.weight"
                ),
                "vision_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_bias": (
                    f"{vision_hf_block_prefix}.blocks.*.norm1.bias"
                ),
                # LayerNorm: norm2 fused into linear_fc1 (TELayerNormColumnParallelLinear)
                "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": (
                    f"{vision_hf_block_prefix}.blocks.*.norm2.weight"
                ),
                "vision_model.decoder.layers.*.mlp.linear_fc1.layer_norm_bias": (
                    f"{vision_hf_block_prefix}.blocks.*.norm2.bias"
                ),
                # Final LayerNorm (post_layer_norm in TransformerBlock)
                "vision_model.decoder.final_layernorm.weight": (f"{vision_hf_block_prefix}.ln.weight"),
                "vision_model.decoder.final_layernorm.bias": (f"{vision_hf_block_prefix}.ln.bias"),
            }
            for mg_param, hf_param in vit_param_mappings.items():
                mapping_list.append(AutoMapping(megatron_param=mg_param, hf_param=hf_param))

            # Fused QKV: ConcatenatedQKVMapping handles the [Q|K|V] -> interleaved
            # GQA layout conversion, with TP-aware splitting.
            # ERNIE ViT uses fused attn.qkv.weight/bias on disk.
            mapping_list.extend(
                [
                    ConcatenatedQKVMapping(
                        megatron_param="vision_model.decoder.layers.*.self_attention.linear_qkv.weight",
                        hf_param=f"{vision_hf_block_prefix}.blocks.*.attn.qkv.weight",
                    ),
                    ConcatenatedQKVMapping(
                        megatron_param="vision_model.decoder.layers.*.self_attention.linear_qkv.bias",
                        hf_param=f"{vision_hf_block_prefix}.blocks.*.attn.qkv.bias",
                    ),
                ]
            )

            # Patch embedding: replicated across TP ranks (not TP-sharded).
            # ERNIE ViT PatchEmbed is nn.Linear with weight only (no bias).
            mapping_list.append(
                ReplicatedMapping(
                    megatron_param="vision_model.patch_embed.proj.**",
                    hf_param=f"{vision_hf_block_prefix}.patch_embed.proj.**",
                ),
            )
        else:
            # HF-wrapped ViT: all weights replicated across TP ranks.
            mapping_list.append(
                ReplicatedMapping(
                    megatron_param="vision_tower.**",
                    hf_param=vision_hf_prefix,
                ),
            )

        # =====================================================================
        # Special mappings requiring parameter transformation
        # =====================================================================
        mapping_list.extend(
            [
                # =============================================================
                # Resampler / projector: replicated across TP ranks
                #
                # On-disk keys use sequential indices (0, 2, 3) for
                # spatial_linear and temporal_linear sub-modules, while
                # Megatron/HF in-memory uses named attributes (fc1, fc2, ln).
                # HF's _checkpoint_conversion_mapping reverses this:
                #   spatial_linear.0 <-> spatial_linear.fc1
                #   spatial_linear.2 <-> spatial_linear.fc2
                #   spatial_linear.3 <-> spatial_linear.ln
                # (same for temporal_linear)
                #
                # We must use on-disk key format since SafeTensorsStateSource
                # returns raw on-disk keys (no HF renaming applied).
                # =============================================================
                # spatial_linear: fc1 -> 0, fc2 -> 2, ln -> 3
                ReplicatedMapping(
                    megatron_param="resampler_model.spatial_linear.fc1.**",
                    hf_param="model.resampler_model.spatial_linear.0.**",
                ),
                ReplicatedMapping(
                    megatron_param="resampler_model.spatial_linear.fc2.**",
                    hf_param="model.resampler_model.spatial_linear.2.**",
                ),
                ReplicatedMapping(
                    megatron_param="resampler_model.spatial_linear.ln.**",
                    hf_param="model.resampler_model.spatial_linear.3.**",
                ),
                # temporal_linear: fc1 -> 0, fc2 -> 2, ln -> 3
                ReplicatedMapping(
                    megatron_param="resampler_model.temporal_linear.fc1.**",
                    hf_param="model.resampler_model.temporal_linear.0.**",
                ),
                ReplicatedMapping(
                    megatron_param="resampler_model.temporal_linear.fc2.**",
                    hf_param="model.resampler_model.temporal_linear.2.**",
                ),
                ReplicatedMapping(
                    megatron_param="resampler_model.temporal_linear.ln.**",
                    hf_param="model.resampler_model.temporal_linear.3.**",
                ),
                # Remaining resampler params (no on-disk renaming)
                ReplicatedMapping(
                    megatron_param="resampler_model.mlp.**",
                    hf_param="model.resampler_model.mlp.**",
                ),
                ReplicatedMapping(
                    megatron_param="resampler_model.after_norm.**",
                    hf_param="model.resampler_model.after_norm.**",
                ),
                # =============================================================
                # Text MoE router weight (transposed on disk:
                # on-disk [hidden_size, num_experts] -> Megatron [num_experts, hidden_size])
                # =============================================================
                AutoMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.text_moe_layer.router.weight",
                    hf_param="model.layers.*.mlp.gate.weight",
                    permute_dims=(1, 0),
                ),
                # =============================================================
                # Vision MoE router weight (saved as gate.weight_1 on disk, also transposed)
                # =============================================================
                AutoMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.vision_moe_layer.router.weight",
                    hf_param="model.layers.*.mlp.gate.weight_1",
                    permute_dims=(1, 0),
                ),
                # =============================================================
                # QKV (fused Q, K, V into single QKV matrix)
                # =============================================================
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                # =============================================================
                # Dense MLP (layer 0): gate_proj + up_proj -> fused linear_fc1
                # =============================================================
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                # =============================================================
                # Text MoE experts (per-expert, using DeepSeek-style wildcards)
                # Megatron: local_experts.{j}.linear_fc1.weight
                # HF on-disk: experts.{j}.gate_proj.weight + experts.{j}.up_proj.weight
                # Direct index mapping (text expert j -> HF expert j)
                # =============================================================
                GatedMLPMapping(
                    megatron_param=(
                        "language_model.decoder.layers.*.mlp.text_moe_layer.experts.local_experts.*.linear_fc1.weight"
                    ),
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param=(
                        "language_model.decoder.layers.*.mlp.text_moe_layer.experts.local_experts.*.linear_fc2.weight"
                    ),
                    hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                # =============================================================
                # Vision MoE experts (per-expert, with expert index offset)
                # Megatron vision expert j -> HF expert (j + num_experts)
                # =============================================================
                _OffsetGatedMLPMapping(
                    megatron_param=(
                        "language_model.decoder.layers.*.mlp.vision_moe_layer"
                        ".experts.local_experts.*.linear_fc1.weight"
                    ),
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                    expert_offset=num_experts,
                ),
                _OffsetAutoMapping(
                    megatron_param=(
                        "language_model.decoder.layers.*.mlp.vision_moe_layer"
                        ".experts.local_experts.*.linear_fc2.weight"
                    ),
                    hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                    expert_offset=num_experts,
                ),
                # =============================================================
                # Shared experts: gate+up -> fused linear_fc1
                # =============================================================
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
                # =============================================================
                # Expert bias: concatenated [text; vision] on disk
                # Text router expert_bias -> first N entries
                # Vision router expert_bias -> last N entries
                # =============================================================
                _ConcatBiasMapping(
                    megatron_param=("language_model.decoder.layers.*.mlp.text_moe_layer.router.expert_bias"),
                    hf_param="model.layers.*.mlp.moe_statics.e_score_correction_bias",
                    slice_name="text",
                    num_experts=num_experts,
                ),
                _ConcatBiasMapping(
                    megatron_param=("language_model.decoder.layers.*.mlp.vision_moe_layer.router.expert_bias"),
                    hf_param="model.layers.*.mlp.moe_statics.e_score_correction_bias",
                    slice_name="vision",
                    num_experts=num_experts,
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
