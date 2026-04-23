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
Megatron Bridge for Gemma 4 VL (Vision-Language).

Extends the Gemma 4 text bridge to handle the full VLM checkpoint with
vision tower, multimodal embedder, and language model.

Weight prefixes in HF VLM checkpoint (after stripping outer ``model.``):
- ``language_model.layers.*``     → language model decoder layers
- ``language_model.embed_tokens`` → language model embedding
- ``language_model.norm``         → final layernorm
- ``vision_tower.*``              → HF vision encoder (replicated)
- ``embed_vision.*``              → multimodal projector (replicated)
"""

import re
from typing import Mapping

import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.transformers_compat import (
    rope_local_base_freq_from_hf,
    rope_theta_from_hf,
)
from megatron.bridge.models.gemma.gemma4_bridge import _Gemma4QKVMapping, _infer_attn_pattern
from megatron.bridge.models.gemma.gemma4_provider import Gemma4ModelProvider
from megatron.bridge.models.gemma_vl.gemma4_vl_provider import Gemma4VLModelProvider
from megatron.bridge.models.gemma_vl.modeling_gemma4_vl import Gemma4VLModel
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM


@MegatronModelBridge.register_bridge(
    source="Gemma4ForConditionalGeneration",
    target=Gemma4VLModel,
    provider=Gemma4VLModelProvider,
    model_type="gemma4_vl",
)
class Gemma4VLBridge(MegatronModelBridge):
    """Megatron Bridge for Gemma 4 Vision-Language models.

    Handles conversion between HuggingFace Gemma4ForConditionalGeneration and
    Megatron-Core Gemma4VLModel.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("google/gemma-4-26B-A4B")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> Gemma4VLModelProvider:
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        vision_config = hf_config.vision_config

        # Use base class helper for common config conversion from text_config
        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        provider = Gemma4VLModelProvider(**provider_kwargs)

        # === Gemma 4 text-specific features (same as Gemma4Bridge) ===
        provider.window_size = getattr(text_config, "sliding_window", 1024)

        # Dual RoPE bases
        provider.rotary_base = (
            rope_local_base_freq_from_hf(text_config),
            rope_theta_from_hf(text_config),
        )

        # QK norm
        head_dim = getattr(text_config, "head_dim", 256)
        provider.softmax_scale = 1.0
        provider.kv_channels = head_dim
        provider.qk_layernorm = True

        # Global attention overrides
        provider.global_head_dim = getattr(text_config, "global_head_dim", 512)
        provider.num_global_key_value_heads = getattr(text_config, "num_global_key_value_heads", 2)

        # Parse partial_rotary_factor
        rope_params = getattr(text_config, "rope_parameters", {})
        if isinstance(rope_params, dict):
            full_attn_rope = rope_params.get("full_attention", {})
            provider.global_rotary_percent = full_attn_rope.get("partial_rotary_factor", 0.25)

        # Sliding/global layer pattern
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types:
            provider.interleaved_attn_pattern = _infer_attn_pattern(layer_types)

        # MoE vs dense MLP configuration
        enable_moe = getattr(text_config, "enable_moe_block", False)
        if enable_moe:
            provider.num_moe_experts = getattr(text_config, "num_experts", None) or 128
            provider.moe_router_topk = getattr(text_config, "top_k_experts", None) or 8
            provider.moe_ffn_hidden_size = getattr(text_config, "moe_intermediate_size", None) or 704
            provider.moe_shared_expert_intermediate_size = getattr(text_config, "intermediate_size", 2112)
            provider.moe_shared_expert_overlap = False
            provider.moe_shared_expert_gate = False
            provider.moe_layer_freq = 1
        else:
            # Dense model: disable MoE, use standard FFN.
            # NOTE: gemma-4-e2b-it uses use_double_wide_mlp=True, which doubles
            # ffn_hidden_size for the last (num_hidden_layers - num_kv_shared_layers)
            # layers. MCore uses a single ffn_hidden_size for all layers, so we use
            # the larger size here; layers with the smaller intermediate_size will
            # have shape mismatches during weight loading (known limitation).
            intermediate_size = getattr(text_config, "intermediate_size", 6144)
            use_double_wide = getattr(text_config, "use_double_wide_mlp", False)
            # use_double_wide_mlp doubles ffn_hidden_size for layers >= num_kv_shared_layers.
            # MCore uses a single ffn_hidden_size; layers with the other size will fail.
            # Use the base (non-doubled) size so the first num_kv_shared_layers convert correctly.
            provider.ffn_hidden_size = intermediate_size
            provider.num_moe_experts = None
            # moe_layer_freq must be a list of zeros (one per layer) to get all-dense layers.
            # Using 0 as an integer would cause ZeroDivisionError in MCore's modulo check.
            provider.moe_layer_freq = [0] * provider.num_layers

        # Logit softcapping
        provider.final_logit_softcapping = getattr(text_config, "final_logit_softcapping", 30.0)

        # Override dtype and vocab settings
        provider.bf16 = True
        provider.params_dtype = torch.bfloat16
        provider.autocast_dtype = torch.bfloat16
        provider.make_vocab_size_divisible_by = 128

        # === VL-specific config ===
        provider.vision_config = vision_config
        provider.text_config = text_config
        provider.vision_soft_tokens_per_image = getattr(hf_config, "vision_soft_tokens_per_image", 280)

        # Token IDs
        provider.bos_token_id = getattr(hf_config, "bos_token_id", 2)
        provider.eos_token_id = getattr(hf_config, "eos_token_id", 1)
        provider.image_token_id = getattr(hf_config, "image_token_id", 258_880)
        provider.video_token_id = getattr(hf_config, "video_token_id", 258_884)

        return provider

    def _is_moe_model(self) -> bool:
        """Return True if the model uses MoE (enable_moe_block=True)."""
        text_config = getattr(getattr(self, "hf_config", None), "text_config", None)
        return bool(getattr(text_config, "enable_moe_block", False))

    def load_weights_hf_to_megatron(self, hf_pretrained, megatron_model, **kwargs):
        """Override to handle shape mismatches from variable-FFN models.

        gemma-4-e2b-it uses use_double_wide_mlp=True, which doubles the MLP
        size for layers >= num_kv_shared_layers. Since MCore uses a single
        ffn_hidden_size (set to intermediate_size, i.e. the smaller size),
        the double-wide layers produce shape mismatches.  Those layers are
        whitelisted so the conversion continues with a warning rather than
        crashing; their MLP weights will remain randomly initialized.
        """
        if not self._is_moe_model():
            text_config = getattr(getattr(self, "hf_config", None), "text_config", None)
            use_double_wide = getattr(text_config, "use_double_wide_mlp", False)
            if use_double_wide:
                kwargs.setdefault(
                    "allowed_mismatched_params",
                    ["*.mlp.linear_fc1.weight", "*.mlp.linear_fc2.weight"],
                )
        return super().load_weights_hf_to_megatron(hf_pretrained, megatron_model, **kwargs)

    def maybe_modify_converted_hf_weight(
        self,
        task,
        converted_weights_dict,
        hf_state_dict,
    ):
        """Un-fuse fused weights and drop synthesized keys on export.

        On import, ``maybe_modify_loaded_hf_weight`` applies two non-trivial fusions
        to the MoE layers to simplify the MCore forward pass:

        1. **Router fusion**: ``mg = hf * (scale * sqrt_hidden⁻¹ / pffl2)``
        2. **Shared-expert gate/up fusion**: ``mg = hf * (pffl / pffl2)``

        On export (Megatron → HF), this method inverts both fusions so the
        resulting HF weights exactly match the original checkpoint.  It also
        drops the synthesized ``v_proj`` key produced by ``QKVMapping.megatron_to_hf``
        for K=V global-attention layers where ``v_proj`` is absent in HF.
        """
        if not hf_state_dict:
            return converted_weights_dict

        if not self._is_moe_model():
            # Dense model: only need to drop absent keys (no fusions were applied)
            return {k: v for k, v in converted_weights_dict.items() if k in hf_state_dict}

        result = {}
        for hf_name, tensor in converted_weights_dict.items():
            # Drop synthesized v_proj (absent for K=V global-attention layers)
            if hf_name not in hf_state_dict:
                continue

            # ── Router weight inverse: mg = hf * (scale * hidden^-0.5 / pffl2)
            #                         hf = mg / (scale * hidden^-0.5 / pffl2)
            #                            = mg * pffl2 / (scale * hidden^-0.5)
            if hf_name.endswith("router.proj.weight"):
                layer_match = re.search(r"layers\.(\d+)\.", hf_name)
                if layer_match:
                    layer_idx = layer_match.group(1)
                    prefix = hf_name.rsplit("layers.", 1)[0]
                    scale_key = f"{prefix}layers.{layer_idx}.router.scale"
                    ln2_key = f"{prefix}layers.{layer_idx}.pre_feedforward_layernorm_2.weight"
                    if scale_key in hf_state_dict and ln2_key in hf_state_dict:
                        router_scale = hf_state_dict[scale_key].float().to(tensor.device)
                        ln2_weight = hf_state_dict[ln2_key].float().to(tensor.device)
                        hidden_size = tensor.shape[-1]
                        scalar_root_size = hidden_size ** -0.5
                        fusion_factor = router_scale * scalar_root_size / ln2_weight  # [hidden]
                        tensor = (tensor.float() / fusion_factor.unsqueeze(0)).to(tensor.dtype)

            # ── Shared-expert gate/up inverse: mg = hf * (pffl / pffl2)
            #                                  hf = mg * (pffl2 / pffl)
            elif (
                hf_name.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight"))
                and "experts" not in hf_name
            ):
                layer_match = re.search(r"layers\.(\d+)\.", hf_name)
                if layer_match:
                    layer_idx = layer_match.group(1)
                    prefix = hf_name.rsplit("layers.", 1)[0]
                    pffl_key = f"{prefix}layers.{layer_idx}.pre_feedforward_layernorm.weight"
                    pffl2_key = f"{prefix}layers.{layer_idx}.pre_feedforward_layernorm_2.weight"
                    if pffl_key in hf_state_dict and pffl2_key in hf_state_dict:
                        w_pffl = hf_state_dict[pffl_key].float().to(tensor.device)
                        w_pffl2 = hf_state_dict[pffl2_key].float().to(tensor.device)
                        correction = w_pffl / w_pffl2  # [hidden]
                        tensor = (tensor.float() / correction.unsqueeze(0)).to(tensor.dtype)

            result[hf_name] = tensor

        return result

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Handle special weight loading for Gemma 4 VLM.

        For MoE models: K=V synthesis, router weight fusion, shared expert
        pre-norm fusion.
        For dense models: K=V synthesis only (no router/shared-expert fusions).

        HF param names have ``model.language_model.`` prefix (raw safetensors
        keys include the outer ``model.`` from Gemma4ForConditionalGeneration).
        """
        # Handle K=V on global layers (applies to both MoE and dense)
        if isinstance(hf_param, dict) and "v" in hf_param:
            v_name = hf_param["v"]
            if v_name not in hf_state_dict:
                k_name = hf_param["k"]
                hf_weights = {}
                for role, name in hf_param.items():
                    if role == "v":
                        hf_weights[role] = hf_state_dict[k_name].clone()
                    else:
                        hf_weights[role] = hf_state_dict[name]
                return hf_weights

        # MoE-only fusions
        if self._is_moe_model():
            # Fuse pre-norm correction into shared expert gate/up weights
            if isinstance(hf_param, dict) and "gate" in hf_param:
                gate_name = hf_param["gate"]
                if "mlp.gate_proj" in gate_name:
                    return self._fuse_shared_expert_prenorm(hf_param, hf_state_dict)

            # Fuse router scaling into router.proj.weight
            if isinstance(hf_param, str) and hf_param.endswith("router.proj.weight"):
                return self._fuse_router_weight(hf_param, hf_state_dict)

        return super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)

    def _fuse_router_weight(
        self, hf_param: str, hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Fuse router preprocessing into projection weight (VLM version)."""
        proj_weight = hf_state_dict[hf_param]

        layer_match = re.search(r"layers\.(\d+)\.", hf_param)
        if layer_match is None:
            return proj_weight
        layer_idx = layer_match.group(1)

        # VLM prefix: language_model.layers.{idx}.router.*
        prefix = hf_param.rsplit("layers.", 1)[0]
        scale_key = f"{prefix}layers.{layer_idx}.router.scale"
        ln2_key = f"{prefix}layers.{layer_idx}.pre_feedforward_layernorm_2.weight"

        if scale_key not in hf_state_dict or ln2_key not in hf_state_dict:
            return proj_weight

        router_scale = hf_state_dict[scale_key].float()
        ln2_weight = hf_state_dict[ln2_key].float()
        hidden_size = proj_weight.shape[-1]
        scalar_root_size = hidden_size ** -0.5

        fusion_factor = router_scale * scalar_root_size / ln2_weight
        fused_weight = proj_weight.float() * fusion_factor.unsqueeze(0)
        return fused_weight.to(proj_weight.dtype)

    def _fuse_shared_expert_prenorm(
        self, hf_param: dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Fuse pre-norm correction into shared expert gate/up weights (VLM version)."""
        gate_name = hf_param["gate"]
        layer_match = re.search(r"layers\.(\d+)\.", gate_name)
        if layer_match is None:
            return {role: hf_state_dict[name] for role, name in hf_param.items()}

        layer_idx = layer_match.group(1)
        prefix = gate_name.rsplit("layers.", 1)[0]
        pffl_key = f"{prefix}layers.{layer_idx}.pre_feedforward_layernorm.weight"
        pffl2_key = f"{prefix}layers.{layer_idx}.pre_feedforward_layernorm_2.weight"

        if pffl_key not in hf_state_dict or pffl2_key not in hf_state_dict:
            return {role: hf_state_dict[name] for role, name in hf_param.items()}

        w_pffl = hf_state_dict[pffl_key].float()
        w_pffl2 = hf_state_dict[pffl2_key].float()
        correction = w_pffl / w_pffl2

        hf_weights = {}
        for role, name in hf_param.items():
            weight = hf_state_dict[name]
            fused = weight.float() * correction.unsqueeze(0)
            hf_weights[role] = fused.to(weight.dtype)
        return hf_weights

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Define parameter mappings for Gemma 4 VLM.

        HF VLM param names (raw safetensors keys include outer ``model.`` prefix):
        - ``model.language_model.layers.*`` → language model
        - ``model.vision_tower.*`` → vision encoder (replicated)
        - ``model.embed_vision.*`` → multimodal projector (replicated)

        Two MLP path variants depending on ``enable_moe_block``:
        - MoE (enable_moe_block=True): shared expert + routed expert + router
        - Dense (enable_moe_block=False): standard gated MLP (linear_fc1/fc2)
        """
        enable_moe = self._is_moe_model()

        # Common attention + norm mappings (both MoE and dense)
        param_mappings = {
            # === Embeddings ===
            "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",

            # === Per-layer attention ===
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": (
                "model.language_model.layers.*.input_layernorm.weight"
            ),
            "language_model.decoder.layers.*.input_layernorm.weight": (
                "model.language_model.layers.*.input_layernorm.weight"
            ),
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight": (
                "model.language_model.layers.*.self_attn.q_norm.weight"
            ),
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight": (
                "model.language_model.layers.*.self_attn.k_norm.weight"
            ),
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": (
                "model.language_model.layers.*.self_attn.o_proj.weight"
            ),
            "language_model.decoder.layers.*.self_attention.linear_proj.post_layernorm.weight": (
                "model.language_model.layers.*.post_attention_layernorm.weight"
            ),

            # === Post-feedforward layernorm (common to both MoE and dense) ===
            "language_model.decoder.layers.*.post_ffn_layernorm.weight": (
                "model.language_model.layers.*.post_feedforward_layernorm.weight"
            ),
        }

        if enable_moe:
            # === MoE-specific mappings ===
            param_mappings.update({
                # Pre-MLP layernorm (MoE pre-norm for routed experts)
                "language_model.decoder.layers.*.pre_mlp_layernorm.weight": (
                    "model.language_model.layers.*.pre_feedforward_layernorm_2.weight"
                ),
                # Dense MLP → Shared Expert fc2
                "language_model.decoder.layers.*.mlp.shared_experts.linear_fc2.weight": (
                    "model.language_model.layers.*.mlp.down_proj.weight"
                ),
                "language_model.decoder.layers.*.mlp.post_shared_expert_layernorm.weight": (
                    "model.language_model.layers.*.post_feedforward_layernorm_1.weight"
                ),
                # MoE Router
                "language_model.decoder.layers.*.mlp.router.weight": (
                    "model.language_model.layers.*.router.proj.weight"
                ),
            })
        else:
            # === Dense MLP mappings ===
            # NOTE: gemma-4-e2b-it uses use_double_wide_mlp=True, giving layers
            # 0..(num_kv_shared_layers-1) intermediate_size=6144 and layers
            # num_kv_shared_layers..(num_hidden_layers-1) intermediate_size=12288.
            # MCore uses a single ffn_hidden_size (set to the larger 12288).
            # Layers with the smaller MLP will encounter shape mismatches on load.
            param_mappings.update({
                # Pre-MLP layernorm: in TE's TELayerNormMLP the layernorm is fused
                # into linear_fc1 as layer_norm_weight (not a separate module).
                "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": (
                    "model.language_model.layers.*.pre_feedforward_layernorm.weight"
                ),
                # Dense MLP fc2
                "language_model.decoder.layers.*.mlp.linear_fc2.weight": (
                    "model.language_model.layers.*.mlp.down_proj.weight"
                ),
            })

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # === QKV: K=V tolerant mapping (common to MoE and dense) ===
        mapping_list.append(
            _Gemma4QKVMapping(
                megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.language_model.layers.*.self_attn.q_proj.weight",
                k="model.language_model.layers.*.self_attn.k_proj.weight",
                v="model.language_model.layers.*.self_attn.v_proj.weight",
            )
        )

        if enable_moe:
            mapping_list.extend([
                # === Dense MLP → Shared Expert gated FC1 ===
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.up_proj.weight",
                ),
                # === MoE Experts (fused format) ===
                FusedGatedExpertMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    hf_param="model.language_model.layers.*.experts.gate_up_proj",
                ),
                FusedExpertMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.language_model.layers.*.experts.down_proj",
                ),
                # === Router per-expert scaling (buffer) ===
                ReplicatedMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.router.per_expert_scale",
                    hf_param="model.language_model.layers.*.router.per_expert_scale",
                ),
                # === Router input scale (fused into router weight on import; stored as buffer) ===
                ReplicatedMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.router.scale",
                    hf_param="model.language_model.layers.*.router.scale",
                ),
                # === Dense/shared-expert pre-norm (fused into gate/up on import; stored as buffer) ===
                ReplicatedMapping(
                    megatron_param="language_model.decoder.layers.*.pffl_weight",
                    hf_param="model.language_model.layers.*.pre_feedforward_layernorm.weight",
                ),
                # === Post-MoE layernorm ===
                ReplicatedMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.post_moe_layernorm.weight",
                    hf_param="model.language_model.layers.*.post_feedforward_layernorm_2.weight",
                ),
            ])
        else:
            # === Dense MLP gated FC1 ===
            mapping_list.append(
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.up_proj.weight",
                )
            )
            # === Per-Layer Embeddings (PLE) — NOT IMPLEMENTED in MCore forward pass ===
            # PLE is controlled by config.hidden_size_per_layer_input (0 = disabled).
            # gemma-4-e2b-it has hidden_size_per_layer_input=256 (PLE active).
            # gemma-4-26B-A4B has hidden_size_per_layer_input=0 (PLE disabled, not used).
            #
            # What PLE does (HF: Gemma4TextDecoderLayer, after attention + FFN):
            #   1. embed_tokens_per_layer: a second vocab embedding table → low-dim vector per token
            #   2. per_layer_model_projection: projects main hidden → (num_layers × hidden_per_layer)
            #      and splits into per-layer slices — each layer gets its own conditioning vector
            #   3. Per decoder layer, after the FFN:
            #        gate = act_fn(per_layer_input_gate(hidden_states))   # project down
            #        gate = gate * per_layer_input[layer_idx]             # modulate with PLE vector
            #        gate = per_layer_projection(gate)                    # project back up
            #        gate = post_per_layer_input_norm(gate)               # RMSNorm
            #        hidden_states = hidden_states + gate                 # residual add
            #
            # To fully support gemma-4-e2b-it, PLE must be implemented in MCore:
            #   - Add embed_tokens_per_layer + per_layer_model_projection to the embedding module
            #   - Add per_layer_input_gate, per_layer_projection, post_per_layer_input_norm
            #     to each Gemma4TransformerLayer and call them in _forward_post_mlp
            #   - Wire the per-layer slice from the embedding into each layer's forward
            #
            # For now, the weights are stored as replicated buffers so roundtrip conversion
            # preserves them, but they are silently skipped during inference in MCore.
            # Inference output will differ from HF for any model that uses PLE.
            for hf_suffix, mg_suffix in [
                ("per_layer_input_gate.weight", "per_layer_input_gate.weight"),
                ("per_layer_projection.weight", "per_layer_projection.weight"),
                ("post_per_layer_input_norm.weight", "post_per_layer_input_norm.weight"),
            ]:
                mapping_list.append(
                    ReplicatedMapping(
                        megatron_param=f"language_model.decoder.layers.*.{mg_suffix}",
                        hf_param=f"model.language_model.layers.*.{hf_suffix}",
                    )
                )
            # Per-layer embedding table (shared across all layers, stored but not forwarded)
            mapping_list.append(
                ReplicatedMapping(
                    megatron_param="language_model.embedding.embed_tokens_per_layer.weight",
                    hf_param="model.language_model.embed_tokens_per_layer.weight",
                )
            )

        mapping_list.extend([
            # === Vision tower (replicated — all weights pass through) ===
            ReplicatedMapping(
                megatron_param="vision_tower.**",
                hf_param="model.vision_tower.**",
            ),
            # === Multimodal embedder (replicated) ===
            ReplicatedMapping(
                megatron_param="embed_vision.**",
                hf_param="model.embed_vision.**",
            ),
            # === Per-layer output scaling (buffer, common to both MoE and dense) ===
            ReplicatedMapping(
                megatron_param="language_model.decoder.layers.*.layer_scalar",
                hf_param="model.language_model.layers.*.layer_scalar",
            ),
        ])

        return MegatronMappingRegistry(*mapping_list)
