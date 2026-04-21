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

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni import NemotronOmniModel
from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import (
    NemotronNano3Bv3OmniModelProvider,
    NemotronNano12Bv2OmniModelProvider,
    NemotronOmniModelProvider,
)
from megatron.bridge.models.nemotron_vl.nemotron_vl_bridge import NemotronVLBridge


@MegatronModelBridge.register_bridge(source="NemotronH_Nano_VL_V2", target=NemotronOmniModel)
class NemotronOmniBridge(NemotronVLBridge):
    """Bridge for Nemotron Omni (VL + sound) models.

    Overrides VL's registration for the same source string. Imported after VL
    in models/__init__.py so this registration wins. Handles both VL-only and
    VL+sound checkpoints (backward compatible).
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> NemotronOmniModelProvider:
        """Create an omni provider from HuggingFace pretrained model.

        Always returns an omni provider (never a VL provider), even for
        VL-only checkpoints. When sound_config is absent, has_sound=False
        and the model is functionally identical to NemotronVLModel.
        """
        hf_config = hf_pretrained.config
        llm_config = hf_config.llm_config

        provider_kwargs = {
            "num_layers": llm_config.num_hidden_layers,
            "hidden_size": llm_config.hidden_size,
            "ffn_hidden_size": llm_config.intermediate_size,
            "num_attention_heads": llm_config.num_attention_heads,
            "init_method_std": llm_config.initializer_range,
            "layernorm_epsilon": getattr(llm_config, "layer_norm_epsilon", 1e-5),
            "make_vocab_size_divisible_by": self.make_vocab_size_divisible_by(llm_config.vocab_size),
            "share_embeddings_and_output_weights": getattr(llm_config, "tie_word_embeddings", False),
            "vocab_size": llm_config.vocab_size,
            "seq_length": llm_config.max_position_embeddings,
            "fp16": (self.dtype_from_hf(llm_config, default=__import__("torch").float32) == __import__("torch").float16),
            "bf16": (self.dtype_from_hf(llm_config, default=__import__("torch").float32) == __import__("torch").bfloat16),
            "params_dtype": self.dtype_from_hf(llm_config, default=__import__("torch").float32),
        }
        if hasattr(llm_config, "hybrid_override_pattern"):
            provider_kwargs["hybrid_override_pattern"] = llm_config.hybrid_override_pattern
        if hasattr(llm_config, "n_routed_experts"):
            provider_kwargs["num_moe_experts"] = llm_config.n_routed_experts
        if hasattr(hf_config, "projector_hidden_size"):
            provider_kwargs["vision_proj_ffn_hidden_size"] = hf_config.projector_hidden_size

        # Temporal compression config (conv3d tubelet embedding)
        # Check top-level first, then fall back to vision_config
        hf_config_vision = getattr(hf_config, "vision_config", None)
        vision_cfg_dict = (
            hf_config_vision if isinstance(hf_config_vision, dict)
            else vars(hf_config_vision) if hf_config_vision is not None and hasattr(hf_config_vision, "__dict__")
            else {}
        )
        provider_kwargs["video_temporal_patch_size"] = getattr(
            hf_config, "video_temporal_patch_size",
            vision_cfg_dict.get("video_temporal_patch_size", 1),
        )
        provider_kwargs["separate_video_embedder"] = getattr(
            hf_config, "separate_video_embedder",
            vision_cfg_dict.get("separate_video_embedder", False),
        )
        # Dynamic resolution: read from config so it doesn't rely solely on
        # the dataclass default (which can be wrong due to MRO).
        if getattr(hf_config, "dynamic_resolution", None) is not None:
            provider_kwargs["dynamic_resolution"] = hf_config.dynamic_resolution

        has_sound = hasattr(hf_config, "sound_config") and hf_config.sound_config is not None
        if has_sound:
            sc = hf_config.sound_config
            provider_kwargs["has_sound"] = True
            provider_kwargs["sound_model_type"] = getattr(sc, "model_type", "parakeet")
            provider_kwargs["sound_hidden_size"] = sc.hidden_size
            provider_kwargs["sound_projection_hidden_size"] = sc.projection_hidden_size
            provider_kwargs["sound_context_token_id"] = hf_config.sound_context_token_id
            provider_kwargs["sound_config"] = sc.to_dict() if hasattr(sc, "to_dict") else dict(sc)

        is_moe_model = getattr(llm_config, "n_routed_experts", None) is not None
        if is_moe_model:
            return NemotronNano3Bv3OmniModelProvider(**provider_kwargs)
        else:
            return NemotronNano12Bv2OmniModelProvider(**provider_kwargs)

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Inherit VL mappings and add sound projection + sound encoder mappings."""
        vl_registry = super().mapping_registry()
        mapping_list = list(vl_registry.mappings)

        # Sound projection (same MultimodalProjector structure as vision projection)
        for megatron_param, hf_param in {
            "llava_model.sound_projection.encoder.linear_fc1.layer_norm_weight": "sound_projection.norm.weight",
            "llava_model.sound_projection.encoder.linear_fc1.weight": "sound_projection.linear1.weight",
            "llava_model.sound_projection.encoder.linear_fc2.weight": "sound_projection.linear2.weight",
        }.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Sound encoder: explicit NeMo-to-HF key translations.
        # Megatron side uses NeMo-compatible naming (from vlm2's FastConformerModel),
        # HF checkpoint uses HF-standard naming. Bridge translates between them.
        # Megatron prefix: llava_model.sound_model.model.encoder.*
        #   (BridgeSoundEncoder.model = FastConformerModel, .encoder = FastConformerEncoder)
        # All sound encoder params are replicated (no TP sharding), same as RADIO ViT.
        _build_sound_encoder_mappings(mapping_list)

        return MegatronMappingRegistry(*mapping_list)


# Per-conformer-layer NeMo attr suffix -> HF checkpoint suffix translations.
# Verified against the actual HF checkpoint keys in the omni safetensors.
_SOUND_LAYER_MAPPINGS = {
    # Self-attention
    "self_attn.linear_q.weight": "self_attn.q_proj.weight",
    "self_attn.linear_k.weight": "self_attn.k_proj.weight",
    "self_attn.linear_v.weight": "self_attn.v_proj.weight",
    "self_attn.linear_out.weight": "self_attn.o_proj.weight",
    "self_attn.pos_bias_u": "self_attn.bias_u",
    "self_attn.pos_bias_v": "self_attn.bias_v",
    "self_attn.linear_pos.weight": "self_attn.relative_k_proj.weight",
    # Feed-forward 1 & 2 (HF uses same linear1/linear2 naming as NeMo)
    "feed_forward1.linear1.weight": "feed_forward1.linear1.weight",
    "feed_forward1.linear2.weight": "feed_forward1.linear2.weight",
    "feed_forward2.linear1.weight": "feed_forward2.linear1.weight",
    "feed_forward2.linear2.weight": "feed_forward2.linear2.weight",
    # Convolution module (HF uses "conv" prefix, same as Megatron)
    "conv.pointwise_conv1.weight": "conv.pointwise_conv1.weight",
    "conv.depthwise_conv.weight": "conv.depthwise_conv.weight",
    "conv.pointwise_conv2.weight": "conv.pointwise_conv2.weight",
    "conv.batch_norm.weight": "conv.norm.weight",
    "conv.batch_norm.bias": "conv.norm.bias",
    "conv.batch_norm.running_mean": "conv.norm.running_mean",
    "conv.batch_norm.running_var": "conv.norm.running_var",
    "conv.batch_norm.num_batches_tracked": "conv.norm.num_batches_tracked",
    # Layer norms
    "norm_feed_forward1.weight": "norm_feed_forward1.weight",
    "norm_feed_forward1.bias": "norm_feed_forward1.bias",
    "norm_self_att.weight": "norm_self_att.weight",
    "norm_self_att.bias": "norm_self_att.bias",
    "norm_conv.weight": "norm_conv.weight",
    "norm_conv.bias": "norm_conv.bias",
    "norm_feed_forward2.weight": "norm_feed_forward2.weight",
    "norm_feed_forward2.bias": "norm_feed_forward2.bias",
    "norm_out.weight": "norm_out.weight",
    "norm_out.bias": "norm_out.bias",
}

# Subsampling Conv2D indices with parameters (skipping ReLU layers).
# Both Megatron and HF use the same nn.Sequential layout with interleaved ReLU:
#   [0]=Conv2d, [1]=ReLU, [2]=DWConv, [3]=PWConv, [4]=ReLU, [5]=DWConv, [6]=PWConv, [7]=ReLU
# Indices that carry weight/bias: 0, 2, 3, 5, 6
_SUBSAMPLING_CONV_INDICES = [0, 2, 3, 5, 6]


def _build_sound_encoder_mappings(mapping_list):
    """Generate sound encoder ReplicatedMapping entries.

    Translates between NeMo-compatible naming (Megatron side, from vlm2's
    FastConformerModel) and HF checkpoint naming. Uses wildcard ``*`` for
    conformer layer indices, matching the pattern used by other Bridge modules.
    """
    meg_prefix = "llava_model.sound_model.model.encoder"
    hf_prefix = "sound_encoder.encoder"

    # Per-layer mappings (wildcard ``*`` matches any layer index at runtime)
    for nemo_suffix, hf_suffix in _SOUND_LAYER_MAPPINGS.items():
        mapping_list.append(ReplicatedMapping(
            megatron_param=f"{meg_prefix}.layers.*.{nemo_suffix}",
            hf_param=f"{hf_prefix}.layers.*.{hf_suffix}",
        ))

    # Subsampling conv mappings (same indices on both sides, skipping ReLU layers)
    for idx in _SUBSAMPLING_CONV_INDICES:
        for param in ["weight", "bias"]:
            mapping_list.append(ReplicatedMapping(
                megatron_param=f"{meg_prefix}.subsampling.conv.{idx}.{param}",
                hf_param=f"{hf_prefix}.subsampling.layers.{idx}.{param}",
            ))

    # Subsampling linear (Megatron "subsampling.out" -> HF "subsampling.linear")
    mapping_list.append(ReplicatedMapping(
        megatron_param=f"{meg_prefix}.subsampling.out.weight",
        hf_param=f"{hf_prefix}.subsampling.linear.weight",
    ))
    mapping_list.append(ReplicatedMapping(
        megatron_param=f"{meg_prefix}.subsampling.out.bias",
        hf_param=f"{hf_prefix}.subsampling.linear.bias",
    ))

    # Feature extractor buffers (feature_extractor.featurizer.fb, .window)
    # are NOT mapped. They don't exist in FastConformerModel -- they belong to
    # the separate FastConformerFeatureExtractor. Skipped on import, regenerated
    # from config on export.
