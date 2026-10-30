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


from typing import Any

import torch
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import ExaoneMoeForCausalLM, PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, register_bridge_implementation
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.exaone.exaone_moe.exaone_moe_config import ExaoneMoeModelConfig
from megatron.bridge.models.exaone.exaone_moe.exaone_moe_provider import (
    ExaoneMoeModelProvider,
    build_exaone_moe_layer_spec,
)


@MegatronModelBridge.register_bridge(
    source=ExaoneMoeForCausalLM,
    target=GPTModel,
    provider=ExaoneMoeModelProvider,
    model_type="exaone_moe",
)
class ExaoneMoeBridge(MegatronModelBridge):
    """Megatron Bridge for Hugging Face EXAONE MoE causal language models.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("LGAI-EXAONE/K-EXAONE-236B-A23B")
        >>> model_config = bridge.get_model_config()
    """

    MODEL_CONFIG_CLASS = ExaoneMoeModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True

    @classmethod
    def megatron_to_hf_config(cls, provider: ExaoneMoeModelProvider | ExaoneMoeModelConfig) -> dict:
        """Preserve EXAONE's distinction between physical and repeated MTP layers."""
        hf_config = super().megatron_to_hf_config(provider)
        mtp_num_speculative_steps = int(getattr(provider, "mtp_num_layers", 0) or 0)
        mtp_layer_types = getattr(provider, "mtp_layer_types", None)
        if mtp_layer_types is not None:
            physical_mtp_layers = len(mtp_layer_types)
        elif getattr(provider, "mtp_use_repeated_layer", False) and mtp_num_speculative_steps:
            physical_mtp_layers = 1
        else:
            physical_mtp_layers = mtp_num_speculative_steps
        hf_config["num_nextn_predict_layers"] = physical_mtp_layers
        hf_config["mtp_num_speculative_steps"] = mtp_num_speculative_steps
        return hf_config

    def dtype_from_hf(self, config: PretrainedConfig, default: torch.dtype | None = None) -> torch.dtype:
        """Read the model dtype from ``torch_dtype``, or from ``dtype`` when the config has no ``torch_dtype``."""
        if hasattr(config, "torch_dtype"):
            return super().dtype_from_hf(config, default=default)
        hf_dtype = getattr(config, "dtype", None)
        if isinstance(hf_dtype, torch.dtype):
            return hf_dtype
        if isinstance(hf_dtype, str):
            return self.dtype_from_str(hf_dtype)
        return torch.float32

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Convert a Hugging Face K-EXAONE config to builder config kwargs.

        Translates EXAONE's per-layer MLP, attention-window, SwiGLU-limit and MTP
        schedules and validates them against the decoder depth.
        """
        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)

        rope_parameters = hf_config.rope_parameters
        rope_theta = rope_parameters["rope_theta"]
        rope_scaling_factor = rope_parameters.get("factor")
        rope_scaling = rope_scaling_factor is not None

        mlp_layer_types = getattr(hf_config, "mlp_layer_types", None)
        if mlp_layer_types is not None:
            mlp_layer_types = list(mlp_layer_types)
            invalid_mlp_layer_types = set(mlp_layer_types) - {"dense", "sparse"}
            if invalid_mlp_layer_types:
                raise ValueError(f"Unsupported EXAONE MLP layer types: {sorted(invalid_mlp_layer_types)}.")
            moe_layer_freq = [int(layer_type == "sparse") for layer_type in mlp_layer_types]
        else:
            is_moe_layer = getattr(hf_config, "is_moe_layer", None)
            first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", None)
            if is_moe_layer is not None:
                moe_layer_freq = [int(value) for value in is_moe_layer]
            elif first_k_dense_replace is not None:
                moe_layer_freq = [0] * first_k_dense_replace + [1] * (
                    hf_config.num_hidden_layers - first_k_dense_replace
                )
            else:
                raise ValueError("EXAONE config must define mlp_layer_types, is_moe_layer, or first_k_dense_replace.")
        if len(moe_layer_freq) != hf_config.num_hidden_layers:
            raise ValueError(
                "EXAONE MoE layer schedule must contain one entry per decoder layer: "
                f"got {len(moe_layer_freq)}, expected {hf_config.num_hidden_layers}."
            )

        layer_types = getattr(hf_config, "layer_types", None)
        sliding_windows = getattr(hf_config, "sliding_windows", None)
        if sliding_windows is not None:
            sliding_windows = [int(value) for value in sliding_windows]
            if layer_types is None:
                layer_types = ["full_attention" if value == 0 else "sliding_attention" for value in sliding_windows]
        else:
            sliding_window = getattr(hf_config, "sliding_window", None)
            if sliding_window is None:
                raise ValueError("EXAONE config must define sliding_windows or sliding_window.")
            layer_types = (
                list(layer_types) if layer_types is not None else ["sliding_attention"] * hf_config.num_hidden_layers
            )
            sliding_windows = [
                int(sliding_window) if layer_type == "sliding_attention" else 0 for layer_type in layer_types
            ]

        layer_types = list(layer_types)
        if len(layer_types) != hf_config.num_hidden_layers:
            raise ValueError(
                "EXAONE attention layer_types must contain one entry per decoder layer: "
                f"got {len(layer_types)}, expected {hf_config.num_hidden_layers}."
            )
        if len(sliding_windows) != hf_config.num_hidden_layers:
            raise ValueError(
                "EXAONE sliding_windows must contain one entry per decoder layer: "
                f"got {len(sliding_windows)}, expected {hf_config.num_hidden_layers}."
            )

        window_attn_skip_freq, no_rope_freq = [], []
        for layer_idx, (layer_type, layer_window) in enumerate(zip(layer_types, sliding_windows)):
            if layer_type not in {"full_attention", "sliding_attention"}:
                raise ValueError(f"Unsupported EXAONE attention layer type: {layer_type!r}.")
            if layer_window < 0:
                raise ValueError(f"EXAONE sliding_windows[{layer_idx}] must be non-negative.")
            is_sliding = layer_type == "sliding_attention"
            if not is_sliding and layer_window != 0:
                raise ValueError(f"EXAONE full-attention layer {layer_idx} must use sliding_windows[{layer_idx}]=0.")
            if is_sliding and layer_window == 0:
                raise ValueError(f"EXAONE sliding-attention layer {layer_idx} must use a positive window.")
            no_rope_freq.append(0 if is_sliding else 1)
            window_attn_skip_freq.append(1 if is_sliding else 0)

        positive_windows = [value for value in sliding_windows or [] if value > 0]
        if positive_windows:
            window_size = (positive_windows[0] - 1, 0)
        else:
            window_size = None

        swiglu_limits = getattr(hf_config, "swiglu_limits", None)
        if swiglu_limits is not None:
            swiglu_limits = [float(value) for value in swiglu_limits]
            if len(swiglu_limits) != hf_config.num_hidden_layers:
                raise ValueError(
                    "EXAONE swiglu_limits must contain one entry per decoder layer: "
                    f"got {len(swiglu_limits)}, expected {hf_config.num_hidden_layers}."
                )
            if any(value < 0.0 for value in swiglu_limits):
                raise ValueError("EXAONE swiglu_limits values must be non-negative.")

        mtp_layer_types = getattr(hf_config, "mtp_layer_types", None)
        if mtp_layer_types is not None:
            mtp_layer_types = list(mtp_layer_types)
            physical_mtp_layers = len(mtp_layer_types)
            mtp_sliding_windows = getattr(hf_config, "mtp_sliding_windows", None)
            if mtp_sliding_windows is None:
                raise ValueError("EXAONE config with mtp_layer_types must define mtp_sliding_windows.")
            mtp_sliding_windows = [int(value) for value in mtp_sliding_windows]
            if len(mtp_sliding_windows) != physical_mtp_layers:
                raise ValueError(
                    "EXAONE mtp_sliding_windows must contain one entry per MTP layer: "
                    f"got {len(mtp_sliding_windows)}, expected {physical_mtp_layers}."
                )
        else:
            physical_mtp_layers = int(getattr(hf_config, "num_nextn_predict_layers", 0) or 0)
            mtp_layer_types = ["full_attention"] * physical_mtp_layers
            mtp_sliding_windows = [0] * physical_mtp_layers

        for layer_idx, (layer_type, layer_window) in enumerate(zip(mtp_layer_types, mtp_sliding_windows)):
            if layer_type not in {"full_attention", "sliding_attention"}:
                raise ValueError(f"Unsupported EXAONE MTP attention layer type: {layer_type!r}.")
            if layer_window < 0:
                raise ValueError(f"EXAONE mtp_sliding_windows[{layer_idx}] must be non-negative.")
            if layer_type == "full_attention" and layer_window != 0:
                raise ValueError(f"EXAONE full-attention MTP layer {layer_idx} must use a zero sliding window.")
            if layer_type == "sliding_attention" and layer_window == 0:
                raise ValueError(f"EXAONE sliding-attention MTP layer {layer_idx} must use a positive window.")

        mtp_num_speculative_steps = int(getattr(hf_config, "mtp_num_speculative_steps", physical_mtp_layers) or 0)
        mtp_use_repeated_layer = bool(getattr(hf_config, "mtp_share_layers", False))
        if mtp_num_speculative_steps > physical_mtp_layers:
            if physical_mtp_layers != 1:
                raise ValueError("EXAONE repeated MTP requires exactly one physical next-token-prediction layer.")
            mtp_use_repeated_layer = True
        mtp_num_layers = mtp_num_speculative_steps if mtp_use_repeated_layer else physical_mtp_layers

        topk_method = getattr(hf_config, "topk_method", None)
        model_dtype = self.dtype_from_hf(hf_config, default=torch.float32)

        config_kwargs.update(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            moe_ffn_hidden_size=hf_config.moe_intermediate_size,  # Maps to moe_intermediate_size in HF
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=hf_config.num_key_value_heads,
            kv_channels=getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads),
            num_moe_experts=hf_config.num_experts,
            moe_router_topk=hf_config.num_experts_per_tok,  # Maps to num_experts_per_tok in HF
            moe_router_num_groups=hf_config.n_group,
            moe_router_group_topk=hf_config.topk_group,
            moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
            moe_router_score_function=getattr(hf_config, "scoring_func", "sigmoid"),
            init_method_std=hf_config.initializer_range,
            layernorm_epsilon=hf_config.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),
            rotary_base=rope_theta,
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            vocab_size=hf_config.vocab_size,
            seq_length=hf_config.max_position_embeddings,
            attention_dropout=hf_config.attention_dropout,
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            qk_layernorm=True,
            moe_grouped_gemm=True,
            moe_shared_expert_intermediate_size=hf_config.num_shared_experts * hf_config.moe_intermediate_size,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            window_attn_skip_freq=window_attn_skip_freq,
            window_size=window_size,
            no_rope_freq=no_rope_freq,
            moe_layer_freq=moe_layer_freq,
            layer_types=list(layer_types) if layer_types else None,
            sliding_windows=sliding_windows,
            swiglu_limits=swiglu_limits,
            # MTP
            mtp_num_layers=mtp_num_layers or None,
            mtp_loss_scaling_factor=getattr(hf_config, "mtp_loss_scaling_factor", 0.1),
            mtp_use_repeated_layer=mtp_use_repeated_layer,
            mtp_layer_types=mtp_layer_types,
            mtp_sliding_windows=mtp_sliding_windows,
            # EXAONE settings that differ from the builder config's defaults
            transformer_layer_spec=build_exaone_moe_layer_spec,
            normalization="RMSNorm",
            activation_func=F.silu,
            position_embedding_type="rope",
            add_bias_linear=False,
            hidden_dropout=0.0,
            persist_layer_norm=True,
            attention_softmax_in_fp32=True,
            masked_softmax_fusion=False,
            gradient_accumulation_fusion=False,
            fp8_recipe="tensorwise",
            moe_router_pre_softmax=True,
            moe_token_dispatcher_type="alltoall",
            moe_router_load_balancing_type="global_aux_loss",
            moe_shared_expert_overlap=True,
            moe_router_dtype="fp32",
            moe_aux_loss_coeff=1e-2,
            moe_z_loss_coeff=1e-3,
            moe_permute_fusion=True,
            moe_router_enable_expert_bias=True,
        )
        if topk_method == "noaux_tc":
            config_kwargs.update(moe_router_load_balancing_type="none", moe_aux_loss_coeff=0.0)

        return config_kwargs

    def hf_config_to_provider_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Adapt the canonical builder mapping to the deprecated provider path."""
        return self.hf_config_to_model_config_kwargs(hf_config)

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from HF to Megatron format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        # Dictionary maps Megatron parameter names -> HF parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        hf_config = getattr(self, "hf_config", None)
        share_embeddings_and_output_weights = getattr(
            hf_config,
            "share_embeddings_and_output_weights",
            getattr(hf_config, "tie_word_embeddings", False),
        )
        output_layer_hf_param = (
            "model.embed_tokens.weight" if share_embeddings_and_output_weights else "lm_head.weight"
        )
        mtp_layer_types = getattr(hf_config, "mtp_layer_types", None)
        if mtp_layer_types is not None:
            mtp_num_layers = len(mtp_layer_types)
        else:
            mtp_num_layers = getattr(hf_config, "num_nextn_predict_layers", 0) or 0

        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.e_score_correction_bias",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
        }

        if hf_config is None or mtp_num_layers > 0:
            param_mappings.update(
                {
                    "mtp.layers.*.mtp_model_layer.self_attention.q_layernorm.weight": "mtp.layers.*.self_attn.q_norm.weight",
                    "mtp.layers.*.mtp_model_layer.self_attention.k_layernorm.weight": "mtp.layers.*.self_attn.k_norm.weight",
                    "mtp.layers.*.mtp_model_layer.self_attention.linear_proj.weight": "mtp.layers.*.self_attn.o_proj.weight",
                    "mtp.layers.*.mtp_model_layer.self_attention.linear_qkv.layer_norm_weight": "mtp.layers.*.input_layernorm.weight",
                    "mtp.layers.*.mtp_model_layer.pre_mlp_layernorm.weight": "mtp.layers.*.post_attention_layernorm.weight",
                    "mtp.layers.*.mtp_model_layer.mlp.linear_fc1.layer_norm_weight": "mtp.layers.*.post_attention_layernorm.weight",
                    "mtp.layers.0.final_layernorm.weight": "mtp.norm.weight",
                    "mtp.layers.0.eh_proj.weight": "mtp.fc.weight",
                    "mtp.layers.0.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
                    "mtp.layers.0.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
                }
            )

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        mapping_list.append(AutoMapping(hf_param=output_layer_hf_param, megatron_param="output_layer.weight"))

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                # K-EXAONE checkpoints store per-expert projections; Transformers fuses them only in memory.
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                # Shared Experts
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
                    hf_param="model.layers.*.mlp.shared_experts.down_proj.weight",
                ),
                # Dense MLP
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc2.weight",
                    hf_param="model.layers.*.mlp.down_proj.weight",
                ),
            ]
        )

        if hf_config is None or mtp_num_layers > 0:
            mapping_list.extend(
                [
                    # MTP
                    AutoMapping(
                        hf_param="mtp.layers.*.mlp.down_proj.weight",
                        megatron_param="mtp.layers.*.mtp_model_layer.mlp.linear_fc2.weight",
                    ),
                    GatedMLPMapping(
                        gate="mtp.layers.*.mlp.gate_proj.weight",
                        up="mtp.layers.*.mlp.up_proj.weight",
                        megatron_param="mtp.layers.*.mtp_model_layer.mlp.linear_fc1.weight",
                    ),
                    QKVMapping(
                        q="mtp.layers.*.self_attn.q_proj.weight",
                        k="mtp.layers.*.self_attn.k_proj.weight",
                        v="mtp.layers.*.self_attn.v_proj.weight",
                        megatron_param="mtp.layers.*.mtp_model_layer.self_attention.linear_qkv.weight",
                    ),
                ]
            )

        return MegatronMappingRegistry(*mapping_list)


# Some K-EXAONE configs use the upstream architecture spelling
# ``ExaoneMoEForCausalLM``, while Transformers exposes ``ExaoneMoeForCausalLM``.
register_bridge_implementation(source="ExaoneMoEForCausalLM", target=GPTModel, bridge_class=ExaoneMoeBridge)
