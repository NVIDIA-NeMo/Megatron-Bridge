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

"""HF config and parameter mappings for the Muse Spark Open Hybrid integration."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import (
    HFSourcedWeightTuple,
    HFWeightTuple,
    MegatronModelBridge,
    WeightConversionTask,
)
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    ReplicatedMapping,
    RMSNorm2ZeroCenteredRMSNormMapping,
)
from megatron.bridge.models.hf_pretrained import PreTrainedCausalLM
from megatron.bridge.models.muse_glimmer.muse_glimmer_bridge import MuseGlimmerQKVGMapping
from megatron.bridge.models.muse_spark_open.modeling_muse_spark_open.model import MuseSparkModel
from megatron.bridge.models.muse_spark_open.muse_spark_open_config import (
    MuseSparkModelConfig,
    MuseSparkTransformerConfig,
)
from megatron.bridge.models.muse_spark_open.text_only import export_text_artifacts, select_text_pretrained


@MegatronModelBridge.register_bridge(
    source="MuseSparkForConditionalGeneration", target=MuseSparkModel, model_type="muse_spark"
)
class MuseSparkBridge(MegatronModelBridge):
    """Convert the supplied multimodal HF checkpoint to a native Hybrid stack."""

    MODEL_CONFIG_CLASS = MuseSparkModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True

    def text_only_pretrained(self, hf_pretrained: PreTrainedCausalLM) -> PreTrainedCausalLM:
        """Select language weights and a standalone HF text configuration."""
        return select_text_pretrained(hf_pretrained)

    def postprocess_hf_export_artifacts(self, path: Path) -> None:
        """Preserve the VL head computation in standalone HF text exports."""
        if not hasattr(self.hf_config, "text_config"):
            export_text_artifacts(path)

    @torch.no_grad()
    def stream_weights_megatron_to_hf(
        self,
        megatron_model: MuseSparkModel | list[MuseSparkModel],
        hf_pretrained: Any,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks: list[WeightConversionTask] | None = None,
        merge_adapter_weights: bool = True,
        weight_dtype: torch.dtype | None = None,
        with_megatron_names: bool = False,
    ) -> Iterable[HFWeightTuple | HFSourcedWeightTuple]:
        """Emit small tensors first so interleaved HF shards can release expert weights."""
        if conversion_tasks is not None and weight_dtype is not None:
            raise ValueError("weight_dtype cannot be combined with caller-supplied conversion_tasks")
        models = megatron_model if isinstance(megatron_model, list) else [megatron_model]
        if conversion_tasks is None:
            conversion_tasks = self.build_conversion_tasks(hf_pretrained, models, weight_dtype=weight_dtype)
        # Source shards mix each layer's large expert tensors with small tensors
        # from many other layers. Exporting those small tensors first prevents
        # the streaming writer from retaining most of the model until late layers.
        tasks = sorted(conversion_tasks, key=lambda task: bool(getattr(task.mapping, "is_grouped_export", False)))
        yield from super().stream_weights_megatron_to_hf(
            models,
            hf_pretrained,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=tasks,
            merge_adapter_weights=merge_adapter_weights,
            with_megatron_names=with_megatron_names,
        )

    def hf_config_to_model_config(self, hf_config: PretrainedConfig) -> MuseSparkModelConfig:
        """Read all architecture sizes from HF and expand schedules to Hybrid positions."""
        source = hf_config.to_dict()
        text_only = "text_config" not in source
        text = source if text_only else source["text_config"]
        num_layers = int(text["num_hidden_layers"])
        layer_types = text["layer_types"]
        rope_thetas = text["layer_rope_theta"]
        if any(len(text[key]) != num_layers for key in ("layer_types", "layer_rope_theta", "mlp_layer_types")):
            raise ValueError("All decoder schedules must match num_hidden_layers.")
        if set(text["mlp_layer_types"]) != {"sparse"} or text["num_shared_experts"]:
            raise ValueError("The initial Muse Spark Open bridge supports all-sparse layers with no shared experts.")
        if set(layer_types) - {"sliding_attention", "full_attention"}:
            raise ValueError("Unsupported attention layer type.")
        if text["hidden_act"] != "silu" or text["attention_bias"]:
            raise ValueError("Muse Spark Open requires bias-free attention and SwiGLU experts.")
        rope_base = float(text["rope_parameters"]["rope_theta"])
        if text["rope_parameters"]["rope_type"] != "default" or any(
            theta not in (0, rope_base) for theta in rope_thetas
        ):
            raise ValueError("Muse Spark Open currently supports one RoPE base with per-layer NoPE.")
        sliding_window = int(text["sliding_window"])
        if sliding_window <= 0:
            raise ValueError("sliding_window must be positive.")
        params_dtype = self.dtype_from_hf(hf_config, default=torch.bfloat16)
        transformer = MuseSparkTransformerConfig(
            num_layers=2 * num_layers,
            hidden_size=int(text["hidden_size"]),
            ffn_hidden_size=int(text["intermediate_size"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_query_groups=int(text["num_key_value_heads"]),
            kv_channels=int(text["head_dim"]),
            normalization="RMSNorm",
            layernorm_epsilon=float(text["rms_norm_eps"]),
            layernorm_zero_centered_gamma=True,
            post_norm_epsilon=float(text["post_norm_eps"]),
            residual_gate_temperature=float(text["residual_gate_temperature"]),
            gated_linear_unit=True,
            activation_func=F.silu,
            add_bias_linear=False,
            add_qkv_bias=False,
            hidden_dropout=0.0,
            attention_dropout=float(text["attention_dropout"]),
            # The attention composition supplies native parameter-free Q/K norms.
            qk_layernorm=False,
            attention_output_gate=True,
            softmax_scale=float(text["qk_scale_factor"]) / math.sqrt(text["head_dim"]),
            window_size=(sliding_window - 1, 0),
            window_attn_skip_freq=[value for layer in layer_types for value in (int(layer == "sliding_attention"), 0)],
            no_rope_freq=[value for theta in rope_thetas for value in (int(theta == 0), 0)],
            num_moe_experts=int(text["num_local_experts"]),
            moe_router_topk=int(text["num_experts_per_tok"]),
            moe_latent_size=int(text["moe_hidden_size"]),
            moe_ffn_hidden_size=int(text["moe_intermediate_size"]),
            moe_router_score_function="sigmoid",
            moe_router_dtype="fp32",
            moe_router_enable_expert_bias=True,
            moe_router_bias_update_rate=0.0,
            moe_router_topk_scaling_factor=float(text["routed_scaling_factor"]),
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0.0,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            expert_tensor_parallel_size=1,
            init_method_std=float(text["initializer_range"]),
            params_dtype=params_dtype,
            fp16=params_dtype == torch.float16,
            bf16=params_dtype == torch.bfloat16,
            transformer_impl="transformer_engine",
            cuda_graph_impl="none",
            output_multiplier=float(text["output_multiplier"]),
            final_logit_softcapping=float(text["final_logit_softcapping"]),
        )
        return MuseSparkModelConfig(
            transformer=transformer,
            hf_config=copy.deepcopy(source),
            hf_model_text_only=text_only,
            vocab_size=int(text["vocab_size"]),
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(int(text["vocab_size"])),
            should_pad_vocab=False,
            seq_length=int(text["max_position_embeddings"]),
            share_embeddings_and_output_weights=bool(source.get("tie_word_embeddings", text["tie_word_embeddings"])),
            hybrid_layer_pattern="*E" * num_layers,
            position_embedding_type="rope",
            rotary_base=rope_base,
        )

    @classmethod
    def megatron_to_hf_config(cls, model_config: MuseSparkModelConfig) -> dict[str, Any]:
        """Export current language settings and retain the original HF perception config."""
        source = copy.deepcopy(model_config.hf_config)
        transformer = model_config.transformer
        source["dtype"] = str(transformer.params_dtype).removeprefix("torch.")
        text = source if model_config.hf_model_text_only else source["text_config"]
        text.update(
            num_hidden_layers=transformer.num_layers // 2,
            hidden_size=transformer.hidden_size,
            intermediate_size=transformer.ffn_hidden_size,
            num_attention_heads=transformer.num_attention_heads,
            num_key_value_heads=transformer.num_query_groups,
            head_dim=transformer.kv_channels,
            vocab_size=model_config.vocab_size,
            max_position_embeddings=model_config.seq_length,
            tie_word_embeddings=model_config.share_embeddings_and_output_weights,
            num_local_experts=transformer.num_moe_experts,
            num_experts_per_tok=transformer.moe_router_topk,
            moe_hidden_size=transformer.moe_latent_size,
            moe_intermediate_size=transformer.moe_ffn_hidden_size,
            rms_norm_eps=transformer.layernorm_epsilon,
            post_norm_eps=transformer.post_norm_epsilon,
            residual_gate_temperature=transformer.residual_gate_temperature,
            output_multiplier=transformer.output_multiplier,
            final_logit_softcapping=transformer.final_logit_softcapping,
            routed_scaling_factor=transformer.moe_router_topk_scaling_factor,
            qk_scale_factor=transformer.softmax_scale * math.sqrt(transformer.kv_channels),
            sliding_window=transformer.window_size[0] + 1,
            layer_types=[
                "sliding_attention" if value else "full_attention" for value in transformer.window_attn_skip_freq[::2]
            ],
            layer_rope_theta=[0 if value else model_config.rotary_base for value in transformer.no_rope_freq[::2]],
            mlp_layer_types=["sparse"] * (transformer.num_layers // 2),
            rope_parameters={"rope_type": "default", "rope_theta": model_config.rotary_base},
        )
        source["tie_word_embeddings"] = model_config.share_embeddings_and_output_weights
        return source

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Map HF blocks to separate attention and MoE Hybrid positions."""
        text_only = not hasattr(self.hf_config, "text_config")
        text = self.hf_config if text_only else self.hf_config.text_config
        prefix = "model" if text_only else "model.language_model"
        mappings = [
            AutoMapping("embedding.word_embeddings.weight", f"{prefix}.embed_tokens.weight"),
            AutoMapping("output_layer.weight", "lm_head.weight"),
            RMSNorm2ZeroCenteredRMSNormMapping("decoder.final_norm.weight", f"{prefix}.norm.weight"),
        ]
        for index in range(text.num_hidden_layers):
            hf = f"{prefix}.layers.{index}"
            attention = f"decoder.layers.{2 * index}"
            moe = f"decoder.layers.{2 * index + 1}"
            mappings.extend(
                [
                    MuseGlimmerQKVGMapping(
                        f"{attention}.self_attention.linear_qkv.weight",
                        q=f"{hf}.self_attn.q_proj.weight",
                        k=f"{hf}.self_attn.k_proj.weight",
                        v=f"{hf}.self_attn.v_proj.weight",
                        gate=f"{hf}.self_attn.gate_proj.weight",
                    ),
                    AutoMapping(f"{attention}.self_attention.linear_proj.weight", f"{hf}.self_attn.o_proj.weight"),
                    ReplicatedMapping(f"{attention}.input_layernorm.weight", f"{hf}.input_layernorm.weight"),
                    ReplicatedMapping(f"{attention}.self_attn_bda.gate", f"{hf}.post_attention_residual_gate.gate"),
                    ReplicatedMapping(f"{moe}.pre_mlp_layernorm.weight", f"{hf}.pre_feedforward_layernorm.weight"),
                    ReplicatedMapping(f"{moe}.mlp_bda.gate", f"{hf}.post_feedforward_residual_gate.gate"),
                    ReplicatedMapping(f"{moe}.mlp_bda.post_norm.weight", f"{hf}.post_feedforward_layernorm.weight"),
                    ReplicatedMapping(f"{moe}.mlp.router.weight", f"{hf}.mlp.gate.weight"),
                    ReplicatedMapping(f"{moe}.mlp.router.expert_bias", f"{hf}.mlp.gate.e_score_correction_bias"),
                    ReplicatedMapping(f"{moe}.mlp.fc1_latent_proj.weight", f"{hf}.mlp.pre_expert_proj.weight"),
                    ReplicatedMapping(f"{moe}.mlp.pre_expert_norm.weight", f"{hf}.mlp.pre_expert_norm.weight"),
                    ReplicatedMapping(f"{moe}.mlp.fc2_latent_proj.weight", f"{hf}.mlp.post_expert_proj.weight"),
                    ReplicatedMapping(
                        f"{moe}.mlp.experts.post_expert_norm.weight", f"{hf}.mlp.experts.post_expert_norm.weight"
                    ),
                    FusedGatedExpertMapping(f"{moe}.mlp.experts.linear_fc1.weight*", f"{hf}.mlp.experts.gate_up_proj"),
                    FusedExpertMapping(f"{moe}.mlp.experts.linear_fc2.weight*", f"{hf}.mlp.experts.down_proj"),
                ]
            )
        if not text_only:
            for name in ("vision_tower", "audio_tower", "vision_adapter", "vision_projection"):
                mappings.append(ReplicatedMapping(f"{name}.**", f"model.{name}.**"))
        return MegatronMappingRegistry(*mappings)


@MegatronModelBridge.register_bridge(
    source="MuseSparkForCausalLM", target=MuseSparkModel, model_type="musespark1x_omni_text"
)
class MuseSparkTextBridge(MuseSparkBridge):
    """Use the same Hybrid decoder and mappings for the standalone language model."""
