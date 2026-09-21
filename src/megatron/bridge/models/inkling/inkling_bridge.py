# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Native Megatron text-backbone support for published Inkling checkpoints."""

import contextlib
from copy import copy, deepcopy
from dataclasses import dataclass, field

import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    FusedExpertMapping,
    QKVMapping,
    ReplicatedMapping,
    split_qkv_weights,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.inkling.inkling_mapping import (
    InklingExpertGatedMapping,
    InklingGatedMapping,
    InklingSharedExpertMapping,
)
from megatron.bridge.models.logit_dtype import logit_dtype_kwarg
from megatron.bridge.utils.common_utils import extract_expert_number_from_param


@dataclass
class InklingModelProvider(GPTModelProvider):
    """Architecture fields and native GPT construction for the Inkling text backbone."""

    inkling_layer_types: tuple[str, ...] = ()
    inkling_swa_num_attention_heads: int = 0
    inkling_swa_num_query_groups: int = 0
    inkling_swa_kv_channels: int = 0
    inkling_d_rel: int = 0
    inkling_rel_extent: int = 0
    inkling_sliding_window: int = 0
    inkling_log_scaling_n_floor: int | None = None
    inkling_log_scaling_alpha: float = 0.0
    inkling_conv_kernel_size: int = 0
    inkling_n_shared_experts: int = 0
    inkling_route_scale: float = 1.0
    inkling_logits_mup_width_multiplier: float = 1.0
    inkling_unpadded_vocab_size: int = 0
    inkling_hf_config: dict = field(default_factory=dict)

    def _get_num_floating_point_operations_with_runtime_stats(
        self,
        *,
        batch_size: int,
        seqlen_sum: int | None,
        seqlen_squared_sum: int | None,
        cross_seqlen_sum: int | None = None,
        cross_seqlen_product_sum: int | None = None,
    ) -> float:
        """Estimate full-training text FLOPs, including relative bias and shared experts.

        Follow Bridge's three-pass GEMM convention. Sliding attention uses the
        smaller of dense attention and the local window's upper bound.
        """
        del cross_seqlen_sum, cross_seqlen_product_sum
        tokens = batch_size * self.seq_length if seqlen_sum is None else seqlen_sum
        squared = tokens * (tokens / batch_size) if seqlen_squared_sum is None else seqlen_squared_sum
        flops = 6.0 * tokens * self.hidden_size * self.vocab_size
        for i, layer_type in enumerate(self.inkling_layer_types):
            sliding = layer_type == "hybrid_sliding"
            heads = self.inkling_swa_num_attention_heads if sliding else self.num_attention_heads
            groups = self.inkling_swa_num_query_groups if sliding else self.num_query_groups
            dim = self.inkling_swa_kv_channels if sliding else self.kv_channels
            extent = self.inkling_sliding_window if sliding else self.inkling_rel_extent
            projections = self.hidden_size * ((2 * heads + 2 * groups) * dim + heads * self.inkling_d_rel)
            relative = heads * self.inkling_d_rel * extent
            convolutions = 2 * self.inkling_conv_kernel_size * (self.hidden_size + groups * dim)
            if self.moe_layer_freq[i]:
                mlp = (
                    3
                    * self.hidden_size
                    * self.moe_ffn_hidden_size
                    * (self.moe_router_topk + self.inkling_n_shared_experts)
                )
                mlp += self.hidden_size * (self.num_moe_experts + self.inkling_n_shared_experts)
            else:
                mlp = 3 * self.hidden_size * self.ffn_hidden_size
            flops += 6.0 * tokens * (projections + relative + convolutions + mlp)
            attention_pairs = min(squared, tokens * extent) if sliding else squared
            flops += 12.0 * heads * dim * attention_pairs
        return flops

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> GPTModel:
        """Construct the native GPT subclass without changing shared provider code."""
        from megatron.bridge.models.inkling.modeling_inkling import InklingGPTModel, inkling_layer_spec

        if self.context_parallel_size != 1:
            raise ValueError("Inkling currently requires context_parallel_size=1")
        if self.hidden_dropout or self.attention_dropout:
            raise ValueError("Inkling requires zero hidden and attention dropout")
        if self.moe_router_enable_expert_bias:
            raise ValueError("Inkling preserves the checkpoint expert bias; automatic bias updates are unsupported")
        if self.mtp_num_layers:
            raise ValueError("Inkling's auxiliary MTP checkpoint is preserved but not trained")
        if self.vocab_size is None or self.should_pad_vocab:
            raise ValueError("Inkling requires the checkpoint's existing padded vocab_size")
        if self.cuda_graph_impl != "none":
            raise ValueError("Inkling CUDA graph execution has not been implemented")
        if pre_process is None:
            pre_process = is_vp_first_stage(vp_stage, self.virtual_pipeline_model_parallel_size) and is_pp_first_stage(
                self._pg_collection.pp
            )
        if post_process is None:
            post_process = is_vp_last_stage(vp_stage, self.virtual_pipeline_model_parallel_size) and is_pp_last_stage(
                self._pg_collection.pp
            )
        self._vp_stage = vp_stage
        context = torch.device("meta") if self.init_model_with_meta_device else contextlib.nullcontext()
        with context:
            return InklingGPTModel(
                self,
                transformer_layer_spec=inkling_layer_spec(self, vp_stage=vp_stage),
                vocab_size=self.vocab_size,
                max_sequence_length=self.seq_length,
                fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
                **logit_dtype_kwarg(GPTModel, self.logit_dtype),
                parallel_output=self.parallel_output,
                share_embeddings_and_output_weights=False,
                position_embedding_type="none",
                pre_process=pre_process,
                post_process=post_process,
                scatter_embedding_sequence_parallel=self.scatter_embedding_sequence_parallel,
                pg_collection=self._pg_collection,
                vp_stage=vp_stage,
            )


@MegatronModelBridge.register_bridge(
    source="InklingForConditionalGeneration",
    target=GPTModel,
    provider=InklingModelProvider,
    model_type="inkling_mm_model",
)
class InklingBridge(MegatronModelBridge):
    """Convert the published raw checkpoint; train only its text backbone.

    Audio, vision and MTP tensors remain in the source checkpoint and are copied
    unchanged on full export. A Transformers-materialized state dict uses a
    different parameter schema; use the native lazy checkpoint reader instead.
    """

    _HF_PASSTHROUGH_PREFIXES = ("model.audio.", "model.visual.", "model.mtp.")

    def provider_bridge(self, hf_pretrained) -> InklingModelProvider:
        """Read raw metadata before Transformers can reinterpret the expert width."""
        if getattr(hf_pretrained, "_model", None) is not None:
            raise ValueError("Inkling conversion requires a lazy raw checkpoint, not a materialized HF model")
        path = getattr(hf_pretrained, "model_name_or_path", None)
        if path:
            raw, _ = PretrainedConfig.get_config_dict(str(path), **hf_pretrained.init_kwargs)
        else:
            raw = hf_pretrained.config.to_dict()
        raw.pop("_commit_hash", None)
        if raw.get("quantization_config"):
            raise ValueError("Inkling conversion requires the unquantized checkpoint")
        text = raw["text_config"]
        required = {
            "use_embed_norm": True,
            "use_sconv": True,
            "shared_expert_sink": True,
            "use_gate_bias": True,
            "gate_activation": "sigmoid",
            "norm_after_topk": True,
            "use_global_scale": True,
            "q_bias": False,
            "o_bias": False,
            "final_logit_softcapping": None,
        }
        for name, expected in required.items():
            if text.get(name) != expected:
                raise ValueError(f"Unsupported Inkling text_config.{name}={text.get(name)!r}; expected {expected!r}")
        n_layers = text["num_hidden_layers"]
        local = set(text["local_layer_ids"])
        provider = InklingModelProvider(
            num_layers=n_layers,
            hidden_size=text["hidden_size"],
            num_attention_heads=text["num_attention_heads"],
            num_query_groups=text["num_key_value_heads"],
            kv_channels=text["head_dim"],
            ffn_hidden_size=text["dense_intermediate_size"],
            moe_ffn_hidden_size=text["intermediate_size"],
            num_moe_experts=text["n_routed_experts"],
            moe_router_topk=text["num_experts_per_tok"],
            moe_layer_freq=[int(i >= text["dense_mlp_idx"]) for i in range(n_layers)],
            vocab_size=text["vocab_size"],
            seq_length=text["model_max_length"],
            layernorm_epsilon=text["rms_norm_eps"],
            normalization="RMSNorm",
            gated_linear_unit=True,
            activation_func=torch.nn.functional.silu,
            add_bias_linear=False,
            add_qkv_bias=False,
            share_embeddings_and_output_weights=False,
            position_embedding_type="none",
            hidden_dropout=0.0,
            attention_dropout=0.0,
            params_dtype=torch.bfloat16,
            autocast_dtype=torch.bfloat16,
            bf16=True,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_router_dtype="fp32",
            moe_router_score_function="sigmoid",
            moe_router_load_balancing_type="none",
            moe_router_enable_expert_bias=False,
            moe_router_bias_update_rate=0.0,
            moe_aux_loss_coeff=0.0,
            moe_shared_expert_overlap=False,
            bias_activation_fusion=False,
            bias_dropout_fusion=False,
            apply_rope_fusion=False,
            qk_layernorm=True,
            inkling_layer_types=tuple("hybrid_sliding" if i in local else "hybrid" for i in range(n_layers)),
            inkling_swa_num_attention_heads=text["swa_num_attention_heads"],
            inkling_swa_num_query_groups=text["swa_num_key_value_heads"],
            inkling_swa_kv_channels=text["swa_head_dim"],
            inkling_d_rel=text["d_rel"],
            inkling_rel_extent=text["rel_extent"],
            inkling_sliding_window=text["sliding_window_size"],
            inkling_log_scaling_n_floor=text["log_scaling_n_floor"],
            inkling_log_scaling_alpha=text["log_scaling_alpha"],
            inkling_conv_kernel_size=text["sconv_kernel_size"],
            inkling_n_shared_experts=text["n_shared_experts"],
            inkling_route_scale=text["route_scale"],
            inkling_logits_mup_width_multiplier=text["logits_mup_width_multiplier"],
            inkling_unpadded_vocab_size=text["unpadded_vocab_size"],
            inkling_hf_config=deepcopy(raw),
        )
        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider: InklingModelProvider) -> dict:
        """Preserve multimodal metadata alongside the converted text checkpoint."""
        return deepcopy(provider.inkling_hf_config)

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Map published names, including the relative r projection for native LoRA."""
        m, h = "decoder.layers.*", "model.llm.layers.*"
        mappings = [
            AutoMapping("embedding.word_embeddings.weight", "model.llm.embed.weight"),
            AutoMapping("output_layer.weight", "model.llm.unembed.weight"),
            ReplicatedMapping("embedding_norm.weight", "model.llm.embed_norm.weight"),
            ReplicatedMapping("decoder.final_layernorm.weight", "model.llm.norm.weight"),
            ReplicatedMapping(f"{m}.input_layernorm.weight", f"{h}.attn_norm.weight"),
            ReplicatedMapping(f"{m}.pre_mlp_layernorm.weight", f"{h}.mlp_norm.weight"),
            ReplicatedMapping(f"{m}.attn_sconv.conv1d.weight", f"{h}.attn_sconv.weight"),
            ReplicatedMapping(f"{m}.mlp_sconv.conv1d.weight", f"{h}.mlp_sconv.weight"),
            ReplicatedMapping(f"{m}.self_attention.q_layernorm.weight", f"{h}.attn.q_norm.weight"),
            ReplicatedMapping(f"{m}.self_attention.k_layernorm.weight", f"{h}.attn.k_norm.weight"),
            ReplicatedMapping(f"{m}.self_attention.rel_logits_proj.proj", f"{h}.attn.rel_logits_proj.proj"),
            ColumnParallelMapping(f"{m}.self_attention.k_sconv.conv1d.weight", f"{h}.attn.k_sconv.weight"),
            ColumnParallelMapping(f"{m}.self_attention.v_sconv.conv1d.weight", f"{h}.attn.v_sconv.weight"),
            QKVMapping(
                f"{m}.self_attention.linear_qkv.weight",
                q=f"{h}.attn.wq_du.weight",
                k=f"{h}.attn.wk_dv.weight",
                v=f"{h}.attn.wv_dv.weight",
            ),
            AutoMapping(f"{m}.self_attention.linear_r.weight", f"{h}.attn.wr_du.weight"),
            AutoMapping(f"{m}.self_attention.linear_proj.weight", f"{h}.attn.wo_ud.weight"),
            InklingGatedMapping(f"{m}.mlp.linear_fc1.weight", f"{h}.mlp.w13_dn.weight"),
            AutoMapping(f"{m}.mlp.linear_fc2.weight", f"{h}.mlp.w2_md.weight"),
            ReplicatedMapping(f"{m}.mlp.global_scale", f"{h}.mlp.global_scale"),
            ReplicatedMapping(f"{m}.mlp.router.weight", f"{h}.mlp.gate.weight"),
            ReplicatedMapping(f"{m}.mlp.router.expert_bias", f"{h}.mlp.gate.bias"),
            ReplicatedMapping(f"{m}.mlp.router.global_scale", f"{h}.mlp.gate.global_scale"),
            InklingExpertGatedMapping(f"{m}.mlp.experts.linear_fc1.weight*", f"{h}.mlp.experts.w13_weight"),
            FusedExpertMapping(f"{m}.mlp.experts.linear_fc2.weight*", f"{h}.mlp.experts.w2_weight"),
            InklingSharedExpertMapping(
                f"{m}.mlp.shared_experts.*.linear_fc1.weight", f"{h}.mlp.shared_experts.shared_w13_weight"
            ),
            InklingSharedExpertMapping(
                f"{m}.mlp.shared_experts.*.linear_fc2.weight", f"{h}.mlp.shared_experts.shared_w2_weight"
            ),
        ]
        return MegatronMappingRegistry(*mappings)

    def _get_fused_adapter_linear_out_slices(
        self, megatron_model, base_hf_weight_names, linear_out_tensor, is_expert=False
    ) -> dict[str, torch.Tensor] | None:
        """Split packed QKV adapters using the published names and layer geometry."""
        roles = {name: name.rsplit(".", 2)[-2] for name in base_hf_weight_names}
        if set(roles.values()) != {"wq_du", "wk_dv", "wv_dv"}:
            return super()._get_fused_adapter_linear_out_slices(
                megatron_model, base_hf_weight_names, linear_out_tensor, is_expert=is_expert
            )
        model = megatron_model[0] if isinstance(megatron_model, list) else megatron_model
        config = copy(model.config)
        layer = int(base_hf_weight_names[0].split(".layers.", 1)[1].split(".", 1)[0])
        if config.inkling_layer_types[layer] == "hybrid_sliding":
            config.num_attention_heads = config.inkling_swa_num_attention_heads
            config.num_query_groups = config.inkling_swa_num_query_groups
            config.kv_channels = config.inkling_swa_kv_channels
        q, k, v = split_qkv_weights(config, linear_out_tensor, feature_dim=linear_out_tensor.shape[-1])
        projections = {"wq_du": q, "wk_dv": k, "wv_dv": v}
        return {name: projections[role] for name, role in roles.items()}

    def _merge_lora_adapter_weights(self, megatron_model, converted_weights_dict, adapter_weights):
        """Merge the packed attention adapter into each published Q/K/V tensor."""
        names = list(converted_weights_dict)
        if {name.rsplit(".", 2)[-2] for name in names} != {"wq_du", "wk_dv", "wv_dv"}:
            return super()._merge_lora_adapter_weights(megatron_model, converted_weights_dict, adapter_weights)
        (adapter,) = adapter_weights
        slices = self._get_fused_adapter_linear_out_slices(megatron_model, names, adapter.linear_out_weight.weight)
        return {
            name: self._merge_single_adapter_weight(
                tensor, adapter.alpha, adapter.dim, adapter.linear_in_weight.weight, slices[name]
            )
            for name, tensor in converted_weights_dict.items()
        }

    def _accumulate_grouped_export(
        self, task, converted_weights_dict, model_config, grouped_buffers, hf_state_dict, grouped_sources=None
    ):
        if not isinstance(task.mapping, InklingSharedExpertMapping):
            return super()._accumulate_grouped_export(
                task, converted_weights_dict, model_config, grouped_buffers, hf_state_dict, grouped_sources
            )
        # Shared experts are TP-sharded but replicated across EP, unlike routed experts.
        index = extract_expert_number_from_param(task.param_name)
        result = {}
        for name, tensor in converted_weights_dict.items():
            values = grouped_buffers.setdefault(name, {})
            values[index] = tensor
            if grouped_sources is not None:
                grouped_sources.setdefault(name, []).append(task.param_name)
            if len(values) == model_config.inkling_n_shared_experts:
                result[name] = torch.stack([values[i] for i in range(model_config.inkling_n_shared_experts)])
                del grouped_buffers[name]
        return result or None

    def stream_weights_megatron_to_hf(
        self,
        megatron_model,
        hf_pretrained,
        cpu=True,
        show_progress=True,
        conversion_tasks=None,
        *,
        merge_adapter_weights=True,
        weight_dtype=None,
        with_megatron_names=False,
    ):
        """Export trained text weights and copy untouched auxiliary checkpoint tensors."""
        yield from super().stream_weights_megatron_to_hf(
            megatron_model,
            hf_pretrained,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=conversion_tasks,
            merge_adapter_weights=merge_adapter_weights,
            weight_dtype=weight_dtype,
            with_megatron_names=with_megatron_names,
        )
        state = hf_pretrained.state
        for name in state.source.get_all_keys():
            if name.startswith(self._HF_PASSTHROUGH_PREFIXES):
                yield from HFWeightTuple(name, state[name]).iter_finalized(
                    cpu=cpu, megatron_param_names=() if with_megatron_names else None
                )
