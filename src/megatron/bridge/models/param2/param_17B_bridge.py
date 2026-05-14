import contextlib
import fnmatch
from typing import Dict, List, Mapping, Optional

import torch
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping
from megatron.bridge.models.param2.param_17B_provider import Param2ModelProvider


@MegatronModelBridge.register_bridge(
    source="Param2MoEForCausalLM",
    target=GPTModel,
)
class Param2Bridge(MegatronModelBridge):
    """Megatron Bridge for BharatGen Param2MoE Causal LM."""

    ADDITIONAL_FILE_PATTERNS = [
        "configuration_param2moe.py",
        "modeling_param2moe.py",
        "parsers.py",
        "chat_template.jinja",
    ]

    NUM_LAYERS = 21
    FIRST_K_DENSE_REPLACE = 1
    NUM_EXPERTS = 64

    NUM_ATTENTION_HEADS = 32
    NUM_QUERY_GROUPS = 8
    HEAD_DIM = 64

    def provider_bridge(self, hf_pretrained) -> Param2ModelProvider:
        cfg = hf_pretrained.config

        provider = Param2ModelProvider(
            num_layers=int(cfg.num_hidden_layers),
            hidden_size=int(cfg.hidden_size),
            ffn_hidden_size=int(cfg.intermediate_size),
            num_attention_heads=int(cfg.num_attention_heads),
            num_query_groups=int(cfg.num_key_value_heads),
            kv_channels=int(
                getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
            ),
            vocab_size=int(cfg.vocab_size),
            seq_length=int(cfg.max_position_embeddings),
            max_position_embeddings=int(cfg.max_position_embeddings),
            share_embeddings_and_output_weights=bool(
                getattr(cfg, "tie_word_embeddings", True)
            ),
            normalization="RMSNorm",
            gated_linear_unit=True,
            add_bias_linear=False,
            add_qkv_bias=False,
            qk_layernorm=True,
            layernorm_epsilon=float(cfg.rms_norm_eps),
            rotary_base=float(cfg.rope_theta),
            hidden_dropout=float(getattr(cfg, "embedding_dropout", 0.0)),
            attention_dropout=float(getattr(cfg, "attention_dropout", 0.0)),
            autocast_dtype=torch.bfloat16,
            params_dtype=torch.bfloat16,
            bf16=True,
            fp16=False,
            # MoE params
            num_moe_experts=int(cfg.num_experts),
            moe_router_topk=int(cfg.num_experts_per_tok),
            moe_ffn_hidden_size=int(cfg.moe_intermediate_size),
            moe_shared_expert_intermediate_size=int(cfg.moe_shared_expert_intermediate_size),
            num_shared_experts=int(cfg.num_shared_experts),
            first_k_dense_replace=int(cfg.first_k_dense_replace),
            norm_topk_prob=bool(getattr(cfg, "norm_topk_prob", True)),
            moe_router_num_groups=int(getattr(cfg, "n_group", 1)),
            moe_router_group_topk=int(getattr(cfg, "topk_group", 1)),
            moe_router_topk_scaling_factor=float(getattr(cfg, "routed_scaling_factor", 1.0)),
            moe_router_score_function=str(getattr(cfg, "score_function", "sigmoid")),
            moe_router_dtype=str(getattr(cfg, "router_dtype", "fp32")),
            moe_router_enable_expert_bias=bool(
                getattr(cfg, "moe_router_enable_expert_bias", False)
            ),
            # Misc
            hidden_act=str(getattr(cfg, "hidden_act", "silu")),
            use_rmsnorm=bool(getattr(cfg, "use_rmsnorm", True)),
            use_qk_norm=bool(getattr(cfg, "use_qk_norm", True)),
            use_qkv_bias=bool(getattr(cfg, "use_qkv_bias", False)),
            use_bias=bool(getattr(cfg, "use_bias", False)),
            embedding_dropout=float(getattr(cfg, "embedding_dropout", 0.0)),
            output_dropout=float(getattr(cfg, "output_dropout", 0.0)),
            output_router_logits=bool(getattr(cfg, "output_router_logits", False)),
            pad_token_id=int(getattr(cfg, "pad_token_id", 0)),
            eos_token_id=int(getattr(cfg, "eos_token_id", 3)),
            partial_rotary_factor=float(getattr(cfg, "partial_rotary_factor", 1.0)),
            rope_scaling=getattr(cfg, "rope_scaling", None),
            max_window_layers=int(getattr(cfg, "max_window_layers", cfg.num_hidden_layers - 1)),
            num_nextn_predict_layers=int(getattr(cfg, "num_nextn_predict_layers", 0)),
            mtp_loss_scaling_factor=float(getattr(cfg, "mtp_loss_scaling_factor", 0.0)),
            router_dtype=str(getattr(cfg, "router_dtype", "fp32")),
            score_function=str(getattr(cfg, "score_function", "sigmoid")),
            routed_scaling_factor=float(getattr(cfg, "routed_scaling_factor", 1.0)),
            n_group=int(getattr(cfg, "n_group", 1)),
            topk_group=int(getattr(cfg, "topk_group", 1)),
        )

        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        hf_config = super().megatron_to_hf_config(provider)

        hf_config.update(
            {
                "architectures": ["Param2MoEForCausalLM"],
                "model_type": "param2moe",
                "hidden_act": getattr(provider, "hidden_act", "silu"),
                "tie_word_embeddings": bool(
                    getattr(provider, "share_embeddings_and_output_weights", True)
                ),
                "use_rmsnorm": bool(getattr(provider, "use_rmsnorm", True)),
                "use_qk_norm": bool(getattr(provider, "use_qk_norm", True)),
                "use_qkv_bias": bool(getattr(provider, "use_qkv_bias", False)),
                "use_bias": bool(getattr(provider, "use_bias", False)),
                "embedding_dropout": float(getattr(provider, "embedding_dropout", 0.0)),
                "attention_dropout": float(getattr(provider, "attention_dropout", 0.0)),
                "output_dropout": float(getattr(provider, "output_dropout", 0.0)),
                "num_experts": int(getattr(provider, "num_moe_experts")),
                "num_experts_per_tok": int(getattr(provider, "moe_router_topk")),
                "moe_intermediate_size": int(getattr(provider, "moe_ffn_hidden_size")),
                "moe_shared_expert_intermediate_size": int(
                    getattr(provider, "moe_shared_expert_intermediate_size")
                ),
                "num_shared_experts": int(getattr(provider, "num_shared_experts")),
                "first_k_dense_replace": int(getattr(provider, "first_k_dense_replace")),
                "norm_topk_prob": bool(getattr(provider, "norm_topk_prob", True)),
                "moe_router_enable_expert_bias": bool(
                    getattr(provider, "moe_router_enable_expert_bias", True)
                ),
                "score_function": getattr(provider, "moe_router_score_function", "sigmoid"),
                "router_dtype": getattr(provider, "moe_router_dtype", "fp32"),
                "routed_scaling_factor": float(
                    getattr(provider, "moe_router_topk_scaling_factor", 2.5)
                ),
                "n_group": int(getattr(provider, "moe_router_num_groups", 1)),
                "topk_group": int(getattr(provider, "moe_router_group_topk", 1)),
                "pad_token_id": int(getattr(provider, "pad_token_id", 0)),
                "eos_token_id": int(getattr(provider, "eos_token_id", 3)),
                "partial_rotary_factor": float(getattr(provider, "partial_rotary_factor", 1.0)),
                "rope_scaling": getattr(provider, "rope_scaling", None),
                "max_window_layers": int(getattr(provider, "max_window_layers", 20)),
                "output_router_logits": bool(getattr(provider, "output_router_logits", False)),
                "num_nextn_predict_layers": int(
                    getattr(provider, "num_nextn_predict_layers", 0)
                ),
                "mtp_loss_scaling_factor": float(
                    getattr(provider, "mtp_loss_scaling_factor", 0.0)
                ),
                "torch_dtype": "bfloat16"
                if getattr(provider, "params_dtype", torch.float32) == torch.bfloat16
                else "float16"
                if getattr(provider, "params_dtype", torch.float32) == torch.float16
                else "float32",
            }
        )

        hf_config["max_position_embeddings"] = int(getattr(provider, "seq_length"))
        hf_config["head_dim"] = int(getattr(provider, "kv_channels"))
        return hf_config

    @classmethod
    def _hf_qkv_to_megatron_qkv(cls, weight: torch.Tensor) -> torch.Tensor:
        """
        Convert HF packed QKV:
            [Q_all, K_all, V_all]
        to Megatron grouped-QKV:
            [Q_group0, K_group0, V_group0, Q_group1, K_group1, V_group1, ...]
        """
        num_attention_heads = cls.NUM_ATTENTION_HEADS
        num_query_groups = cls.NUM_QUERY_GROUPS
        head_dim = cls.HEAD_DIM
        heads_per_group = num_attention_heads // num_query_groups

        hidden_size = weight.shape[1]

        q_rows = num_attention_heads * head_dim
        kv_rows = num_query_groups * head_dim

        q, k, v = torch.split(weight, [q_rows, kv_rows, kv_rows], dim=0)

        q = q.view(num_query_groups, heads_per_group, head_dim, hidden_size)
        k = k.view(num_query_groups, 1, head_dim, hidden_size)
        v = v.view(num_query_groups, 1, head_dim, hidden_size)

        grouped = torch.cat([q, k, v], dim=1)
        return grouped.reshape(-1, hidden_size)

    @classmethod
    def _megatron_qkv_to_hf_qkv(cls, weight: torch.Tensor) -> torch.Tensor:
        """
        Inverse of _hf_qkv_to_megatron_qkv.
        """
        num_attention_heads = cls.NUM_ATTENTION_HEADS
        num_query_groups = cls.NUM_QUERY_GROUPS
        head_dim = cls.HEAD_DIM
        heads_per_group = num_attention_heads // num_query_groups

        hidden_size = weight.shape[1]

        grouped = weight.view(
            num_query_groups,
            heads_per_group + 2,
            head_dim,
            hidden_size,
        )

        q = grouped[:, :heads_per_group, :, :].reshape(
            num_attention_heads * head_dim, hidden_size
        )
        k = grouped[:, heads_per_group : heads_per_group + 1, :, :].reshape(
            num_query_groups * head_dim, hidden_size
        )
        v = grouped[:, heads_per_group + 1 :, :, :].reshape(
            num_query_groups * head_dim, hidden_size
        )

        return torch.cat([q, k, v], dim=0)

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = []

        # Global mappings
        mapping_list.extend(
            [
                AutoMapping(
                    megatron_param="embedding.word_embeddings.weight",
                    hf_param="model.word_embeddings.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.final_layernorm.weight",
                    hf_param="model.norm.weight",
                ),
                AutoMapping(
                    megatron_param="output_layer.weight",
                    hf_param="lm_head.weight",
                ),
            ]
        )

        for layer_idx in range(self.NUM_LAYERS):
            layer = str(layer_idx)

            mapping_list.extend(
                [
                    AutoMapping(
                        megatron_param=f"decoder.layers.{layer}.self_attention.linear_qkv.layer_norm_weight",
                        hf_param=f"model.layers.{layer}.input_layernorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"decoder.layers.{layer}.self_attention.linear_qkv.weight",
                        hf_param=f"model.layers.{layer}.attention.query_key_value.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"decoder.layers.{layer}.self_attention.linear_proj.weight",
                        hf_param=f"model.layers.{layer}.attention.dense.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"decoder.layers.{layer}.self_attention.q_layernorm.weight",
                        hf_param=f"model.layers.{layer}.attention.query_layernorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"decoder.layers.{layer}.self_attention.k_layernorm.weight",
                        hf_param=f"model.layers.{layer}.attention.key_layernorm.weight",
                    ),
                ]
            )

            if layer_idx < self.FIRST_K_DENSE_REPLACE:
                mapping_list.extend(
                    [
                        AutoMapping(
                            megatron_param=f"decoder.layers.{layer}.mlp.linear_fc1.layer_norm_weight",
                            hf_param=f"model.layers.{layer}.post_attention_layernorm.weight",
                        ),
                        GatedMLPMapping(
                            megatron_param=f"decoder.layers.{layer}.mlp.linear_fc1.weight",
                            gate=f"model.layers.{layer}.mlp.gate_proj.weight",
                            up=f"model.layers.{layer}.mlp.up_proj.weight",
                        ),
                        AutoMapping(
                            megatron_param=f"decoder.layers.{layer}.mlp.linear_fc2.weight",
                            hf_param=f"model.layers.{layer}.mlp.down_proj.weight",
                        ),
                    ]
                )
                continue

            mapping_list.extend(
                [
                    AutoMapping(
                        megatron_param=f"decoder.layers.{layer}.pre_mlp_layernorm.weight",
                        hf_param=f"model.layers.{layer}.post_attention_layernorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"decoder.layers.{layer}.mlp.router.weight",
                        hf_param=f"model.layers.{layer}.mlp.gate.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"decoder.layers.{layer}.mlp.router.expert_bias",
                        hf_param=f"model.layers.{layer}.mlp.gate.expert_bias",
                    ),
                ]
            )

            for expert_idx in range(self.NUM_EXPERTS):
                expert = str(expert_idx)
                mapping_list.extend(
                    [
                        GatedMLPMapping(
                            megatron_param=f"decoder.layers.{layer}.mlp.experts.linear_fc1.weight{expert}",
                            gate=f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                            up=f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                        ),
                        AutoMapping(
                            megatron_param=f"decoder.layers.{layer}.mlp.experts.linear_fc2.weight{expert}",
                            hf_param=f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                        ),
                    ]
                )

            mapping_list.extend(
                [
                    GatedMLPMapping(
                        megatron_param=f"decoder.layers.{layer}.mlp.shared_experts.linear_fc1.weight",
                        gate=f"model.layers.{layer}.mlp.shared_experts.gate_proj.weight",
                        up=f"model.layers.{layer}.mlp.shared_experts.up_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"decoder.layers.{layer}.mlp.shared_experts.linear_fc2.weight",
                        hf_param=f"model.layers.{layer}.mlp.shared_experts.down_proj.weight",
                    ),
                ]
            )

        return MegatronMappingRegistry(*mapping_list)

    def build_conversion_tasks(self, hf_pretrained, megatron_model):
        tasks = super().build_conversion_tasks(hf_pretrained, megatron_model)
        filtered = [t for t in tasks if t is not None]

        dropped = len(tasks) - len(filtered)
        if dropped:
            print(
                f"[Param2Bridge] WARNING: dropped {dropped} null conversion task(s) "
                f"produced by build_conversion_tasks()."
            )

        return filtered

    def load_weights_hf_to_megatron(
        self,
        hf_pretrained,
        megatron_model,
        allowed_mismatched_params: Optional[List[str]] = None,
    ):
        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        with contextlib.ExitStack() as stack:
            if hasattr(megatron_model[0], "hide_teacher_model"):
                stack.enter_context(megatron_model[0].hide_teacher_model())
            if hasattr(megatron_model[0], "hide_loss_modules"):
                stack.enter_context(megatron_model[0].hide_loss_modules())

            hf_to_megatron_tasks = self.build_conversion_tasks(
                hf_pretrained, megatron_model
            )
            hf_state_dict: Mapping[str, torch.Tensor] = (
                hf_pretrained.state if hasattr(hf_pretrained, "state") else {}
            )

            description = f"Loading from {hf_pretrained.model_name_or_path}"
            _hf_import_cache: Dict[str, torch.Tensor] = {}

            for task in self._with_progress_tracking(hf_to_megatron_tasks, description):
                if task is None:
                    continue
                if task.megatron_module is None:
                    continue

                hf_param_key = str(task.mapping.hf_param)
                is_grouped = getattr(task.mapping, "is_grouped_export", False)

                if is_grouped and hf_param_key in _hf_import_cache:
                    hf_weights = _hf_import_cache[hf_param_key]
                else:
                    hf_weights = self.maybe_modify_loaded_hf_weight(
                        task.mapping.hf_param, hf_state_dict
                    )
                    if is_grouped:
                        _hf_import_cache[hf_param_key] = hf_weights

                if (
                    str(task.mapping.hf_param).endswith(".attention.query_key_value.weight")
                    and str(task.mapping.megatron_param).endswith(".self_attention.linear_qkv.weight")
                ):
                    hf_weights = self._hf_qkv_to_megatron_qkv(hf_weights)

                converted_weights = task.mapping.hf_to_megatron(
                    hf_weights, task.megatron_module
                )

                if converted_weights is not None:
                    assert task.param_weight is not None, (
                        "param_weight is required for HF->Megatron conversion"
                    )

                    if converted_weights.shape != task.param_weight.shape:
                        is_whitelisted = False
                        if allowed_mismatched_params:
                            for pattern in allowed_mismatched_params:
                                if (
                                    fnmatch.fnmatch(task.mapping.megatron_param, pattern)
                                    or fnmatch.fnmatch(task.param_name, pattern)
                                ):
                                    is_whitelisted = True
                                    break

                        if is_whitelisted:
                            print(
                                f"WARNING: Shape mismatch for megatron param "
                                f"{task.mapping.megatron_param} allowed by whitelist. Skipping."
                            )
                            continue

                        raise ValueError(
                            f"Shape mismatch for megatron param {task.mapping.megatron_param}:\n"
                            f" Expected shape: {task.param_weight.shape}\n"
                            f" Got shape: {converted_weights.shape}\n"
                            f" Bridge type: {type(task.mapping).__name__}\n"
                            f" HF mapping: {task.mapping.hf_param}"
                        )

                    task.param_weight.data.copy_(converted_weights)

            self._broadcast_shared_embeddings(megatron_model)
            return megatron_model
