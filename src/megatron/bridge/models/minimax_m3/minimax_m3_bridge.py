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

from dataclasses import dataclass, replace
from functools import partial

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig

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
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

try:
    from megatron.core.fusions.fused_bias_geglu import quick_gelu
except ImportError:
    # Fallback if fused_bias_geglu is not available
    quick_gelu = torch.nn.functional.gelu


class MiniMaxM3TopKRouter(TopKRouter):
    """MiniMax-M3 router that computes its projection in the weight dtype."""

    def gating(self, input: torch.Tensor) -> torch.Tensor:
        """Match HF by widening router inputs to the FP32 router weight dtype."""
        return super().gating(input.to(dtype=self.weight.dtype))


def minimax_m3_block_spec(
    config: TransformerConfig,
    use_transformer_engine: bool = True,
    normalization: str | None = None,
    qk_l2_norm: bool | None = False,
    vp_stage: int | None = None,
    pp_rank: int | None = None,
    **kwargs: object,
) -> TransformerBlockSubmodules:
    """Build a GPT block spec that uses MiniMax-M3's FP32 router projection."""
    block_spec = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=use_transformer_engine,
        normalization=normalization,
        qk_l2_norm=qk_l2_norm,
        vp_stage=vp_stage,
        pp_rank=pp_rank,
        **kwargs,
    )

    for layer_spec in block_spec.layer_specs:
        mlp_spec = layer_spec.submodules.mlp
        if isinstance(mlp_spec, partial) and isinstance(mlp_spec.func, type) and issubclass(mlp_spec.func, MoELayer):
            mlp_kwargs = dict(mlp_spec.keywords or {})
            mlp_submodules = mlp_kwargs["submodules"]
            if mlp_submodules.router is not TopKRouter:
                continue
            mlp_kwargs["submodules"] = replace(mlp_submodules, router=MiniMaxM3TopKRouter)
            layer_spec.submodules.mlp = partial(mlp_spec.func, *mlp_spec.args, **mlp_kwargs)

    return block_spec


AutoMapping.register_module_type("MiniMaxM3TopKRouter", "replicated")


def _promote_router_weights_to_float32(model: list[torch.nn.Module]) -> list[torch.nn.Module]:
    """Keep MiniMax-M3 router parameters in FP32 for every load path.

    Megatron initializes router parameters in ``params_dtype`` even when
    ``moe_router_dtype="fp32"``. Promoting them immediately after construction
    prevents truncation when loading either HF weights or a native Megatron
    checkpoint.
    """
    for model_chunk in model:
        for module in model_chunk.modules():
            if isinstance(module, TopKRouter) and module.weight.dtype != torch.float32:
                module.weight.data = module.weight.data.float()
            if isinstance(module, TopKRouter):
                module._keep_in_float32_parameter_names = ("weight",)
    return model


@dataclass
class MiniMaxM3ModelProvider(GPTModelProvider):
    """GPT provider that preserves MiniMax-M3's FP32 router parameters."""

    hf_max_position_embeddings: int | None = None
    hf_bos_token_id: int | None = None
    hf_eos_token_id: int | list[int] | None = None
    hf_pad_token_id: int | None = None

    def __post_init__(self) -> None:
        """Install MiniMax-M3 router behavior on fresh and deserialized providers."""
        super().__post_init__()
        layer_spec = self.transformer_layer_spec
        if layer_spec is get_gpt_decoder_block_spec:
            self.transformer_layer_spec = minimax_m3_block_spec
        elif isinstance(layer_spec, partial) and layer_spec.func is get_gpt_decoder_block_spec:
            self.transformer_layer_spec = partial(
                minimax_m3_block_spec,
                *layer_spec.args,
                **(layer_spec.keywords or {}),
            )

        if not hasattr(self, "_pre_wrap_hooks") or _promote_router_weights_to_float32 not in self._pre_wrap_hooks:
            self.register_pre_wrap_hook(_promote_router_weights_to_float32, prepend=True)


class _FusedGateUpMapping(MegatronParamMapping[torch.Tensor]):
    """Map one fused HF gate/up tensor to Megatron's TP-aware fused MLP weight."""

    def __init__(self, megatron_param: str, hf_param: str):
        """Initialize the fused gate/up mapping."""
        super().__init__(megatron_param, hf_param)
        self._gated_mapping = GatedMLPMapping(
            megatron_param=megatron_param,
            gate=f"{hf_param}.gate",
            up=f"{hf_param}.up",
        )

    def hf_to_megatron(
        self,
        hf_weights: torch.Tensor,
        megatron_module: torch.nn.Module,
    ) -> torch.Tensor:
        """Split the native HF tensor and delegate TP distribution."""
        if hf_weights.shape[0] % 2 != 0:
            raise ValueError(
                f"Expected an even gate/up dimension for {self.hf_param}, got shape {tuple(hf_weights.shape)}."
            )
        gate, up = torch.chunk(hf_weights, 2, dim=0)
        return self._gated_mapping.hf_to_megatron({"gate": gate, "up": up}, megatron_module)

    def megatron_to_hf(
        self,
        megatron_weights: torch.Tensor | None,
        megatron_module: torch.nn.Module | None,
    ) -> dict[str, torch.Tensor]:
        """Gather Megatron TP shards and fuse the native HF gate/up tensor."""
        converted = self._gated_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not converted:
            return {}
        gate = converted[f"{self.hf_param}.gate"]
        up = converted[f"{self.hf_param}.up"]
        return {str(self.hf_param): torch.cat((gate, up), dim=0)}

    def resolve(self, captures: tuple[str, ...]) -> MegatronParamMapping:
        """Return a mapping with all layer wildcards resolved."""
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        if not isinstance(resolved_hf_param, str):
            raise TypeError("Fused gate/up HF parameter must resolve to one tensor name.")
        return type(self)(resolved_megatron_param, resolved_hf_param)


@MegatronModelBridge.register_bridge(
    source="MiniMaxM3SparseForConditionalGeneration",
    target=GPTModel,
    model_type="minimax_m3_vl",
)
class MiniMaxM3Bridge(MegatronModelBridge):
    """
    Megatron Bridge for the MiniMax-M3 language model.

    MiniMax-M3 ships as a natively multimodal checkpoint
    (``MiniMaxM3SparseForConditionalGeneration``): a CLIP-style vision tower
    plus a sparse-MoE text backbone. This bridge converts the *language model*
    (``language_model.*`` weights) to a Megatron-Core ``GPTModel``; the vision
    tower and multimodal projector are not yet bridged.

    Text backbone architecture:
        - Mixed dense/MoE decoder: the first layers are dense
          (``dense_intermediate_size`` MLP), the rest use 128 routed experts
          (top-4) plus one shared expert.
        - Sigmoid router scoring with expert-bias correction and
          ``routed_scaling_factor`` applied to the normalized top-k weights
          (same routing math as DeepSeek-V3).
        - SwiGLU-OAI expert/MLP activation: clamped gate/up projections with a
          ``+1`` linear offset (same as GPT-OSS), expressed via
          ``activation_func_clamp_value`` and ``glu_linear_offset``.
        - Gemma-style RMSNorm (``x * (1 + w)``) on every norm, expressed via
          ``layernorm_zero_centered_gamma``.
        - GQA attention with per-head QK RMSNorm and partial RoPE
          (``rotary_dim`` of ``head_dim`` channels rotated).

    Known limitations:
        - The lightning-indexer block-sparse attention branch
          (``self_attn.index_{q,k}_{proj,norm}``) is not mapped; the Megatron
          model runs full causal attention on every layer. Selection happens at
          ``index_block_size`` granularity with ``index_topk_blocks`` kept per
          query, so full attention is mathematically identical for sequences up
          to ``index_topk_blocks * index_block_size`` tokens (2048 for the
          released checkpoint) and an approximation beyond that.
        - The vision tower, multimodal projector, and patch-merge MLP are not
          mapped (language model only).
        - MTP (Multi-Token Prediction) modules are not mapped. The released
          checkpoint advertises ``num_nextn_predict_layers`` in its config but
          ships no ``mtp.*`` weights, so ``mtp_num_layers`` is forced to None.
        - This bridge imports the text backbone from the public multimodal
          checkpoint. Standalone Hugging Face export is handled by
          :class:`MiniMaxM3TextBridge` using the native Transformers text layout.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("MiniMaxAI/MiniMax-M3", trust_remote_code=True)
        >>> provider = bridge.to_megatron_provider()
    """

    SUPPORTS_HF_PRETRAINED_EXPORT = False

    # AutoTokenizer.save_pretrained() consolidates these artifacts. Preserve the
    # pinned source tokenizer files in standalone text exports.
    ADDITIONAL_FILE_PATTERNS = [
        "added_tokens.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
    ]

    @classmethod
    def hf_to_megatron_activation(cls, hidden_act: str):
        """Convert HF activation name to Megatron activation function.

        The released MiniMax-M3 checkpoint declares ``hidden_act="swigluoai"``,
        which is not a standard ACT2FN key (transformers normalizes it to
        ``silu`` and computes the gate inline from ``swiglu_alpha`` /
        ``swiglu_limit``). Map it to ``quick_gelu`` — the SwiGLU-OAI gate is
        ``gate * sigmoid(1.702 * gate)``, i.e. exactly quick-GELU; the clamp
        and ``+1`` offset are carried by separate provider fields.
        """
        if hidden_act == "swigluoai":
            return quick_gelu
        return super().hf_to_megatron_activation(hidden_act)

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MiniMaxM3ModelProvider:
        """Convert the HuggingFace MiniMax-M3 config to a GPTModelProvider."""
        hf_config = hf_pretrained.config
        text_config = (
            hf_config
            if getattr(hf_config, "model_type", None) == "minimax_m3_vl_text"
            else getattr(hf_config, "text_config", hf_config)
        )

        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        provider_kwargs.pop("_mla_rope_params", None)
        valid_fields = MiniMaxM3ModelProvider.__dataclass_fields__
        provider = MiniMaxM3ModelProvider(**{k: v for k, v in provider_kwargs.items() if k in valid_fields})

        # Use decoder block spec to properly handle moe_layer_freq (mixed dense/MoE layers)
        provider.transformer_layer_spec = partial(minimax_m3_block_spec, use_transformer_engine=HAVE_TE)

        # Gemma-style RMSNorm: weights stored zero-centered, applied as x * (1 + w)
        provider.normalization = "RMSNorm"
        provider.layernorm_zero_centered_gamma = bool(getattr(text_config, "use_gemma_norm", True))
        provider.qk_layernorm = bool(getattr(text_config, "use_qk_norm", True))

        provider.position_embedding_type = "rope"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.hidden_dropout = 0.0

        # tie_word_embeddings lives on text_config for MiniMax-M3 and is False
        # for the released checkpoint (distinct lm_head and embed_tokens tensors)
        provider.share_embeddings_and_output_weights = bool(
            getattr(text_config, "tie_word_embeddings", getattr(hf_config, "tie_word_embeddings", False))
        )
        provider.hf_max_position_embeddings = getattr(text_config, "max_position_embeddings", None)
        provider.hf_bos_token_id = getattr(text_config, "bos_token_id", None)
        provider.hf_eos_token_id = getattr(text_config, "eos_token_id", None)
        provider.hf_pad_token_id = getattr(text_config, "pad_token_id", None)

        # SwiGLU-OAI activation (same as GPT-OSS, but non-interleaved weights):
        # gate = clamp(gate, max=limit); up = clamp(up, +-limit)
        # out = (up + 1) * gate * sigmoid(alpha * gate), alpha = 1.702 (quick-GELU)
        provider.activation_func = quick_gelu
        provider.activation_func_clamp_value = float(getattr(text_config, "swiglu_limit", 7.0))
        provider.glu_linear_offset = 1.0

        # Partial RoPE: only rotary_dim of head_dim channels are rotated
        rotary_dim = getattr(text_config, "rotary_dim", None)
        head_dim = getattr(text_config, "head_dim", None)
        if rotary_dim is not None and head_dim:
            provider.rotary_percent = rotary_dim / head_dim

        # Dense layers use dense_intermediate_size; text_config.intermediate_size
        # is the per-expert FFN size (CONFIG_MAPPING would put it in ffn_hidden_size)
        provider.moe_ffn_hidden_size = text_config.intermediate_size
        dense_ffn_hidden_size = getattr(text_config, "dense_intermediate_size", None)
        if dense_ffn_hidden_size is not None:
            provider.ffn_hidden_size = dense_ffn_hidden_size

        # MoE settings — sigmoid routing with expert bias correction and
        # normalized top-k weights scaled by routed_scaling_factor (DeepSeek-V3 style)
        provider.moe_grouped_gemm = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True
        provider.moe_router_pre_softmax = False
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_dtype = "fp32"
        # HF exposes a token-global Switch auxiliary loss. MCore's closest
        # equivalent uses normalized sigmoid scores rather than HF's softmax
        # scores, but preserves the global (rather than per-sequence) scope.
        provider.moe_router_load_balancing_type = "aux_loss"
        provider.moe_aux_loss_coeff = getattr(text_config, "router_aux_loss_coef", 1e-3)
        provider.moe_router_topk_scaling_factor = getattr(text_config, "routed_scaling_factor", 1.0)
        # The overlapped shared-expert path applies a generic GLU and does not
        # honor activation_func_clamp_value or glu_linear_offset. Keep it off so
        # the shared expert uses MiniMax-M3's clamped (up + 1) SwiGLU-OAI math.
        provider.moe_shared_expert_overlap = False

        n_shared_experts = getattr(text_config, "n_shared_experts", None)
        shared_intermediate_size = getattr(text_config, "shared_intermediate_size", 0) or 0
        if n_shared_experts is None:
            n_shared_experts = 1 if shared_intermediate_size else 0
        provider.moe_shared_expert_intermediate_size = (n_shared_experts * shared_intermediate_size) or None

        # Per-layer dense/MoE pattern. The checkpoint config carries a 0/1
        # moe_layer_freq list; the native transformers config converts it into
        # mlp_layer_types ("dense"/"sparse") strings.
        moe_layer_freq = getattr(text_config, "moe_layer_freq", None)
        if moe_layer_freq is None:
            mlp_layer_types = getattr(text_config, "mlp_layer_types", None)
            if mlp_layer_types is not None:
                moe_layer_freq = [1 if layer_type == "sparse" else 0 for layer_type in mlp_layer_types]
        if moe_layer_freq is not None:
            provider.moe_layer_freq = [int(f) for f in moe_layer_freq]

        # The released checkpoint advertises num_nextn_predict_layers in its
        # config but ships no mtp.* weights — keep MTP disabled so conversion
        # does not look for weights that do not exist.
        provider.mtp_num_layers = None

        provider.persist_layer_norm = True
        # The fused bias-activation path only supports quick-GELU when MoE
        # routing probabilities are supplied. MiniMax-M3 also uses the same
        # activation in dense layers and the shared expert, so use the
        # unfused path that applies the clamp and linear offset in both cases.
        provider.bias_activation_fusion = False
        provider.bias_dropout_fusion = True

        # Released checkpoints are bf16; text_config carries no dtype of its own
        provider.fp16 = False
        provider.bf16 = True
        provider.params_dtype = torch.bfloat16
        provider.autocast_dtype = torch.bfloat16

        # max_position_embeddings is 1M; keep the provider default conservative
        # (recipes override seq_length explicitly)
        provider.seq_length = 4096

        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider: GPTModelProvider) -> dict:
        """Convert the Megatron provider to the native standalone text config."""
        text_config = super().megatron_to_hf_config(provider)
        text_config["architectures"] = ["MiniMaxM3VLForCausalLM"]
        text_config["model_type"] = "minimax_m3_vl_text"
        text_config.pop("num_nextn_predict_layers", None)
        text_config.pop("mtp_num_hidden_layers", None)
        text_config.update(
            {
                "intermediate_size": provider.moe_ffn_hidden_size,
                "dense_intermediate_size": provider.ffn_hidden_size,
                "shared_intermediate_size": provider.moe_shared_expert_intermediate_size,
                "head_dim": provider.kv_channels,
                "hidden_act": "silu",
                "max_position_embeddings": (
                    provider.hf_max_position_embeddings
                    if isinstance(provider, MiniMaxM3ModelProvider) and provider.hf_max_position_embeddings is not None
                    else provider.seq_length
                ),
                "bos_token_id": (provider.hf_bos_token_id if isinstance(provider, MiniMaxM3ModelProvider) else None),
                "eos_token_id": (provider.hf_eos_token_id if isinstance(provider, MiniMaxM3ModelProvider) else None),
                "pad_token_id": (provider.hf_pad_token_id if isinstance(provider, MiniMaxM3ModelProvider) else None),
                "rotary_dim": round(provider.rotary_percent * provider.kv_channels),
                "partial_rotary_factor": provider.rotary_percent,
                "rope_parameters": {
                    "rope_theta": provider.rotary_base,
                    "partial_rotary_factor": provider.rotary_percent,
                    "rope_type": "default",
                },
                "use_gemma_norm": provider.layernorm_zero_centered_gamma,
                "use_qk_norm": provider.qk_layernorm,
                "qk_norm_type": "per_head",
                "num_local_experts": provider.num_moe_experts,
                "num_experts_per_tok": provider.moe_router_topk,
                "scoring_func": provider.moe_router_score_function,
                "use_routing_bias": provider.moe_router_enable_expert_bias,
                "routed_scaling_factor": provider.moe_router_topk_scaling_factor,
                "router_aux_loss_coef": provider.moe_aux_loss_coeff,
                "mlp_layer_types": ["sparse" if int(is_moe) else "dense" for is_moe in provider.moe_layer_freq],
                # The Bridge text model intentionally omits the lightning
                # indexer, so the native HF model must instantiate full
                # attention on every layer as well.
                "layer_types": ["full_attention"] * provider.num_layers,
                "swiglu_alpha": 1.702,
                "swiglu_limit": provider.activation_func_clamp_value,
            }
        )
        return text_config

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return the parameter mappings for the MiniMax-M3 language model.

        All HF weights live under the ``language_model.`` prefix of the
        multimodal checkpoint. MoE weights use the legacy ``block_sparse_moe``
        layout with per-expert ``w1`` (gate), ``w3`` (up), and ``w2`` (down)
        tensors, the same on-disk format as MiniMax-M2.
        """
        param_mappings = {
            # Global weights
            "embedding.word_embeddings.weight": "language_model.model.embed_tokens.weight",
            "output_layer.weight": "language_model.lm_head.weight",
            "decoder.final_layernorm.weight": "language_model.model.norm.weight",
            # Input layernorm (fused into linear_qkv for the TE backend)
            "decoder.layers.*.input_layernorm.weight": "language_model.model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "language_model.model.layers.*.input_layernorm.weight",
            # Post-attention layernorm: pre_mlp_layernorm on MoE layers,
            # fused into linear_fc1 on dense layers
            "decoder.layers.*.pre_mlp_layernorm.weight": "language_model.model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "language_model.model.layers.*.post_attention_layernorm.weight",
            # Attention
            "decoder.layers.*.self_attention.linear_proj.weight": "language_model.model.layers.*.self_attn.o_proj.weight",
            # Per-head QK RMSNorm (weight shape = head_dim)
            "decoder.layers.*.self_attention.q_layernorm.weight": "language_model.model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "language_model.model.layers.*.self_attn.k_norm.weight",
            # Dense-layer MLP down projection
            "decoder.layers.*.mlp.linear_fc2.weight": "language_model.model.layers.*.mlp.down_proj.weight",
            # MoE router and expert bias — on-disk uses the block_sparse_moe prefix
            "decoder.layers.*.mlp.router.weight": "language_model.model.layers.*.block_sparse_moe.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "language_model.model.layers.*.block_sparse_moe.e_score_correction_bias",
            # Shared expert down projection
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "language_model.model.layers.*.block_sparse_moe.shared_experts.down_proj.weight",
        }

        mapping_list = [
            AutoMapping(megatron_param=megatron_param, hf_param=hf_param)
            for megatron_param, hf_param in param_mappings.items()
        ]

        mapping_list.extend(
            [
                # QKV
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="language_model.model.layers.*.self_attn.q_proj.weight",
                    k="language_model.model.layers.*.self_attn.k_proj.weight",
                    v="language_model.model.layers.*.self_attn.v_proj.weight",
                ),
                # Dense-layer gated MLP
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="language_model.model.layers.*.mlp.gate_proj.weight",
                    up="language_model.model.layers.*.mlp.up_proj.weight",
                ),
                # Shared expert gated MLP
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="language_model.model.layers.*.block_sparse_moe.shared_experts.gate_proj.weight",
                    up="language_model.model.layers.*.block_sparse_moe.shared_experts.up_proj.weight",
                ),
                # Routed experts — on-disk layout: per-expert w1 (gate), w3 (up), w2 (down)
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="language_model.model.layers.*.block_sparse_moe.experts.*.w1.weight",
                    up="language_model.model.layers.*.block_sparse_moe.experts.*.w3.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="language_model.model.layers.*.block_sparse_moe.experts.*.w2.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)


@MegatronModelBridge.register_bridge(
    source="MiniMaxM3VLForCausalLM",
    target=GPTModel,
    model_type="minimax_m3_vl_text",
)
class MiniMaxM3TextBridge(MiniMaxM3Bridge):
    """Bidirectional bridge for a standalone native Transformers MiniMax-M3 text model.

    The public checkpoint is multimodal and uses a legacy on-disk layout. After
    its language backbone is imported through :class:`MiniMaxM3Bridge`, this
    bridge exports a stock ``MiniMaxM3VLForCausalLM`` checkpoint containing only
    text weights. The exported model uses full causal attention on every layer,
    matching the Megatron representation that intentionally omits the lightning
    indexer.
    """

    SUPPORTS_HF_PRETRAINED_EXPORT = True
    HF_EXPORT_OPTIONAL_ARTIFACTS = ("generation_config",)
    HF_EXPORT_TRUST_REMOTE_CODE = False

    def get_expected_hf_export_keys(self, hf_config: object) -> set[str]:
        """Return the exact stock Transformers state keys for strict export."""
        try:
            from transformers import MiniMaxM3VLForCausalLM
        except ImportError as error:
            raise RuntimeError("Standalone MiniMax-M3 text export requires Transformers 5.12.1 or newer.") from error

        with torch.device("meta"):
            hf_model = MiniMaxM3VLForCausalLM(hf_config)
        return set(hf_model.state_dict())

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return mappings for the native standalone MiniMax-M3 text layout."""
        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": ("model.layers.*.input_layernorm.weight"),
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": ("model.layers.*.post_attention_layernorm.weight"),
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": (
                "model.layers.*.mlp.shared_experts.down_proj.weight"
            ),
        }
        mapping_list: list[MegatronParamMapping] = [
            AutoMapping(megatron_param=megatron_param, hf_param=hf_param)
            for megatron_param, hf_param in param_mappings.items()
        ]
        mapping_list.extend(
            [
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                _FusedGateUpMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    hf_param="model.layers.*.mlp.gate_up_proj.weight",
                ),
                _FusedGateUpMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    hf_param="model.layers.*.mlp.shared_experts.gate_up_proj.weight",
                ),
                FusedGatedExpertMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    hf_param="model.layers.*.mlp.experts.gate_up_proj",
                ),
                FusedExpertMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.layers.*.mlp.experts.down_proj",
                ),
                FusedGatedExpertMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                    hf_param="model.layers.*.mlp.experts.gate_up_proj",
                ),
                FusedExpertMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                    hf_param="model.layers.*.mlp.experts.down_proj",
                ),
            ]
        )
        return MegatronMappingRegistry(*mapping_list)
