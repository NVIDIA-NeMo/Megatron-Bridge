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

import copy
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Iterable

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.minimax_m3.modeling_minimax_m3_vl import MiniMaxM3VLModel


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


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    """Read a value from either a Hugging Face config object or a dictionary."""
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _config_to_dict(config: Any) -> dict[str, Any]:
    """Return a detached dictionary representation of an HF config."""
    if config is None:
        return {}
    if isinstance(config, dict):
        return copy.deepcopy(config)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        return copy.deepcopy(to_dict())
    return copy.deepcopy(vars(config))


@dataclass
class MiniMaxM3VLModelProvider(MiniMaxM3ModelProvider):
    """Provider for the complete MiniMax-M3 vision-language model."""

    # The vision stack is intentionally replicated across TP ranks. The text
    # backbone retains its existing TP/PP/EP behavior.
    scatter_embedding_sequence_parallel: bool = False
    vision_config: Any = None
    hf_config_dict: dict[str, Any] = field(default_factory=dict)

    image_token_id: int = 200025
    video_token_id: int = 200026
    projector_hidden_size: int = 6144
    multimodal_projector_bias: bool = True
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    @property
    def special_token_ids(self) -> dict[str, int]:
        """Return the token IDs used by multimodal data pipelines."""
        return {"images": self.image_token_id}

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> MiniMaxM3VLModel:
        """Construct the complete MiniMax-M3 VLM."""
        model = MiniMaxM3VLModel(
            self,
            pre_process=True if pre_process is None else pre_process,
            post_process=True if post_process is None else post_process,
            vp_stage=vp_stage,
        )
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )
        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> GPTModel:
        """Construct only the Megatron text component used inside the VLM."""
        return GPTModelProvider.provide(
            self,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )


@MegatronModelBridge.register_bridge(
    source="MiniMaxM3SparseForConditionalGeneration",
    target=MiniMaxM3VLModel,
    provider=MiniMaxM3VLModelProvider,
    model_type="minimax_m3_vl",
)
class MiniMaxM3Bridge(MegatronModelBridge):
    """
    Megatron Bridge for the complete MiniMax-M3 vision-language model.

    MiniMax-M3 ships as a natively multimodal checkpoint
    (``MiniMaxM3SparseForConditionalGeneration``): a CLIP-style vision tower
    plus a sparse-MoE text backbone. This bridge converts the vision tower,
    both multimodal projector MLPs, and the language model into a
    ``MiniMaxM3VLModel``. The vision stack is replicated across TP ranks while
    the language model retains Megatron TP/PP/EP sharding.

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
        - MTP (Multi-Token Prediction) modules are not mapped. The released
          checkpoint advertises ``num_nextn_predict_layers`` in its config but
          ships no ``mtp.*`` weights, so ``mtp_num_layers`` is forced to None.
        - Persisted Hugging Face export requires the original source checkpoint.
          Converted language/vision/projector tensors replace their source
          values, while the unsupported lightning-indexer tensors are preserved
          byte-for-byte. Config-only CPU export is therefore unsupported.
        - The vision forward matches the native Transformers implementation,
          which concatenates multiple image/video patch grids into one
          bidirectional attention sequence. Segmented multi-image/video
          attention remains future work.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("MiniMaxAI/MiniMax-M3", trust_remote_code=True)
        >>> provider = bridge.to_megatron_provider()
    """

    REQUIRES_HF_SOURCE_FOR_EXPORT = True

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

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MiniMaxM3VLModelProvider:
        """Convert the Hugging Face MiniMax-M3 config to a full VLM provider."""
        hf_config = hf_pretrained.config
        text_config = getattr(hf_config, "text_config", hf_config)

        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        provider_kwargs.pop("_mla_rope_params", None)
        valid_fields = MiniMaxM3VLModelProvider.__dataclass_fields__
        provider = MiniMaxM3VLModelProvider(**{k: v for k, v in provider_kwargs.items() if k in valid_fields})

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
            getattr(hf_config, "tie_word_embeddings", getattr(text_config, "tie_word_embeddings", False))
        )

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

        # Multimodal configuration. The legacy public config stores temporal
        # and spatial merge sizes in img_token_compression_config, while the
        # native Transformers config exposes them directly on vision_config.
        provider.vision_config = getattr(hf_config, "vision_config", None)
        provider.hf_config_dict = _config_to_dict(hf_config)
        compression_config = _config_value(hf_config, "img_token_compression_config", None)
        if compression_config is None and provider.vision_config is not None:
            compression_config = _config_value(provider.vision_config, "img_token_compression_config", None)
        compression_config = compression_config or {}
        provider.spatial_merge_size = int(
            _config_value(
                provider.vision_config,
                "spatial_merge_size",
                _config_value(compression_config, "spatial_merge_size", 2),
            )
        )
        provider.temporal_patch_size = int(
            _config_value(
                provider.vision_config,
                "temporal_patch_size",
                _config_value(compression_config, "temporal_patch_size", 2),
            )
        )
        # Normalize the legacy schema for the local vision implementation.
        if provider.vision_config is not None:
            if isinstance(provider.vision_config, dict):
                provider.vision_config.setdefault("spatial_merge_size", provider.spatial_merge_size)
                provider.vision_config.setdefault("temporal_patch_size", provider.temporal_patch_size)
            else:
                if not hasattr(provider.vision_config, "spatial_merge_size"):
                    provider.vision_config.spatial_merge_size = provider.spatial_merge_size
                if not hasattr(provider.vision_config, "temporal_patch_size"):
                    provider.vision_config.temporal_patch_size = provider.temporal_patch_size

        image_token_id = _config_value(hf_config, "image_token_id")
        if image_token_id is None:
            image_token_id = _config_value(hf_config, "image_token_index", 200025)
        provider.image_token_id = int(image_token_id)

        video_token_id = _config_value(hf_config, "video_token_id")
        if video_token_id is None:
            video_token_id = _config_value(hf_config, "video_token_index", 200026)
        provider.video_token_id = int(video_token_id)
        provider.projector_hidden_size = int(_config_value(hf_config, "projector_hidden_size", provider.hidden_size))
        provider.multimodal_projector_bias = bool(_config_value(hf_config, "multimodal_projector_bias", True))

        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider: GPTModelProvider) -> dict[str, Any]:
        """Build the nested MiniMax-M3 VLM config used for reference-backed export."""
        hf_config = (
            copy.deepcopy(provider.hf_config_dict)
            if isinstance(provider, MiniMaxM3VLModelProvider) and provider.hf_config_dict
            else {}
        )
        text_config = copy.deepcopy(hf_config.get("text_config", {}))
        text_config.update(
            {
                "architectures": ["MiniMaxM3SparseForCausalLM"],
                "hidden_size": provider.hidden_size,
                "intermediate_size": provider.moe_ffn_hidden_size,
                "dense_intermediate_size": provider.ffn_hidden_size,
                "shared_intermediate_size": provider.moe_shared_expert_intermediate_size,
                "num_hidden_layers": provider.num_layers,
                "num_attention_heads": provider.num_attention_heads,
                "num_key_value_heads": provider.num_query_groups,
                "head_dim": provider.kv_channels,
                "vocab_size": provider.vocab_size,
                "rms_norm_eps": provider.layernorm_epsilon,
                "rope_theta": provider.rotary_base,
                "rotary_dim": round(provider.rotary_percent * provider.kv_channels),
                "partial_rotary_factor": provider.rotary_percent,
                "hidden_act": "swigluoai",
                "use_qk_norm": provider.qk_layernorm,
                "use_gemma_norm": provider.layernorm_zero_centered_gamma,
                "tie_word_embeddings": provider.share_embeddings_and_output_weights,
                "num_local_experts": provider.num_moe_experts,
                "num_experts_per_tok": provider.moe_router_topk,
                "n_shared_experts": 1 if provider.moe_shared_expert_intermediate_size else 0,
                "scoring_func": provider.moe_router_score_function,
                "use_routing_bias": provider.moe_router_enable_expert_bias,
                "moe_layer_freq": provider.moe_layer_freq,
                "swiglu_alpha": 1.702,
                "swiglu_limit": provider.activation_func_clamp_value,
                "routed_scaling_factor": provider.moe_router_topk_scaling_factor,
            }
        )
        text_config.setdefault("max_position_embeddings", provider.seq_length)

        if isinstance(provider, MiniMaxM3VLModelProvider):
            vision_config = copy.deepcopy(hf_config.get("vision_config", _config_to_dict(provider.vision_config)))
            vision_config.update(
                {
                    "spatial_merge_size": provider.spatial_merge_size,
                    "temporal_patch_size": provider.temporal_patch_size,
                }
            )
            compression_config = copy.deepcopy(hf_config.get("img_token_compression_config", {}))
            compression_config.update(
                {
                    "image_token_compression_method": "patch_merge",
                    "spatial_merge_size": provider.spatial_merge_size,
                    "temporal_patch_size": provider.temporal_patch_size,
                }
            )
            hf_config.update(
                {
                    "vision_config": vision_config,
                    "img_token_compression_config": compression_config,
                    "image_token_index": provider.image_token_id,
                    "video_token_index": provider.video_token_id,
                    "projector_hidden_size": provider.projector_hidden_size,
                    "multimodal_projector_bias": provider.multimodal_projector_bias,
                }
            )

        dtype_name = str(provider.params_dtype).removeprefix("torch.")
        hf_config.update(
            {
                "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
                "model_type": "minimax_m3_vl",
                "text_config": text_config,
                "torch_dtype": dtype_name,
            }
        )
        return hf_config

    def stream_hf_export_passthrough(
        self,
        hf_pretrained: PreTrainedCausalLM,
        *,
        cpu: bool = True,
    ) -> Iterable[HFWeightTuple]:
        """Preserve unsupported Lightning Indexer weights from the source checkpoint."""
        if not hasattr(hf_pretrained, "state"):
            raise NotImplementedError(
                "MiniMax-M3 VLM export requires the original Hugging Face checkpoint "
                "to preserve Lightning Indexer tensors."
            )

        for name in hf_pretrained.state:
            if not (name.startswith("language_model.") and ".self_attn.index_" in name):
                continue
            tensor = hf_pretrained.state[name].detach()
            yield HFWeightTuple(name, tensor.cpu() if cpu else tensor)

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return parameter mappings for the MiniMax-M3 VLM.

        Text mappings target the nested ``language_model``. Vision and
        projector modules intentionally preserve the legacy checkpoint names,
        so their mappings are replicated identity mappings.
        """
        language_prefix = "language_model."
        param_mappings = {
            # Global weights
            f"{language_prefix}embedding.word_embeddings.weight": "language_model.model.embed_tokens.weight",
            f"{language_prefix}output_layer.weight": "language_model.lm_head.weight",
            f"{language_prefix}decoder.final_layernorm.weight": "language_model.model.norm.weight",
            # Input layernorm (fused into linear_qkv for the TE backend)
            f"{language_prefix}decoder.layers.*.input_layernorm.weight": "language_model.model.layers.*.input_layernorm.weight",
            f"{language_prefix}decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "language_model.model.layers.*.input_layernorm.weight",
            # Post-attention layernorm: pre_mlp_layernorm on MoE layers,
            # fused into linear_fc1 on dense layers
            f"{language_prefix}decoder.layers.*.pre_mlp_layernorm.weight": "language_model.model.layers.*.post_attention_layernorm.weight",
            f"{language_prefix}decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "language_model.model.layers.*.post_attention_layernorm.weight",
            # Attention
            f"{language_prefix}decoder.layers.*.self_attention.linear_proj.weight": "language_model.model.layers.*.self_attn.o_proj.weight",
            # Per-head QK RMSNorm (weight shape = head_dim)
            f"{language_prefix}decoder.layers.*.self_attention.q_layernorm.weight": "language_model.model.layers.*.self_attn.q_norm.weight",
            f"{language_prefix}decoder.layers.*.self_attention.k_layernorm.weight": "language_model.model.layers.*.self_attn.k_norm.weight",
            # Dense-layer MLP down projection
            f"{language_prefix}decoder.layers.*.mlp.linear_fc2.weight": "language_model.model.layers.*.mlp.down_proj.weight",
            # MoE router and expert bias — on-disk uses the block_sparse_moe prefix
            f"{language_prefix}decoder.layers.*.mlp.router.weight": "language_model.model.layers.*.block_sparse_moe.gate.weight",
            f"{language_prefix}decoder.layers.*.mlp.router.expert_bias": "language_model.model.layers.*.block_sparse_moe.e_score_correction_bias",
            # Shared expert down projection
            f"{language_prefix}decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "language_model.model.layers.*.block_sparse_moe.shared_experts.down_proj.weight",
        }

        mapping_list = [
            AutoMapping(megatron_param=megatron_param, hf_param=hf_param)
            for megatron_param, hf_param in param_mappings.items()
        ]

        mapping_list.extend(
            [
                # QKV
                QKVMapping(
                    megatron_param=f"{language_prefix}decoder.layers.*.self_attention.linear_qkv.weight",
                    q="language_model.model.layers.*.self_attn.q_proj.weight",
                    k="language_model.model.layers.*.self_attn.k_proj.weight",
                    v="language_model.model.layers.*.self_attn.v_proj.weight",
                ),
                # Dense-layer gated MLP
                GatedMLPMapping(
                    megatron_param=f"{language_prefix}decoder.layers.*.mlp.linear_fc1.weight",
                    gate="language_model.model.layers.*.mlp.gate_proj.weight",
                    up="language_model.model.layers.*.mlp.up_proj.weight",
                ),
                # Shared expert gated MLP
                GatedMLPMapping(
                    megatron_param=f"{language_prefix}decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="language_model.model.layers.*.block_sparse_moe.shared_experts.gate_proj.weight",
                    up="language_model.model.layers.*.block_sparse_moe.shared_experts.up_proj.weight",
                ),
                # Routed experts — on-disk layout: per-expert w1 (gate), w3 (up), w2 (down)
                GatedMLPMapping(
                    megatron_param=f"{language_prefix}decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="language_model.model.layers.*.block_sparse_moe.experts.*.w1.weight",
                    up="language_model.model.layers.*.block_sparse_moe.experts.*.w3.weight",
                ),
                AutoMapping(
                    megatron_param=f"{language_prefix}decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="language_model.model.layers.*.block_sparse_moe.experts.*.w2.weight",
                ),
                ReplicatedMapping(
                    megatron_param="vision_tower.**",
                    hf_param="vision_tower.**",
                ),
                ReplicatedMapping(
                    megatron_param="multi_modal_projector.**",
                    hf_param="multi_modal_projector.**",
                ),
                ReplicatedMapping(
                    megatron_param="patch_merge_mlp.**",
                    hf_param="patch_merge_mlp.**",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
