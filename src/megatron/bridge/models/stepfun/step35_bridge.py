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

import logging
from functools import partial
from typing import Dict

import torch
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig as MCoreTransformerConfig
from transformers import AutoConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVGMapping,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.stepfun.configuration_step35 import Step35Config
from megatron.bridge.models.stepfun.step35_provider import (
    Step35DecoderLayer,
    Step35ModelProvider,
    Step35SharedExpertMLP,
)


logger = logging.getLogger(__name__)

# Register the Step3.5 config with transformers AutoConfig.
# This allows AutoConfig.from_pretrained to resolve "step3p5" without requiring
# hub access (works in offline CI environments).
#
# The literal strings "step3p5" and "Step3p5ForCausalLM" are *external HF
# identifiers*: they come from the `model_type` and `architectures` fields in
# the config.json shipped on `stepfun-ai/Step-3.5-Flash`. They are intentionally
# NOT renamed to "step35" / "Step35ForCausalLM" — otherwise
# `AutoConfig.from_pretrained("stepfun-ai/Step-3.5-Flash")` would route to a
# different config class and the bridge resolution below would fail.
AutoConfig.register("step3p5", Step35Config, exist_ok=True)


def _mcore_supports_head_wise_attn_gate() -> bool:
    return "head_wise_attn_gate" in getattr(MCoreTransformerConfig, "__dataclass_fields__", {})


class StackedExpertAutoMapping(AutoMapping):
    """Maps Megatron per-expert weight{i} ↔ HF stacked expert tensor[i].

    Step3.5 HF stores all experts in a single stacked tensor, e.g.
    ``model.layers.*.moe.down_proj.weight`` with shape ``[num_experts, H, I]``.
    Megatron creates individual per-expert tensors named ``weight0``, ``weight1``, …

    The ``megatron_param`` uses a trailing ``weight*`` wildcard to match these names;
    ``hf_param`` has one fewer wildcard (no expert index in the path).  During
    wildcard resolution ``_resolve_names`` resets ``capture_index`` to 0 for the HF
    side, so ``hf_param`` only consumes the layer-index capture and the expert-index
    capture is available to slice the stacked tensor in ``hf_to_megatron``.
    """

    is_grouped_export = True  # All per-expert tasks share the same HF stacked tensor.

    def _expert_idx(self) -> int:
        return int(self.megatron_param.rsplit("weight", 1)[-1])

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module) -> torch.Tensor:
        # hf_weights: [num_experts, H, I] — slice to this expert before delegating.
        return super().hf_to_megatron(hf_weights[self._expert_idx()], megatron_module)


class StackedExpertGatedMLPMapping(GatedMLPMapping):
    """GatedMLPMapping for per-expert Megatron weights backed by HF stacked tensors.

    HF stores all experts' gate/up projections as stacked tensors with shape
    [num_experts, I, H].  Megatron creates individual per-expert
    ``linear_fc1.weight{i}`` tensors (shape [2*I, H], gate+up fused).

    ``megatron_param`` uses a trailing ``weight*`` wildcard.  ``gate`` / ``up``
    each have one fewer wildcard (no expert index in the HF path).  During
    wildcard resolution ``_resolve_names`` resets ``capture_index`` for every
    dict key, so both gate/up only consume the layer-index capture.
    """

    is_grouped_export = True  # All per-expert tasks share the same HF stacked tensors.

    def _expert_idx(self) -> int:
        return int(self.megatron_param.rsplit("weight", 1)[-1])

    def hf_to_megatron(self, hf_weights: Dict[str, torch.Tensor], megatron_module) -> torch.Tensor:
        # hf_weights["gate"/"up"]: [num_experts, I, H] — slice to this expert.
        expert_idx = self._expert_idx()
        sliced = {
            "gate": hf_weights["gate"][expert_idx],
            "up": hf_weights["up"][expert_idx],
        }
        return super().hf_to_megatron(sliced, megatron_module)


class _MTPDenseLayerSpecsList(list):
    """List of per-decoder-layer specs that returns a dense spec on negative-index access.

    ``get_gpt_mtp_block_spec_for_backend`` reads ``spec.layer_specs[-1]`` to decide
    which layer type the MTP transformer sub-layers should use.  For Step3.5 the
    last decoder layer (layer 44) is MoE, but MTP layers 45-47 are NOT in
    ``moe_layers_enum`` and must be dense.

    Overriding ``__getitem__`` for negative indices intercepts only that single
    look-up while leaving normal forward iteration (used by ``TransformerBlock``
    to instantiate the 45 main decoder layers) completely unaffected — CPython's
    list iterator operates on the internal C array directly, bypassing
    ``__getitem__``.
    """

    def __init__(self, data, dense_mtp_spec):
        super().__init__(data)
        self._dense_mtp_spec = dense_mtp_spec

    def __getitem__(self, idx):
        if isinstance(idx, int) and idx < 0:
            return self._dense_mtp_spec
        return super().__getitem__(idx)


def build_step35_layer_spec(cfg, **kw):
    """Per-layer spec for Step3.5: dense for layers 0-2 and 45-47, MoE for 3-44.

    Also rewrites every main-decoder layer's ModuleSpec to use
    ``Step35DecoderLayer`` instead of the default ``TransformerLayer``. The
    custom layer reads ``cfg.layer_types`` at init time to determine whether
    the layer is a sliding-attention layer.

    Returns a TransformerBlockSubmodules whose layer_specs list is wrapped in
    _MTPDenseLayerSpecsList so that get_gpt_mtp_block_spec_for_backend receives
    a dense ModuleSpec (via layer_specs[-1]) for the MTP transformer sub-layers.
    """
    block_submodules = get_gpt_decoder_block_spec(cfg, use_transformer_engine=True, normalization="RMSNorm", **kw)
    # Swap the layer module class on every main-decoder spec. The dense MTP
    # spec below is used for MTP layers (which have their own 1-indexed
    # layer_number namespace) so the routed-expert FFN stays disabled even
    # when the last main decoder layer is MoE.
    for spec in block_submodules.layer_specs:
        spec.module = Step35DecoderLayer
        # Re-bind the shared-expert builder on MoE layers so the shared expert
        # honors ``activation_func_clamp_value_shared_expert``. Dense layers
        # have a plain MLP submodule (no ``shared_experts`` attribute) and are
        # skipped by the ``getattr`` guard.
        mlp_submodules = getattr(spec.submodules.mlp, "submodules", None)
        shared = getattr(mlp_submodules, "shared_experts", None)
        if shared is not None:
            mlp_submodules.shared_experts = partial(Step35SharedExpertMLP, **shared.keywords)
    dense_mtp_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=cfg.qk_layernorm,
    )
    dense_mtp_spec.module = Step35DecoderLayer
    block_submodules.layer_specs = _MTPDenseLayerSpecsList(block_submodules.layer_specs, dense_mtp_spec)

    return block_submodules


# ``source`` and ``model_type`` keep the legacy ``Step3p5ForCausalLM`` /
# ``"step3p5"`` spelling because those are the HF identifiers carried by
# ``stepfun-ai/Step-3.5-Flash``'s config.json (``architectures[0]`` and
# ``model_type``). The bridge registry looks the model up by exact string
# match on these, so they must stay in sync with HF — only the Python class
# name (``Step35Bridge``) follows the new ``Step35`` spelling.
@MegatronModelBridge.register_bridge(
    source="Step3p5ForCausalLM",
    target=GPTModel,
    provider=Step35ModelProvider,
    model_type="step3p5",
)
class Step35Bridge(MegatronModelBridge):
    """
    Megatron Bridge for Step3.5 Causal LM.

    This bridge handles the conversion between HuggingFace Step3p5ForCausalLM
    (the HF architecture name; preserved verbatim to match the upstream
    config.json) and Megatron-Core GPTModel formats. Step3.5 models use
    mixture of experts architecture with QK layernorm.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("stepfun-ai/Step-3.5-Flash")
        >>> provider = bridge.to_megatron_provider()
    """

    CONFIG_MAPPING = MegatronModelBridge.CONFIG_MAPPING + [
        ("num_attention_groups", "num_query_groups"),
        ("moe_num_experts", "num_moe_experts"),
        ("moe_top_k", "moe_router_topk"),
        ("share_expert_dim", "moe_shared_expert_intermediate_size"),
        ("share_expert_dims", "moe_shared_expert_intermediate_size"),
        ("use_head_wise_attn_gate", "head_wise_attn_gate"),
        ("attention_output_gate", "attention_output_gate"),
        ("layer_types", "layer_types"),
    ]

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
        """Convert HuggingFace Step3.5 config to GPTModelProvider.

        Layered field-extraction strategy (mirrors the qwen3-vl bridge pattern):

        1. **Common architectural fields** — ``super().provider_bridge``
           internally calls :meth:`hf_config_to_provider_kwargs`, which uses
           ``hasattr`` + ``getattr(..., None)`` against :attr:`CONFIG_MAPPING`
           and silently skips fields the HF config doesn't carry. It also
           sets ``provider.position_embedding_type = "rope"`` (or ``"yarn"``)
           based on ``rope_scaling`` — without that, ``rotary_base_per_layer``
           later collides with the dataclass default ``"learned_absolute"``.

        2. **Step-3.5-specific fields** — applied below with explicit
           ``getattr(hf_config, name, default)`` for every field. This used
           to be 13 bare ``hf_config.X`` reads, which crashed when the bridge
           was reused via a wrapper against a Step3.7 ``text_config`` that
           dropped ``zero_centered``. The ``getattr`` form makes Step35Bridge
           safe to reuse against any HF config schema that may be missing
           Step-3.5-specific top-level fields, and the per-field defaults
           below are documented so the call site doesn't silently fall back
           to wrong values.
        """
        provider = super().provider_bridge(hf_pretrained)

        if getattr(provider, "head_wise_attn_gate", False):
            # Native head-wise gates and the full-head output gate are mutually
            # exclusive; MCore without native support needs the output-gate fallback.
            provider.attention_output_gate = not _mcore_supports_head_wise_attn_gate()

        hf_config = hf_pretrained.config
        mtp_layer_types = getattr(hf_config, "mtp_layer_types", None)
        if provider.layer_types is not None and mtp_layer_types:
            provider.layer_types = list(provider.layer_types) + list(mtp_layer_types)

        # ── Per-layer partial RoPE ────────────────────────────────────────
        partial_rotary_factors = getattr(hf_config, "partial_rotary_factors", None)
        if partial_rotary_factors is not None:
            provider.rotary_percents = list(partial_rotary_factors)

        # ── Sliding-attention shape (Step-3.5-Flash defaults; overlay HF) ─
        provider.sliding_attention_setting = {
            "window_size": [512, 0],
            "num_attention_heads": 96,
            "num_query_groups": 8,
            "kv_channels": 128,
        }
        sliding_window = getattr(hf_config, "sliding_window", None)
        if sliding_window is not None:
            provider.sliding_attention_setting["window_size"] = [int(sliding_window), 0]
        attn_other = getattr(hf_config, "attention_other_setting", None) or {}
        if isinstance(attn_other, dict) and attn_other.get("attention_type") == "sliding_attention":
            if "num_attention_heads" in attn_other:
                provider.sliding_attention_setting["num_attention_heads"] = int(attn_other["num_attention_heads"])
            if "num_attention_groups" in attn_other:
                provider.sliding_attention_setting["num_query_groups"] = int(attn_other["num_attention_groups"])
            if "head_dim" in attn_other:
                provider.sliding_attention_setting["kv_channels"] = int(attn_other["head_dim"])

        # ── Per-layer RoPE base (list) vs scalar ─────────────────────────
        rope_theta = getattr(hf_config, "rope_theta", None)
        if isinstance(rope_theta, list):
            provider.rotary_base = float(rope_theta[0])  # main model
            provider.rotary_base_per_layer = [float(x) for x in rope_theta]
        elif rope_theta is not None:
            provider.rotary_base = float(rope_theta)
        # else: leave provider.rotary_base at the framework-derived value.

        # ── Step-3.5 norm / attention defaults ────────────────────────────
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.hidden_dropout = 0.0
        provider.attention_dropout = 0.0
        # zero_centered defaults to True (Step-3.5 RMSNorm convention). Step-3.7
        # text_config doesn't carry this field, so the default kicks in there.
        provider.layernorm_zero_centered_gamma = bool(getattr(hf_config, "zero_centered", True))
        provider.qk_layernorm = bool(getattr(hf_config, "use_qk_norm", False))

        # ── Dtype: accept str / torch.dtype / missing ─────────────────────
        torch_dtype = getattr(hf_config, "torch_dtype", None)
        if isinstance(torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            if torch_dtype not in dtype_map:
                raise ValueError(f"Unknown torch dtype: {torch_dtype}")
            provider.autocast_dtype = dtype_map[torch_dtype]
        elif isinstance(torch_dtype, torch.dtype):
            provider.autocast_dtype = torch_dtype
        elif torch_dtype is not None:
            raise ValueError(f"Unknown torch dtype: {torch_dtype}")
        # else: leave provider.autocast_dtype at the dataclass default.

        # ── MoE settings ──────────────────────────────────────────────────
        provider.moe_router_enable_expert_bias = bool(getattr(hf_config, "use_moe_router_bias", False))
        moe_router_activation = getattr(hf_config, "moe_router_activation", None)
        if moe_router_activation is not None:
            provider.moe_router_score_function = moe_router_activation
        moe_router_scaling_factor = getattr(hf_config, "moe_router_scaling_factor", None)
        if moe_router_scaling_factor is not None:
            provider.moe_router_topk_scaling_factor = float(moe_router_scaling_factor)
        provider.swiglu_limits = getattr(hf_config, "swiglu_limits", None)
        provider.swiglu_limits_shared = getattr(hf_config, "swiglu_limits_shared", None)
        if bool(getattr(hf_config, "need_fp32_gate", False)):
            provider.moe_router_dtype = "fp32"

        provider.moe_grouped_gemm = True
        provider.moe_router_load_balancing_type = "aux_loss"
        provider.moe_aux_loss_coeff = 1e-3
        provider.moe_router_pre_softmax = False
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True

        # ── Per-layer dense vs MoE schedule + Step-3.5 layer spec ────────
        moe_layers_enum = getattr(hf_config, "moe_layers_enum", None)
        if moe_layers_enum is not None:
            moe_layer_freq = [0] * provider.num_layers
            if isinstance(moe_layers_enum, str):
                moe_layers = [int(layer) for layer in moe_layers_enum.split(",") if layer]
            else:
                moe_layers = [int(layer) for layer in moe_layers_enum]
            for idx in moe_layers:
                if 0 <= idx < provider.num_layers:
                    moe_layer_freq[idx] = 1
            provider.moe_layer_freq = moe_layer_freq
            # build_step35_layer_spec reads moe_layer_freq to produce per-layer dense/MoE
            # specs for the main decoder, and wraps layer_specs with _MTPDenseLayerSpecsList
            # so that get_gpt_mtp_block_spec_for_backend picks up a dense spec for MTP layers
            # (45-47 are not in moe_layers_enum).
            provider.transformer_layer_spec = build_step35_layer_spec

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Dictionary maps Megatron parameter names -> HF parameter names.
        # Supports wildcard (*) patterns for layer-specific parameters.
        param_mappings = {
            # Embedding and output
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            # Pre-attention layernorm (standalone for MoE layers; fused into linear_qkv for dense layers)
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            # Fused pre-attention layernorm weights (TELayerNormColumnParallelLinear).
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            # Layernorm for q, k
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            # Attention o projection
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            # Pre-MLP layernorm (standalone for dense layers; fused into linear_fc1 for dense layers)
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            # Dense MLP fc2 (layers 0–2)
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            # Shared expert fc2 (runs alongside routed experts on MoE layers)
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.share_expert.down_proj.weight",
            # MoE router
            "decoder.layers.*.mlp.router.weight": "model.layers.*.moe.gate.weight",
            # MoE router bias
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.moe.router_bias",
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        mapping_list.extend(
            [
                # QKV + per-head gate: select the native scalar-gate layout or
                # the expanded attention_output_gate fallback via the provider.
                QKVGMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                    g="model.layers.*.self_attn.g_proj.weight",
                ),
                # Dense MLP fc1 (gate+up concatenated; layers 0–2 and MTP layers 45–47)
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                # MoE per-expert fc1: Megatron creates weight0…weightN; HF stores stacked [N, I, H].
                StackedExpertGatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.moe.gate_proj.weight",
                    up="model.layers.*.moe.up_proj.weight",
                ),
                # Shared expert fc1 (gate+up concatenated). MCore names the shared
                # expert ``mlp.shared_experts`` (plural) — matches DeepSeek / GLM /
                # Sarvam bridges and is what TransformerLayerSubmodules expects.
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.share_expert.gate_proj.weight",
                    up="model.layers.*.share_expert.up_proj.weight",
                ),
                StackedExpertAutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.layers.*.moe.down_proj.weight",
                ),
            ]
        )

        # MTP layer mappings (layers 45–47 in Step-3.5-Flash)
        if self.hf_config is None:
            logger.warning("No HF config found, skipping MTP mappings.")
            return MegatronMappingRegistry(*mapping_list)

        mtp_num_layers = getattr(self.hf_config, "num_nextn_predict_layers", 0)
        num_transformer_layers = self.hf_config.num_hidden_layers

        # Layer-specific param patterns to replicate for each MTP transformer sub-layer.
        # Step3.5 MTP layers are always dense (no MoE), so only dense-MLP and attention params.
        # g_proj weight/layernorm are merged into linear_qkv via QKVGMapping
        # below (parallels the main decoder mapping table above).
        mtp_layer_param_mappings = {
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        }

        for mtp_layer in range(mtp_num_layers):
            hf_layer = mtp_layer + num_transformer_layers
            # Megatron may name the sub-layer "mtp_model_layer" or "transformer_layer".
            for layer_prefix in ("mtp_model_layer", "transformer_layer"):
                for megatron_param, hf_param in mtp_layer_param_mappings.items():
                    megatron_param_mtp = (
                        megatron_param.replace(".*", f".*.{layer_prefix}")
                        .replace("decoder", "mtp")
                        .replace(".*", f".{mtp_layer}")
                    )
                    hf_param_mtp = hf_param.replace("layers.*", f"layers.{hf_layer}")
                    mapping_list.append(AutoMapping(megatron_param=megatron_param_mtp, hf_param=hf_param_mtp))

                mapping_list.extend(
                    [
                        QKVGMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.self_attention.linear_qkv.weight",
                            q=f"model.layers.{hf_layer}.self_attn.q_proj.weight",
                            k=f"model.layers.{hf_layer}.self_attn.k_proj.weight",
                            v=f"model.layers.{hf_layer}.self_attn.v_proj.weight",
                            g=f"model.layers.{hf_layer}.self_attn.g_proj.weight",
                        ),
                        GatedMLPMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.linear_fc1.weight",
                            gate=f"model.layers.{hf_layer}.mlp.gate_proj.weight",
                            up=f"model.layers.{hf_layer}.mlp.up_proj.weight",
                        ),
                        AutoMapping(
                            megatron_param=f"mtp.layers.{mtp_layer}.{layer_prefix}.mlp.linear_fc2.weight",
                            hf_param=f"model.layers.{hf_layer}.mlp.down_proj.weight",
                        ),
                    ]
                )

            # MTP-specific normalization and projection layers
            mapping_list.extend(
                [
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.enorm.weight",
                        hf_param=f"model.layers.{hf_layer}.enorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.hnorm.weight",
                        hf_param=f"model.layers.{hf_layer}.hnorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.eh_proj.weight",
                        hf_param=f"model.layers.{hf_layer}.eh_proj.weight",
                    ),
                    # In Megatron, mtp use specific transformer.shared_head.norm different from main model,
                    # and share same transformer.shared_head.output.weight with main model
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.final_layernorm.weight",
                        hf_param=f"model.layers.{hf_layer}.transformer.shared_head.norm.weight",
                    ),
                ]
            )

        return MegatronMappingRegistry(*mapping_list)
