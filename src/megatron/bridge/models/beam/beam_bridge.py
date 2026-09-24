# SPDX-License-Identifier: Apache-2.0
"""Beam config and checkpoint conversion.

Beam uses grouped sigmoid routing, sandwich norms, depth-scaled residuals,
full-width attention gates, SSMax on NoPE layers, and fp32 logit soft-capping.
Router parameters remain fp32; export with ``weight_dtype=None`` preserves them.
The expert-bias update rate defaults to zero for checkpoint fine-tuning. The modules
Megatron-Core lacks for these features live in :mod:`.modeling_beam`.
"""

from functools import partial
from typing import Any, Collection, Iterable, Mapping

import torch
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.beam.beam_config import BeamModelConfig
from megatron.bridge.models.beam.beam_mappings import BeamQKVGateMapping, split_qkv_gate_weights
from megatron.bridge.models.beam.beam_provider import BeamModelProvider
from megatron.bridge.models.beam.modeling_beam import beam_decoder_block_spec
from megatron.bridge.models.conversion import quantization_utils
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping
from megatron.bridge.models.conversion.peft_bridge import _split_hf_lora_weight_name


LAYER_TYPES = ("sliding_attention", "full_attention")
MLP_LAYER_TYPES = ("dense", "sparse")

# The soft-capped output layer uses column-parallel checkpoint sharding.
AutoMapping.register_module_type("BeamOutputLayer", "column")
# The per-head output-gate bias is sharded across TP ranks together with the heads.
AutoMapping.register_module_type("AttentionOutputGateBias", "column")
# The fp32 router subclass is replicated like Megatron-Core's TopKRouter.
AutoMapping.register_module_type("BeamTopKRouter", "replicated")


def _per_layer_types(hf_config, name: str, allowed: Collection[str]) -> list[str]:
    """Read a per-layer type list (``layer_types`` / ``mlp_layer_types``) and validate it."""
    values = getattr(hf_config, name, None)
    if not values:
        raise ValueError(f"Beam requires {name!r} with one entry per layer.")
    values = list(values)
    num_layers = hf_config.num_hidden_layers
    if len(values) != num_layers:
        raise ValueError(f"{name} has {len(values)} entries for {num_layers} layers.")
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"{name} contains {unknown}; expected only {list(allowed)}.")
    return values


def _moe_layer_pattern(hf_config) -> list[int]:
    """Return 1 for sparse layers and 0 for dense layers."""
    return [int(t == "sparse") for t in _per_layer_types(hf_config, "mlp_layer_types", MLP_LAYER_TYPES)]


def _rope_settings(hf_config) -> tuple[list[int] | None, float | None]:
    """``(no_rope_freq, rotary_base)`` from ``rope_parameters`` keyed by layer type.

    HF ``BeamModel`` computes RoPE for a layer type whose ``rope_parameters`` entry is a
    dict and runs NoPE + SSMax for a layer type whose entry is ``None``. Megatron marks
    NoPE per layer (``no_rope_freq[i]`` truthy = skip RoPE) and has a single ``rotary_base``,
    so every RoPE layer type must use plain RoPE with the same theta.

    ``no_rope_freq`` is ``None`` when every layer uses RoPE (core's default); ``rotary_base``
    is ``None`` when no layer does.
    """
    layer_types = _per_layer_types(hf_config, "layer_types", LAYER_TYPES)
    rope_parameters = getattr(hf_config, "rope_parameters", None)
    if not isinstance(rope_parameters, dict) or not set(layer_types) <= set(rope_parameters):
        raise ValueError(
            "Beam configs define rope_parameters keyed by layer type, e.g. "
            "{'sliding_attention': {'rope_type': 'default', 'rope_theta': 10000.0}, 'full_attention': None}; "
            f"got {rope_parameters!r} for layer types {sorted(set(layer_types))}."
        )
    thetas: set[float] = set()
    for layer_type in sorted(set(layer_types)):
        params = rope_parameters[layer_type]
        if params is None:
            continue
        rope_type = params.get("rope_type", "default")
        if rope_type != "default":
            raise ValueError(
                f"rope_parameters[{layer_type!r}] uses rope_type={rope_type!r}; only plain RoPE is supported."
            )
        if params.get("rope_theta") is None:
            raise ValueError(f"rope_parameters[{layer_type!r}] has no rope_theta.")
        thetas.add(float(params["rope_theta"]))
    if len(thetas) > 1:
        raise ValueError(
            f"Megatron has a single rotary_base, but the layer types use different rope_theta: {sorted(thetas)}."
        )
    no_rope_freq = [1 if rope_parameters[t] is None else 0 for t in layer_types]
    return (no_rope_freq if any(no_rope_freq) else None), (thetas.pop() if thetas else None)


def _beam_settings(hf_config) -> dict[str, Any]:
    """Megatron config settings for a ``BeamConfig``."""
    layer_types = _per_layer_types(hf_config, "layer_types", LAYER_TYPES)
    moe_layer_freq = _moe_layer_pattern(hf_config)
    no_rope_freq, rotary_base = _rope_settings(hf_config)

    if getattr(hf_config, "attention_bias", False):
        raise ValueError(
            "attention_bias=True is not supported: the bridge fuses q/k/v/gate_proj into a bias-free linear_qkv "
            "(BeamQKVGateMapping handles weights only) and Beam checkpoints ship attention_bias=False."
        )

    num_experts = getattr(hf_config, "num_experts", None) or 0
    if any(moe_layer_freq) and num_experts <= 0:
        raise ValueError("mlp_layer_types has 'sparse' layers but num_experts is not set.")
    if not getattr(hf_config, "norm_topk_prob", True):
        raise ValueError(
            "Beam's grouped sigmoid router is mapped onto moe_router_pre_softmax=False, which always "
            "renormalises the selected experts; norm_topk_prob=False has no megatron-core equivalent."
        )
    n_group = getattr(hf_config, "n_group", None)
    topk_group = getattr(hf_config, "topk_group", None)
    if n_group and topk_group and hf_config.num_experts_per_tok // topk_group != 2:
        raise ValueError(
            "Beam scores a routing group by the sum of its top-2 experts (hard-coded in "
            "BeamTopKRouter); megatron-core uses the sum of its top-(num_experts_per_tok // topk_group) "
            f"experts. The routers only agree when that ratio is 2, got {hf_config.num_experts_per_tok // topk_group}."
        )

    settings: dict[str, Any] = dict(
        normalization="RMSNorm",
        gated_linear_unit=True,
        add_bias_linear=False,
        hidden_dropout=0.0,
        qk_layernorm=True,  # q_norm / k_norm over head_dim
        attention_softmax_in_fp32=True,
        autocast_dtype=torch.bfloat16,
        share_embeddings_and_output_weights=bool(getattr(hf_config, "tie_word_embeddings", False)),
        position_embedding_type="rope",
        rotary_percent=1.0,
        attention_output_gate=True,
        add_qkv_bias=False,
        attention_output_gate_bias=True,
        post_self_attn_layernorm=True,
        post_mlp_layernorm=True,
        depth_scaled_residual=True,
        final_logit_softcapping=getattr(hf_config, "final_logit_softcapping", None),
    )
    if rotary_base is not None:
        # The generic rope_theta_from_hf() does not understand rope_parameters keyed by layer
        # type and silently falls back to default_theta, so set the base explicitly.
        settings["rotary_base"] = rotary_base

    if any(moe_layer_freq):
        settings.update(
            num_moe_experts=num_experts,
            moe_ffn_hidden_size=hf_config.moe_intermediate_size,
            moe_router_topk=hf_config.num_experts_per_tok,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_pre_softmax=False,  # norm_topk_prob=True, enforced above
            moe_router_dtype="fp32",
            moe_router_fp32_params=True,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=True,
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0.0,
            # Checkpoint fine-tuning keeps the expert bias fixed.
            moe_router_bias_update_rate=0.0,
            moe_layer_freq=moe_layer_freq,
        )
        if n_group:
            settings["moe_router_num_groups"] = n_group
        if topk_group:
            settings["moe_router_group_topk"] = topk_group
        if getattr(hf_config, "routed_scaling_factor", None) is not None:
            settings["moe_router_topk_scaling_factor"] = float(hf_config.routed_scaling_factor)
    else:
        # Dense models ignore the HF config's MoE defaults.
        settings.update(
            num_moe_experts=None,
            moe_ffn_hidden_size=None,
            moe_router_num_groups=None,
            moe_router_group_topk=None,
            moe_router_topk_scaling_factor=None,
        )

    # A truthy window_attn_skip_freq entry enables window attention on that layer.
    sliding = [1 if t == "sliding_attention" else 0 for t in layer_types]
    if any(sliding):
        sliding_window = getattr(hf_config, "sliding_window", None)
        if not sliding_window:
            raise ValueError("layer_types has 'sliding_attention' layers but sliding_window is not set.")
        # HF counts the current token in the window, Megatron does not
        settings["window_size"] = (sliding_window - 1, 0)
        settings["window_attn_skip_freq"] = sliding

    if no_rope_freq is not None:
        settings["no_rope_freq"] = no_rope_freq
        settings["ssmax"] = True
    return settings


#: MoE keys an all-dense export leaves to ``BeamConfig``'s defaults (see ``megatron_to_hf_config``).
_DENSE_EXPORT_DROPS = (
    "num_experts",
    "moe_intermediate_size",
    "num_experts_per_tok",
    "norm_topk_prob",
    "n_group",
    "topk_group",
    "routed_scaling_factor",
)


def _as_layer_list(value, num_layers: int, *, int_rule) -> list[int]:
    """Expand a Megatron ``int | list`` per-layer knob into a list of length ``num_layers``."""
    if isinstance(value, int):
        return [1 if int_rule(i, value) else 0 for i in range(num_layers)]
    return [1 if v else 0 for v in value]


@MegatronModelBridge.register_bridge(
    source="BeamForCausalLM",
    target=GPTModel,
    provider=BeamModelProvider,
    model_type="beam",
)
class BeamBridge(MegatronModelBridge):
    """Bridge between HuggingFace ``BeamForCausalLM`` and Megatron-Core ``GPTModel``.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("<beam-checkpoint>")
        >>> provider = bridge.to_megatron_provider()
    """

    MODEL_CONFIG_CLASS = BeamModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True

    def _adapter_target_modules(self, adapter_weight_names: Iterable[str]) -> list[str]:
        """Use exact paths: Beam's attention and MLP both contain gate_proj.

        This also matters for MLP-only adapters: a bare gate_proj would silently
        attach an unexported adapter to every attention gate and other layers.
        """
        return sorted(
            {_split_hf_lora_weight_name(name)[0].removeprefix("base_model.model.") for name in adapter_weight_names}
        )

    def _is_fused_qkv(self, hf_weight_names: Iterable[str]) -> bool:
        """Recognize Beam's four attention projections, including its independent gate."""
        names = list(hf_weight_names)
        expected = {"q_proj", "k_proj", "v_proj", "gate_proj"}
        return (
            len(names) == 4
            and {name.rsplit(".", 2)[-2] for name in names if name.endswith(".weight")} == expected
            and all(".self_attn." in name for name in names)
        )

    def _infer_qkv_projection_from_name(self, hf_name: str) -> str | None:
        """Keep attention gate_proj distinct from the MLP gate."""
        if hf_name.endswith(".self_attn.gate_proj.weight"):
            return "gate_proj"
        return super()._infer_qkv_projection_from_name(hf_name)

    def _split_qkv_linear_out_weight(
        self, megatron_model: GPTModel | list[GPTModel], linear_out_weight: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Split fused LoRA B along the same Q/gate/K/V rows as the base weight.

        The trailing dimension is the adapter rank, not the model hidden size.
        The export and merged-weight paths both use this hook; LoRA A is shared.
        """
        model = megatron_model[0] if isinstance(megatron_model, list) else megatron_model
        q, k, v, gate = split_qkv_gate_weights(model.config, linear_out_weight)
        return {"q_proj": q, "k_proj": k, "v_proj": v, "gate_proj": gate}

    # FP8_DYNAMIC stores one symmetric weight scale per output channel, shaped [out, 1].
    FP8_WEIGHT_SCALE_SUFFIX = "_scale"

    @staticmethod
    def get_hf_import_param_names(
        hf_param: str | dict[str, str],
        available_hf_param_names: set[str] | None = None,
    ) -> tuple[str, ...]:
        """Also declare the ``weight_scale`` companions so incremental imports wait for them."""
        names = MegatronModelBridge.get_hf_import_param_names(hf_param, available_hf_param_names)
        if available_hf_param_names is None:
            return names
        scales = [
            f"{name}{BeamBridge.FP8_WEIGHT_SCALE_SUFFIX}"
            for name in names
            if name.endswith(".weight") and f"{name}{BeamBridge.FP8_WEIGHT_SCALE_SUFFIX}" in available_hf_param_names
        ]
        return tuple(dict.fromkeys((*names, *scales)))

    def maybe_modify_loaded_hf_weight(
        self,
        hf_param: str | dict[str, str],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Dequantize per-output-channel FP8 weights on the fly; bf16 tensors pass through."""
        hf_weights = super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)
        if isinstance(hf_weights, dict):
            return {
                key: self._maybe_dequantize_fp8(tensor, hf_param[key], hf_state_dict)
                for key, tensor in hf_weights.items()
            }
        return self._maybe_dequantize_fp8(hf_weights, hf_param, hf_state_dict)

    @classmethod
    def _maybe_dequantize_fp8(
        cls,
        weight: torch.Tensor,
        param_name: str,
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """``fp8_weight * weight_scale`` in fp32 (one rounding when the mapping casts to the param dtype)."""
        if not quantization_utils.is_fp8_tensor(weight):
            return weight
        scale_key = f"{param_name}{cls.FP8_WEIGHT_SCALE_SUFFIX}"
        scale = hf_state_dict.get(scale_key)
        if scale is None:
            raise ValueError(
                f"{param_name} is stored as {weight.dtype} but {scale_key} is missing. Beam FP8 checkpoints must "
                "carry compressed-tensors FP8_DYNAMIC weight_scale tensors; a bare cast would silently corrupt "
                "the weights."
            )
        if weight.ndim != 2 or tuple(scale.shape) != (weight.shape[0], 1):
            raise ValueError(
                f"{scale_key} has shape {tuple(scale.shape)} for a weight of shape {tuple(weight.shape)}; expected a "
                "per-output-channel scale of shape [out_features, 1] (FP8_DYNAMIC). Block or per-tensor FP8 "
                "layouts are not supported by the Beam bridge."
            )
        return quantization_utils.maybe_dequantize_fp8(weight, scale, dtype=torch.float32)

    def hf_config_to_model_config_kwargs(self, hf_config) -> dict[str, Any]:
        """Flat kwargs for :class:`BeamModelConfig` (builder-backed path)."""
        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        config_kwargs.update(_beam_settings(hf_config))
        return config_kwargs

    @classmethod
    def megatron_to_hf_config(cls, provider) -> dict:
        """Invert the config mapping so an exported checkpoint loads as ``BeamForCausalLM``.

        ``provider`` is either a :class:`BeamModelProvider` or a
        :class:`BeamModelConfig`; the latter proxies attribute access to its nested
        transformer config, so both read the same way.
        """
        hf_config = super().megatron_to_hf_config(provider)
        num_layers = provider.num_layers

        # linear_qkv has no bias; gate_bias is its own parameter.
        hf_config["attention_bias"] = False
        if provider.window_size is not None:
            hf_config["sliding_window"] = provider.window_size[0] + 1
            skip = provider.window_attn_skip_freq
            if skip is None:
                sliding = [1] * num_layers  # window_size alone slides every layer
            else:
                # int form: every skip-th layer is full attention, like core's is_layer_window_attention
                sliding = _as_layer_list(skip, num_layers, int_rule=lambda i, k: (i + 1) % k != 0)
        else:
            hf_config["sliding_window"] = None
            sliding = [0] * num_layers
        layer_types = ["sliding_attention" if s else "full_attention" for s in sliding]
        hf_config["layer_types"] = layer_types

        # rope_parameters keyed by layer type; None marks a NoPE (SSMax) layer type. Beam ties
        # RoPE/NoPE to the layer type, so all layers of one type must agree.
        no_rope = provider.no_rope_freq
        if no_rope is None:
            skips = [0] * num_layers
        else:
            skips = _as_layer_list(no_rope, num_layers, int_rule=lambda i, k: (i + 1) % k == 0)
        rope_parameters: dict[str, dict | None] = {}
        for layer_type in LAYER_TYPES:
            flags = {skips[i] for i, t in enumerate(layer_types) if t == layer_type}
            if not flags:
                continue
            if len(flags) > 1:
                raise ValueError(
                    f"Beam keys RoPE/NoPE by layer type, but the {layer_type} layers mix RoPE and NoPE "
                    "(no_rope_freq disagrees within the type); this model is not representable as BeamConfig."
                )
            rope_parameters[layer_type] = (
                None if flags.pop() else {"rope_type": "default", "rope_theta": float(provider.rotary_base)}
            )
        hf_config["rope_parameters"] = rope_parameters
        # The generic CONFIG_MAPPING emits flat RoPE keys and expert-count aliases Beam does not
        # declare; drop them so the exported config.json carries only the Beam schema.
        for key in ("rope_theta", "partial_rotary_factor", "num_local_experts", "n_routed_experts"):
            hf_config.pop(key, None)

        if provider.num_moe_experts:
            # int form mirrors get_gpt_decoder_block_spec: layer i is MoE when i % freq == 0
            pattern = _as_layer_list(provider.moe_layer_freq, num_layers, int_rule=lambda i, k: i % k == 0)
            hf_config["norm_topk_prob"] = not provider.moe_router_pre_softmax
            hf_config["n_group"] = provider.moe_router_num_groups
            hf_config["topk_group"] = provider.moe_router_group_topk
            hf_config["routed_scaling_factor"] = provider.moe_router_topk_scaling_factor
        else:
            # BeamConfig requires typed MoE defaults even when no layer uses experts.
            pattern = [0] * num_layers
            for key in _DENSE_EXPORT_DROPS:
                hf_config.pop(key, None)
        hf_config["mlp_layer_types"] = ["sparse" if is_moe else "dense" for is_moe in pattern]

        hf_config["final_logit_softcapping"] = provider.final_logit_softcapping
        return hf_config

    def provider_bridge(self, hf_pretrained) -> BeamModelProvider:
        """Convert ``BeamConfig`` to a :class:`BeamModelProvider`."""
        provider = super().provider_bridge(hf_pretrained)
        for name, value in _beam_settings(hf_pretrained.config).items():
            setattr(provider, name, value)
        # Megatron-Core's decoder block spec (mixed dense and MoE layers) rewritten to the Beam modules.
        provider.transformer_layer_spec = partial(beam_decoder_block_spec, use_transformer_engine=HAVE_TE)
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Map attention, norms, dense MLPs, and MoE parameters by module path."""
        direct = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.output_gate_bias.bias": "model.layers.*.self_attn.gate_bias",
            "decoder.layers.*.post_self_attn_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.pre_feedforward_layernorm.weight",
            "decoder.layers.*.post_mlp_layernorm.weight": "model.layers.*.post_feedforward_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.pre_feedforward_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
        }
        return MegatronMappingRegistry(
            *(AutoMapping(megatron_param=m, hf_param=h) for m, h in direct.items()),
            BeamQKVGateMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
                g="model.layers.*.self_attn.gate_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                up="model.layers.*.mlp.experts.*.up_proj.weight",
            ),
        )
