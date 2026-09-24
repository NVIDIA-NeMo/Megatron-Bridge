# SPDX-License-Identifier: Apache-2.0
"""Beam model configuration: the Beam architecture fields plus fp32 final-logit soft-capping."""

from dataclasses import dataclass
from typing import ClassVar, Optional

from megatron.bridge.models.beam.modeling_beam import validate_beam_config
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig


@dataclass
class BeamArchitectureConfig:
    """Beam fields that Megatron-Core's ``TransformerConfig`` does not declare.

    Mixed into :class:`BeamTransformerConfig` and ``BeamModelProvider``; the modules in
    :mod:`~megatron.bridge.models.beam.modeling_beam` read them.
    """

    attention_output_gate_bias: bool = False
    """Gate the attention output with ``sigmoid(gate + bias[head])``; requires ``attention_output_gate``.
    The bias lives in ``SelfAttention.output_gate_bias`` and is sharded across TP with the heads."""

    post_self_attn_layernorm: bool = False
    """Normalize the self-attention output before the residual add ("sandwich" norm)."""

    post_mlp_layernorm: bool = False
    """Normalize the MLP output before the residual add ("sandwich" norm)."""

    depth_scaled_residual: bool = False
    """Replace each residual add with ``out*(1/sqrt(idx)) + residual*sqrt((idx-1)/idx)``,
    ``idx = 2*layer_number + (0|1)``."""

    ssmax: bool = False
    """Scale queries on NoPE layers by ``(1 + (position + 1) / 2048) ** 0.1``; requires ``no_rope_freq``."""

    moe_router_fp32_params: bool = False
    """Keep router weights and biases in fp32, including through ``Float16Module``;
    requires ``moe_router_dtype='fp32'``."""

    final_logit_softcapping: Optional[float] = None
    """``logits = cap * tanh(logits / cap)`` on the output layer; ``None`` or ``0`` disables it."""

    def finalize(self) -> None:
        """Run the deferred Megatron-Core post-init, then validate the Beam fields."""
        super().finalize()
        validate_beam_config(self)


@dataclass
class BeamTransformerConfig(BeamArchitectureConfig, TransformerConfig):
    """Bridge ``TransformerConfig`` plus the Beam architecture fields."""


@dataclass(kw_only=True)
class BeamModelConfig(BridgeGPTModelConfig):
    """GPT model config whose builder uses the Beam layer spec and soft-caps the output layer.

    Serializes like :class:`BridgeGPTModelConfig`; ``_builder_`` names
    :class:`~.beam_builder.BeamModelBuilder`.
    """

    builder: ClassVar[str] = "megatron.bridge.models.beam.BeamModelBuilder"
    transformer_config_class: ClassVar[type[TransformerConfig]] = BeamTransformerConfig
