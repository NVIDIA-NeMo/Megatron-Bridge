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

"""Beam model configuration: a ``HybridModel`` config plus the Beam architecture fields."""

from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

from megatron.bridge.models.beam.modeling_beam import validate_beam_config
from megatron.bridge.models.common import ModelConfigOverrideMixin, deserialize_model_config
from megatron.bridge.models.hybrid.hybrid_builder import HybridModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.utils.activation_map import callable_to_str, str_to_callable
from megatron.bridge.utils.instantiate_utils import _resolve_target


@dataclass
class BeamTransformerConfig(TransformerConfig):
    """Bridge ``TransformerConfig`` plus the Beam fields Megatron-Core does not declare.

    The modules in :mod:`~megatron.bridge.models.beam.modeling_beam` read them.
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
    ``idx = layer_number + 1`` with the hybrid stack's 1-based ``layer_number``."""

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


def beam_pipeline_layer_pattern(pattern: str, num_stages: int) -> str:
    """Split a Beam hybrid pattern into ``num_stages`` ``|``-separated segments.

    Beam layer ``i`` is the hybrid pair ``pattern[2i:2i+2]`` (attention, then MLP/MoE); pairs
    stay on one stage and earlier stages take the remainder. Existing separators are dropped,
    so a pattern saved under another pipeline size is re-split.
    """
    flat = pattern.replace(Symbols.PIPE, "")
    if len(flat) % 2:
        raise ValueError(f"Beam hybrid patterns pair every attention layer with an MLP layer, got {pattern!r}.")
    if num_stages <= 1:
        return flat
    blocks = [flat[i : i + 2] for i in range(0, len(flat), 2)]
    if len(blocks) < num_stages:
        raise ValueError(f"{len(blocks)} Beam layers cannot fill {num_stages} pipeline stages.")
    base, extra = divmod(len(blocks), num_stages)
    segments, offset = [], 0
    for stage in range(num_stages):
        count = base + int(stage < extra)
        segments.append("".join(blocks[offset : offset + count]))
        offset += count
    return Symbols.PIPE.join(segments)


@dataclass(kw_only=True)
class BeamModelConfig(ModelConfigOverrideMixin, HybridModelConfig):
    """``HybridModel`` config whose builder uses the Beam stack spec and soft-caps the output layer.

    ``hybrid_layer_pattern`` holds two hybrid layers per Beam layer: ``*`` attention followed
    by ``-`` (dense MLP) or ``E`` (MoE), so ``transformer.num_layers`` and the per-layer lists
    (``no_rope_freq``, ``window_attn_skip_freq``) count hybrid layers. ``_builder_`` names
    :class:`~.beam_builder.BeamModelBuilder`.
    """

    builder: ClassVar[str] = "megatron.bridge.models.beam.BeamModelBuilder"
    transformer_config_class: ClassVar[type[TransformerConfig]] = BeamTransformerConfig

    def finalize(self) -> None:
        """Split the pattern at Beam-layer boundaries for pipeline parallelism, then validate."""
        transformer = self.transformer
        num_stages = (transformer.pipeline_model_parallel_size or 1) * (
            transformer.virtual_pipeline_model_parallel_size or 1
        )
        if (
            self.hybrid_layer_pattern
            and transformer.num_layers_in_first_pipeline_stage is None
            and transformer.num_layers_in_last_pipeline_stage is None
        ):
            self.hybrid_layer_pattern = beam_pipeline_layer_pattern(self.hybrid_layer_pattern, num_stages)
        super().finalize()

    def get_builder_cls(self) -> type:
        """Resolve the Beam builder through Bridge's target allowlist."""
        builder_cls = _resolve_target(self.builder, full_key="_builder_")
        if not isinstance(builder_cls, type):
            raise TypeError(f"Builder target '{self.builder}' did not resolve to a class.")
        return builder_cls

    def as_dict(self) -> dict[str, Any]:
        """Serialize the config with a symbolic activation function."""
        data = super().as_dict()
        transformer_data = data.get("transformer")
        if not isinstance(transformer_data, dict):
            raise TypeError("Serialized Beam config must contain a transformer mapping.")
        activation_func = self.transformer.activation_func
        if isinstance(activation_func, str):
            str_to_callable(activation_func)
            activation_name = activation_func
        else:
            activation_name = callable_to_str(activation_func)
        if activation_name is None:
            raise ValueError(f"Cannot serialize unregistered activation callable: {activation_func!r}.")
        transformer_data["activation_func"] = activation_name
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BeamModelConfig":
        """Deserialize a Beam config and restore its activation callable."""
        restored_data = dict(data)
        transformer_data = restored_data.get("transformer")
        if isinstance(transformer_data, dict):
            restored_transformer = dict(transformer_data)
            activation_name = restored_transformer.get("activation_func")
            if isinstance(activation_name, str):
                restored_transformer["activation_func"] = str_to_callable(activation_name)
            restored_data["transformer"] = restored_transformer
        result = deserialize_model_config(restored_data)
        if not isinstance(result, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(result).__name__}.")
        return result
