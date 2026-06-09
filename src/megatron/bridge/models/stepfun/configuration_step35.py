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

from typing import Any, Optional

from transformers.configuration_utils import PretrainedConfig


def _split_mtp_layer_types(
    layer_types: list[str] | None,
    num_hidden_layers: int,
    num_nextn_predict_layers: int,
) -> tuple[list[str] | None, list[str] | None]:
    """Split published main-decoder + MTP layer types before parent validation."""
    if layer_types is None:
        return None, None

    total_layer_count = num_hidden_layers + num_nextn_predict_layers
    if num_nextn_predict_layers > 0 and len(layer_types) == total_layer_count:
        return layer_types[:num_hidden_layers], layer_types[num_hidden_layers:]

    return layer_types, None


class Step35Config(PretrainedConfig):
    """Configuration for the Step-3.5-Flash (``Step35``) Mixture-of-Experts model.

    Step35 is a decoder-only causal language model with grouped-query
    attention, rotary positional embeddings, and a configurable subset of
    transformer layers replaced by MoE FFN blocks (the remaining layers stay
    dense). It mirrors the architecture published under
    ``stepfun-ai/Step-3.5-Flash`` on the Hugging Face Hub.

    Args:
        hidden_size: Dimensionality of the hidden states.
        intermediate_size: Dimensionality of the dense FFN intermediate
            states (used by non-MoE layers).
        num_attention_heads: Number of query heads.
        num_attention_groups: Number of key/value head groups for GQA.
        num_hidden_layers: Total number of transformer layers.
        max_seq_len: Maximum sequence length supported by the model.
        vocab_size: Size of the tokenizer vocabulary.
        rms_norm_eps: Epsilon used by RMSNorm.
        moe_intermediate_size: Per-expert FFN intermediate size inside MoE
            layers.
        moe_num_experts: Number of routed experts per MoE layer.
        moe_top_k: Number of experts each token is routed to.
        rope_theta: Base period of the rotary embeddings.
        rope_scaling: Optional RoPE scaling configuration dict.
        max_position_embeddings: Maximum positions supported by RoPE.
        share_expert_dims: Hidden size of the shared-expert branch that runs
            alongside the routed experts.
        share_expert_dim: Singular alias used by the published Step-3.5-Flash
            config.
        head_dim: Per-head attention dimension.
        norm_expert_weight: Whether to normalize the top-k expert routing
            weights so they sum to 1.
        layer_types: Optional per-layer type override (e.g. attention
            variant); ``None`` uses the default for every layer.
        mtp_layer_types: Optional per-MTP-layer type override split from
            ``layer_types`` when the published config includes both main
            decoder and MTP entries.
        attention_other_setting: Sliding-attention shape override from the HF
            config.
        use_head_wise_attn_gate: Whether Step-3.5 uses per-head ``g_proj``
            gates fused into QKV.
        sliding_window: Sliding-window size for windowed attention; ``None``
            disables windowing.
        num_nextn_predict_layers: Number of MTP layers appended after the main
            decoder.
        moe_layers_enum: Indices of layers that use the MoE FFN. Layers not
            listed here use the dense FFN of ``intermediate_size``.
        **kwargs: Forwarded to :class:`~transformers.PretrainedConfig`.

    Note:
        ``model_type`` and ``architectures`` deliberately keep the
        ``step3p5`` / ``Step3p5ForCausalLM`` spelling so this config stays
        compatible with the ``config.json`` shipped on
        ``stepfun-ai/Step-3.5-Flash``. Only the Python class name uses the
        ``Step35`` spelling internally.
    """

    model_type = "step3p5"
    architectures = ["Step3p5ForCausalLM"]

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 11264,
        num_attention_heads: int = 64,
        num_attention_groups: int = 8,
        num_hidden_layers: int = 45,
        max_seq_len: int = 128000,
        vocab_size: int = 128815,
        rms_norm_eps: float = 1e-5,
        moe_intermediate_size: int = 1280,
        moe_num_experts: int = 288,
        moe_top_k: int = 8,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 128000,
        share_expert_dims: int = 1280,
        share_expert_dim: int | None = None,
        head_dim: int = 128,
        norm_expert_weight: bool = True,
        layer_types: list[str] | None = None,
        mtp_layer_types: list[str] | None = None,
        attention_other_setting: dict[str, Any] | None = None,
        use_head_wise_attn_gate: bool = True,
        sliding_window: Optional[int] = None,
        num_nextn_predict_layers: int = 3,
        moe_layers_enum: tuple[int] = (
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
        ),
        **kwargs,
    ) -> None:
        layer_types, inferred_mtp_layer_types = _split_mtp_layer_types(
            layer_types,
            num_hidden_layers,
            num_nextn_predict_layers,
        )
        if mtp_layer_types is None:
            mtp_layer_types = inferred_mtp_layer_types

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.num_hidden_layers = num_hidden_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings
        self.share_expert_dims = share_expert_dims
        self.share_expert_dim = share_expert_dims if share_expert_dim is None else share_expert_dim
        self.head_dim = head_dim
        self.norm_expert_weight = norm_expert_weight
        self.moe_layers_enum = moe_layers_enum
        self.layer_types = layer_types
        self.mtp_layer_types = mtp_layer_types
        self.attention_other_setting = attention_other_setting
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        self.sliding_window = sliding_window
        self.num_nextn_predict_layers = num_nextn_predict_layers
        super().__init__(**kwargs)
