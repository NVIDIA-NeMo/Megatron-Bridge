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

"""Hugging Face configuration classes for Inkling2 checkpoints.

The ``thinkingmachines/Inkling-2-Small`` release ships no configuration code, and the Transformers version Bridge
pins does not implement Inkling2 (its ``inkling`` model is the first Inkling generation, a different architecture),
so Bridge registers these classes for the release's ``inkling2`` model type. They only parse and serialize
``config.json``; the defaults are the release's values. A Transformers version that implements Inkling2
(huggingface/transformers#50682) keeps its own ``inkling2`` classes: they read every key of the release's
``config.json`` as an attribute too, and Bridge accepts them.
"""

from typing import Any

from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES


# Token ID of <|unused_200054|>, the image placeholder of the release's processor_config.json and chat template; the
# model replaces each occurrence with one image embedding.
INKLING2_IMAGE_TOKEN_ID = 200054


class Inkling2TextConfig(PretrainedConfig):
    """Language decoder configuration nested as ``text_config``."""

    model_type = "inkling2_model"

    def __init__(
        self,
        hidden_size: int = 4608,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        vocab_size: int = 201024,
        unpadded_vocab_size: int = 200058,
        d_rel: int = 16,
        rel_extent: int = 1024,
        q_bias: bool = False,
        o_bias: bool = False,
        log_scaling_n_floor: float = 128000,
        log_scaling_alpha: float = 0.1,
        rms_norm_eps: float = 1e-6,
        pre_norm_pre_affine: bool = False,
        final_norm_pre_affine: bool = False,
        use_value_norm_no_gamma: bool = True,
        qk_l2_clip_threshold: float = 22.627416997969522,
        rel_l2_clip_threshold: float = 8.0,
        local_layer_ids: list[int] | None = None,
        sconv_kernel_size: int = 4,
        final_logit_softcapping: float | None = 30.0,
        swa_head_dim: int | None = None,
        swa_num_attention_heads: int | None = None,
        swa_num_key_value_heads: int | None = None,
        sliding_window_size: int = 1024,
        n_layers_per_attn_res_block: int = 6,
        attn_res_normalize_values: bool = True,
        n_routed_experts: int = 384,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 0,
        intermediate_size: int = 2304,
        mlp_gate_softcap: float = 4.0,
        mlp_up_softcap: float = 25.0,
        route_scale: float = 8.0,
        use_gate_bias: bool = True,
        model_max_length: int = 1048576,
        **kwargs: Any,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.unpadded_vocab_size = unpadded_vocab_size
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.q_bias = q_bias
        self.o_bias = o_bias
        self.log_scaling_n_floor = log_scaling_n_floor
        self.log_scaling_alpha = log_scaling_alpha
        self.rms_norm_eps = rms_norm_eps
        self.pre_norm_pre_affine = pre_norm_pre_affine
        self.final_norm_pre_affine = final_norm_pre_affine
        self.use_value_norm_no_gamma = use_value_norm_no_gamma
        self.qk_l2_clip_threshold = qk_l2_clip_threshold
        self.rel_l2_clip_threshold = rel_l2_clip_threshold
        if local_layer_ids is None:
            # The release's layout: the last layer of each AttnRes block attends globally, the others locally.
            local_layer_ids = [i for i in range(num_hidden_layers) if (i + 1) % n_layers_per_attn_res_block]
        self.local_layer_ids = list(local_layer_ids)
        self.sconv_kernel_size = sconv_kernel_size
        self.final_logit_softcapping = final_logit_softcapping
        self.swa_head_dim = head_dim if swa_head_dim is None else swa_head_dim
        self.swa_num_attention_heads = (
            num_attention_heads if swa_num_attention_heads is None else swa_num_attention_heads
        )
        self.swa_num_key_value_heads = (
            num_key_value_heads if swa_num_key_value_heads is None else swa_num_key_value_heads
        )
        self.sliding_window_size = sliding_window_size
        self.n_layers_per_attn_res_block = n_layers_per_attn_res_block
        self.attn_res_normalize_values = attn_res_normalize_values
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.intermediate_size = intermediate_size
        self.mlp_gate_softcap = mlp_gate_softcap
        self.mlp_up_softcap = mlp_up_softcap
        self.route_scale = route_scale
        self.use_gate_bias = use_gate_bias
        self.model_max_length = model_max_length
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)


class Inkling2Config(PretrainedConfig):
    """Top-level Inkling2 configuration with text, vision, and draft-model sections.

    ``image_token_id`` is not in the release's ``config.json``; it defaults to the release's image placeholder,
    the ``image_token`` of its ``processor_config.json``.
    """

    model_type = "inkling2"
    sub_configs = {"text_config": Inkling2TextConfig}

    def __init__(
        self,
        text_config: dict[str, Any] | Inkling2TextConfig | None = None,
        vision_config: dict[str, Any] | None = None,
        mtp_config: dict[str, Any] | None = None,
        image_token_id: int = INKLING2_IMAGE_TOKEN_ID,
        **kwargs: Any,
    ) -> None:
        if isinstance(text_config, dict) or text_config is None:
            text_config = Inkling2TextConfig(**(text_config or {}))
        self.text_config = text_config
        self.vision_config = vision_config
        self.mtp_config = mtp_config
        self.image_token_id = image_token_id
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)


AutoConfig.register(Inkling2TextConfig.model_type, Inkling2TextConfig, exist_ok=True)
# Registering over a native implementation would hand Transformers' own Inkling2 models this class.
if Inkling2Config.model_type not in CONFIG_MAPPING_NAMES:
    AutoConfig.register(Inkling2Config.model_type, Inkling2Config, exist_ok=True)


__all__ = ["INKLING2_IMAGE_TOKEN_ID", "Inkling2Config", "Inkling2TextConfig"]
