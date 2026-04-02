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

"""
Megatron Bridge for NemotronDiffusion 3 (diffusion) Vision-Language Models.

Converts between HuggingFace Mistral3ForConditionalGeneration and
Megatron-Core GPTModel format, using NemotronDiffusionModelProvider which
replaces core attention with NemotronDiffusionAttention for sbd_block_diff.

Supported models:
- Ministral-3-3B-Base-2512
- Ministral-3-8B-Base-2512
"""

from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.diffusion.models.nemotron_diffusion.nemotron_diffusion_provider import (
    NemotronDiffusionModelProvider,
)
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM

# Ensure the base Ministral3Bridge is registered first so we can override it
from megatron.bridge.models.ministral3 import Ministral3Bridge  # noqa: F401


try:
    from transformers import Mistral3ForConditionalGeneration

    HAS_MISTRAL3 = True
except ImportError:
    Mistral3ForConditionalGeneration = None
    HAS_MISTRAL3 = False


# TODO: Check if NemotronDiffusion has a dedicated HuggingFace model class (e.g. NemotronDiffusionForCausalLM)
# and use it as source instead of Mistral3ForConditionalGeneration to avoid overwriting the
# Ministral3Bridge registration in get_model_bridge dispatch.
@MegatronModelBridge.register_bridge(source=Mistral3ForConditionalGeneration, target=GPTModel)
class NemotronDiffusionBridge(MegatronModelBridge):
    """HF <-> Megatron bridge for NemotronDiffusion diffusion language models."""

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> NemotronDiffusionModelProvider:
        hf_config = hf_pretrained.config
        text_config = getattr(hf_config, "text_config", hf_config)
        return NemotronDiffusionModelProvider(
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,
            num_layers=text_config.num_hidden_layers,
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", False),
            rotary_base=text_config.rope_parameters["rope_theta"],
            vocab_size=text_config.vocab_size,
            hf_config=hf_config,
        )

    def mapping_registry(self) -> MegatronMappingRegistry:
        param_mappings = {
            "language_model.embedding.word_embeddings.weight": "language_model.model.embed_tokens.weight",
            "language_model.output_layer.weight": "language_model.lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "language_model.model.norm.weight",
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "language_model.model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "language_model.model.layers.*.post_attention_layernorm.weight",
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "language_model.model.layers.*.self_attn.o_proj.weight",
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "language_model.model.layers.*.mlp.down_proj.weight",
        }

        mapping_list = [AutoMapping(megatron_param=k, hf_param=v) for k, v in param_mappings.items()]
        mapping_list.extend(
            [
                ReplicatedMapping(
                    megatron_param="vision_tower.**",
                    hf_param="vision_tower.**",
                ),
                ReplicatedMapping(
                    megatron_param="multi_modal_projector.**",
                    hf_param="multi_modal_projector.**",
                ),
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="language_model.model.layers.*.self_attn.q_proj.weight",
                    k="language_model.model.layers.*.self_attn.k_proj.weight",
                    v="language_model.model.layers.*.self_attn.v_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="language_model.model.layers.*.mlp.gate_proj.weight",
                    up="language_model.model.layers.*.mlp.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
