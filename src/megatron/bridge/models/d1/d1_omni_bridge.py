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

"""AutoBridge registration and checkpoint conversion for D1 Omni."""

from copy import deepcopy

import torch
from torch.nn import functional as F
from transformers import Lfm2Config, PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
from megatron.bridge.models.d1.d1_mappings import backbone_mappings, decision_head_mappings
from megatron.bridge.models.d1.d1_omni_provider import D1OmniModelProvider
from megatron.bridge.models.d1.modeling_d1.omni_model import D1OmniModel
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


@MegatronModelBridge.register_bridge(
    source="D1OmniModel", target=D1OmniModel, provider=D1OmniModelProvider, model_type="d1_omni"
)
class D1OmniBridge(MegatronModelBridge):
    """Convert the published Omni checkpoint into native trainable Bridge modules."""

    hf_pretrained: PreTrainedCausalLM | PretrainedConfig | None = None

    def _is_adapter_param_name(self, param_name: str) -> bool:
        """Keep the published audio projector in base-model conversion tasks."""
        name = self._unwrap_name(param_name)
        if name.startswith("audio.adapter."):
            # Audio.adapter is the released speech projector. Nested PEFT
            # adapters still retain their own `.adapter.` segment.
            name = name.removeprefix("audio.")
        return super()._is_adapter_param_name(name)

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> D1OmniModelProvider:
        """Derive all language, head and perception dimensions from the source config.

        Args:
            hf_pretrained: Lazy HF artifact/config source selected by AutoBridge.

        Returns:
            Native Omni provider; model partitioning is restricted by its modules.
        """
        hf_config = hf_pretrained.config
        if getattr(hf_config, "tie_word_embeddings", False):
            raise ValueError("D1 Omni requires an independent marker head, with tie_word_embeddings=False")
        text_dict = hf_config.text_config
        if not isinstance(text_dict, dict):
            text_dict = text_dict.to_dict()
        text = Lfm2Config(**text_dict)
        if text.conv_L_cache != 3 or text.conv_bias:
            raise ValueError("D1 Omni requires bias-free three-tap short convolutions")
        # The released Omni encoder always adjusts the gated MLP width.
        width = int(2 * text.intermediate_size / 3)
        width = int(width * text_dict["block_ffn_dim_multiplier"])
        multiple = text_dict["block_multiple_of"]
        width = multiple * ((width + multiple - 1) // multiple)
        dtype = self.dtype_from_hf(hf_config, default=torch.float32)
        return D1OmniModelProvider(
            num_layers=text.num_hidden_layers,
            hidden_size=text.hidden_size,
            ffn_hidden_size=width,
            num_attention_heads=text.num_attention_heads,
            num_query_groups=text.num_key_value_heads,
            kv_channels=text.hidden_size // text.num_attention_heads,
            vocab_size=text.vocab_size,
            make_vocab_size_divisible_by=1,
            seq_length=hf_config.max_length,
            position_embedding_type="rope",
            rotary_base=text.rope_parameters["rope_theta"],
            normalization="RMSNorm",
            layernorm_epsilon=text.norm_eps,
            layernorm_zero_centered_gamma=False,
            qk_layernorm=True,
            gated_linear_unit=True,
            activation_func=F.silu,
            add_bias_linear=False,
            add_qkv_bias=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            apply_query_key_layer_scaling=False,
            attention_softmax_in_fp32=True,
            params_dtype=dtype,
            fp16=dtype == torch.float16,
            bf16=dtype == torch.bfloat16,
            d1_layer_types=tuple(text.layer_types),
            d1_vision_config=deepcopy(hf_config.vision_config),
            d1_audio_config=deepcopy(hf_config.audio_config),
            d1_projector_hidden_size=hf_config.projector_hidden_size,
            d1_head_layers=hf_config.head_layers,
            d1_max_length=hf_config.max_length,
            d1_hf_config=deepcopy(hf_config.to_dict()),
        )

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return full native language/head and unchanged HF perception mappings."""
        vision_prefix = "vision.tower."
        if self.hf_pretrained is not None and hasattr(self.hf_pretrained, "state"):
            keys = self.hf_pretrained.state.source.get_all_keys()
            if any(name.startswith("vision.tower.vision_model.") for name in keys):
                vision_prefix += "vision_model."
        return MegatronMappingRegistry(
            *backbone_mappings(megatron_prefix="encoder", hf_prefix="encoder"),
            *decision_head_mappings(megatron_prefix="head", hf_prefix="head"),
            ReplicatedMapping("vision.tower.**", vision_prefix + "**"),
            ReplicatedMapping("vision.projector.**", "vision.projector.**"),
            ReplicatedMapping("audio.**", "audio.**"),
        )

    @classmethod
    def megatron_to_hf_config(cls, provider: D1OmniModelProvider) -> dict[str, object]:
        """Preserve the published config and remote-code artifact contract on export.

        Args:
            provider: Provider retaining the original HF configuration.

        Returns:
            HF config dictionary with the current checkpoint dtype.
        """
        if provider.d1_hf_config is None:
            raise ValueError("Omni export requires the original HF configuration")
        result = deepcopy(provider.d1_hf_config)
        result["architectures"] = ["D1OmniModel"]
        result["model_type"] = "d1_omni"
        result["tie_word_embeddings"] = False
        result["dtype"] = str(provider.params_dtype).removeprefix("torch.")
        result.pop("torch_dtype", None)
        text = result["text_config"]
        if not isinstance(text, dict):
            raise ValueError("Omni export requires a dictionary text_config")
        # The released encoder reads this legacy key directly.
        text["rope_theta"] = provider.rotary_base
        return result
