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

"""Native Bridge conversion for the D1 LFM2 vision/language checkpoint."""

from copy import deepcopy

import torch
from torch.nn import functional as F
from transformers import Lfm2VlForConditionalGeneration, PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
from megatron.bridge.models.d1.d1_mappings import backbone_mappings
from megatron.bridge.models.d1.d1_vl_provider import D1VLModelProvider
from megatron.bridge.models.d1.modeling_d1.vl_model import D1VLModel
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


@MegatronModelBridge.register_bridge(
    source=Lfm2VlForConditionalGeneration, target=D1VLModel, provider=D1VLModelProvider, model_type="lfm2_vl"
)
class D1VLBridge(MegatronModelBridge):
    """Convert D1's causal LFM2-VL checkpoints into native MCore language modules."""

    hf_pretrained: PreTrainedCausalLM | PretrainedConfig | None = None

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> D1VLModelProvider:
        """Read all model dimensions from the source's nested language configuration.

        Args:
            hf_pretrained: Source config/artifact wrapper.

        Returns:
            Native VL provider with the released tied vocabulary semantics.
        """
        hf_config = hf_pretrained.config
        if not hf_config.tie_word_embeddings:
            raise ValueError("D1 VL requires tied input/output embeddings")
        text = hf_config.text_config
        if text.conv_L_cache != 3 or text.conv_bias:
            raise ValueError("D1 VL requires bias-free three-tap convolutions")
        width = text.intermediate_size
        if text.block_auto_adjust_ff_dim:
            width = int(2 * width / 3)
            if text.block_ffn_dim_multiplier is not None:
                width = int(width * text.block_ffn_dim_multiplier)
                multiple = text.block_multiple_of
                width = multiple * ((width + multiple - 1) // multiple)
        dtype = self.dtype_from_hf(hf_config, default=torch.float32)
        return D1VLModelProvider(
            num_layers=text.num_hidden_layers,
            hidden_size=text.hidden_size,
            ffn_hidden_size=width,
            num_attention_heads=text.num_attention_heads,
            num_query_groups=text.num_key_value_heads,
            kv_channels=text.hidden_size // text.num_attention_heads,
            vocab_size=text.vocab_size,
            make_vocab_size_divisible_by=1,
            seq_length=text.max_position_embeddings,
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
            d1_hf_config=deepcopy(hf_config.to_dict()),
        )

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Map native language blocks and unchanged HF vision/projector parameters."""
        vision_prefix = "model.vision_tower."
        # Transformers 5.17 flattens SigLIP2's vision_model wrapper. Inspect
        # source names so raw released checkpoints and current HF exports both
        # retain complete, strict parameter coverage without version guessing.
        if self.hf_pretrained is not None and hasattr(self.hf_pretrained, "state"):
            keys = self.hf_pretrained.state.source.get_all_keys()
            if any(name.startswith("model.vision_tower.vision_model.") for name in keys):
                vision_prefix += "vision_model."
        return MegatronMappingRegistry(
            *backbone_mappings(megatron_prefix="model.language_model", hf_prefix="model.language_model"),
            ReplicatedMapping("lm_head.weight", "lm_head.weight"),
            ReplicatedMapping("model.vision_tower.**", vision_prefix + "**"),
            ReplicatedMapping("model.multi_modal_projector.**", "model.multi_modal_projector.**"),
        )

    def _get_tied_output_hf_name(
        self, task: WeightConversionTask, mapping_registry: MegatronMappingRegistry
    ) -> str | None:
        # MCore's default hook recognizes its stock word_embeddings/output_layer
        # names. This composition preserves the published LFM2 names instead.
        if task.global_param_name == "model.language_model.embed_tokens.weight":
            return "lm_head.weight"
        return super()._get_tied_output_hf_name(task, mapping_registry)

    @classmethod
    def megatron_to_hf_config(cls, provider: D1VLModelProvider) -> dict[str, object]:
        """Preserve the published perception and remote-code settings on export.

        Args:
            provider: Provider retaining the original HF configuration.

        Returns:
            Complete HF configuration with the current checkpoint dtype.
        """
        if provider.d1_hf_config is None or not provider.share_embeddings_and_output_weights:
            raise ValueError("D1 VL export requires the original HF configuration and tied weights")
        result = deepcopy(provider.d1_hf_config)
        result["architectures"] = ["Lfm2VlForConditionalGeneration"]
        result["model_type"] = "lfm2_vl"
        result["tie_word_embeddings"] = True
        result["dtype"] = str(provider.params_dtype).removeprefix("torch.")
        result.pop("torch_dtype", None)
        return result
