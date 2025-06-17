# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import TYPE_CHECKING, Any

import torch

from megatron.hub.converters.common import BaseExporter, BaseImporter, dtype_from_hf
from megatron.hub.converters.state_transform import TransformFns, apply_transforms, state_transform
from megatron.hub.models.qwen2 import Qwen2Config


if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, PretrainedConfig


class HFQwen2Exporter(BaseExporter):
    """Exporter to convert megatron hub Qwen2 models to Hugging Face format."""

    def convert_state(self, source: Any, target: Any) -> Any:
        """Convert the state dict from the source megatron hub model to the target HF model.

        Args:
            source: A helper object (_ModelState) containing the loaded megatron hub state dict.
            target: The target Hugging Face model instance.

        Returns:
            The target model with weights transferred from source.
        """
        mapping = {
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
        }

        transforms = [
            state_transform(
                source_key="decoder.layers.*.self_attention.linear_qkv.weight",
                target_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                fn=TransformFns.split_qkv,
            ),
            state_transform(
                source_key="decoder.layers.*.self_attention.linear_qkv.bias",
                target_key=(
                    "model.layers.*.self_attn.q_proj.bias",
                    "model.layers.*.self_attn.k_proj.bias",
                    "model.layers.*.self_attn.v_proj.bias",
                ),
                fn=TransformFns.split_qkv_bias,
            ),
            state_transform(
                source_key="decoder.layers.*.mlp.linear_fc1.weight",
                target_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
                fn=TransformFns.split_fc1,
            ),
            state_transform(
                source_key="embedding.word_embeddings.weight",
                target_key="model.embed_tokens.weight",
                fn=TransformFns.prune_padding,
            ),
            state_transform(
                source_key="output_layer.weight",
                target_key="lm_head.weight",
                fn=TransformFns.prune_padding,
            ),
        ]

        return apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def hf_config(self) -> "PretrainedConfig":
        """Generate a Hugging Face Qwen2 configuration from the megatron hub model configuration.

        This property maps megatron hub configuration parameters to their Hugging Face equivalents.

        Returns:
            HFQwen2Config: A Hugging Face Qwen2 configuration
        """
        if self._hf_config is not None:
            return self._hf_config

        from transformers import Qwen2Config as HFQwen2Config

        source = self.tron_config

        self._hf_config = HFQwen2Config(
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            rope_theta=source.rotary_base,
            vocab_size=source.vocab_size,
            sliding_window=source.seq_length,
            tie_word_embeddings=False,
        )
        return self._hf_config


class HFQwen2Importer(BaseImporter):
    """Importer for converting Hugging Face Qwen2 models to megatron hub format."""

    def init_hf_model(self) -> "AutoModelForCausalLM":
        """Initialize the source Hugging Face Qwen2 model.

        Returns:
            The initialized Hugging Face Qwen2 model instance.
        """
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(str(self.input_path), torch_dtype="auto", trust_remote_code=True)

    def convert_state(self, source: Any, target: Any) -> None:
        """Convert the state dict from the source HF model to the target megatron model.

        Args:
            source: The source Hugging Face model instance.
            target: The target megatron model instance.
        """
        mapping = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.mlp.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "model.norm.weight": "decoder.final_layernorm.weight",
            "lm_head.weight": "output_layer.weight",
        }

        transforms = [
            state_transform(
                source_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                target_key="decoder.layers.*.self_attention.linear_qkv.weight",
                fn=TransformFns.merge_qkv,
            ),
            state_transform(
                source_key=(
                    "model.layers.*.self_attn.q_proj.bias",
                    "model.layers.*.self_attn.k_proj.bias",
                    "model.layers.*.self_attn.v_proj.bias",
                ),
                target_key="decoder.layers.*.self_attention.linear_qkv.bias",
                fn=TransformFns.merge_qkv_bias,
            ),
            state_transform(
                source_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
                target_key="decoder.layers.*.mlp.linear_fc1.weight",
                fn=TransformFns.merge_fc1,
            ),
        ]
        apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def hf_config(self) -> "PretrainedConfig":
        """Load and return the Hugging Face Qwen2Config from the input path.

        Returns:
            The loaded Hugging Face Qwen2Config instance.
        """
        from transformers import Qwen2Config as HFQwen2Config

        if self._hf_config is not None:
            return self._hf_config
        self._hf_config = HFQwen2Config.from_pretrained(str(self.input_path), trust_remote_code=True)
        return self._hf_config

    @property
    def tron_config(self) -> Qwen2Config:
        """Create a megatron hub Qwen2Config from the HF model config.

        Translates the HF configuration parameters to the equivalent megatron hub
        configuration.

        Returns:
            Qwen2Config: megatron hub configuration for Qwen2 models
        """
        if self._tron_config is not None:
            return self._tron_config

        from transformers import GenerationConfig

        source = self.hf_config
        generation_config = GenerationConfig.from_pretrained(str(self.input_path))

        self._tron_config = Qwen2Config(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            num_query_groups=source.num_key_value_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=128,
            rotary_base=source.rope_theta,
            share_embeddings_and_output_weights=False,
            vocab_size=source.vocab_size,
            seq_length=source.max_position_embeddings,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
            generation_config=generation_config,
        )
        return self._tron_config
