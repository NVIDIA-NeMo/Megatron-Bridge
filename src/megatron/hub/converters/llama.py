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

import logging
from functools import partial
from typing import TYPE_CHECKING

import torch

from megatron.hub.converters.common import BaseExporter, BaseImporter, dtype_from_hf
from megatron.hub.converters.state_transform import TransformFns, apply_transforms, state_transform
from megatron.hub.models.llama import Llama4Config, Llama31Config, LlamaConfig


if TYPE_CHECKING:
    from transformers import LlamaConfig as HFLlamaConfig
    from transformers import LlamaForCausalLM

logger = logging.getLogger(__name__)


class HFLlamaExporter(BaseExporter):
    """Exporter to convert megatron_hub Llama models to Hugging Face format."""

    def convert_state(self, source, target, source_config=None):
        # pylint: disable=C0301
        """Convert state dict from megatron_hub format to HF format.

        Maps the weights from the megatron_hub model to the HF model according to
        the appropriate mapping scheme.

        Args:
            source: Source megatron_hub model
            target: Target HF model
            source_config: Source megatron_hub config (optional, used for Llama4)

        Returns:
            The target model with weights transferred from source
        """
        is_llama4 = self.is_llama4()
        if is_llama4:
            assert source_config is not None
            source = self._modify_llama4_source_state(source, source_config)

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
                source_key="decoder.layers.*.mlp.linear_fc1.weight",
                target_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
                fn=TransformFns.split_fc1,
            ),
            state_transform(
                source_key="embedding.word_embeddings.weight",
                target_key="model.embed_tokens.weight",
                fn=TransformFns.prune_padding,
            ),
        ]
        if not self.config.tie_word_embeddings:
            transforms.append(
                state_transform(
                    source_key="output_layer.weight",
                    target_key="lm_head.weight",
                    fn=TransformFns.prune_padding,
                )
            )

        if is_llama4:
            # Llama4's Pre MLP LayerNorm is at decoder.layers.*.pre_mlp_layernorm.weight
            # instead of mlp.linear_fc1.layer_norm_weight
            mapping.pop("decoder.layers.*.mlp.linear_fc1.layer_norm_weight")
            mapping.update(
                {
                    "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
                    "decoder.layers.*.mlp.router.weight": "model.layers.*.feed_forward.router.weight",
                    "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.feed_forward.shared_expert.down_proj.weight",
                    "decoder.layers.*.mlp.experts.linear_fc2.weight": "model.layers.*.feed_forward.experts.down_proj",
                    "decoder.layers.*.mlp.experts.linear_fc1.weight": "model.layers.*.feed_forward.experts.gate_up_proj",
                }
            )
            transforms.extend(
                [
                    state_transform(
                        source_key="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                        target_key=(
                            "model.layers.*.feed_forward.shared_expert.gate_proj.weight",
                            "model.layers.*.feed_forward.shared_expert.up_proj.weight",
                        ),
                        fn=TransformFns.split_fc1,
                    ),
                    state_transform(
                        source_key="decoder.layers.*.mlp.linear_fc1.weight",
                        target_key=(
                            "model.layers.*.feed_forward.gate_proj.weight",
                            "model.layers.*.feed_forward.up_proj.weight",
                        ),
                        fn=TransformFns.split_fc1,
                    ),
                ]
            )

        return apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def hf_config(self) -> "HFLlamaConfig":
        """Generate a Hugging Face Llama configuration from the megatron_hub model configuration.

        This property maps megatron_hub configuration parameters to their Hugging Face equivalents.

        Returns:
            HFLlamaConfig: A Hugging Face Llama configuration
        """
        if self._hf_config is not None:
            return self._hf_config

        source = self.tron_config
        from transformers import LlamaConfig as HFLlamaConfig

        rope_scaling = None
        # For Llama 3.1 and Llama 3.2, rope_scaling is used and thus needed to parsed to the config
        if isinstance(source, Llama31Config):
            rope_scaling = {
                "factor": source.scale_factor,
                "low_freq_factor": source.low_freq_factor,
                "high_freq_factor": source.high_freq_factor,
                "original_max_position_embeddings": source.old_context_len,
                "rope_type": "llama3",
            }

        self._hf_config = HFLlamaConfig(
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
            tie_word_embeddings=source.share_embeddings_and_output_weights,
            rope_scaling=rope_scaling,
            bos_token_id=self.tokenizer.bos_token_id if self.tokenizer else None,
            eos_token_id=self.tokenizer.eos_token_id if self.tokenizer else None,
        )
        return self._hf_config


class HFLlamaImporter(BaseImporter):
    """Importer for converting Hugging Face Llama models to megatron_hub Tron format."""

    def init_hf_model(self) -> "LlamaForCausalLM":
        """Initialize the source Hugging Face Llama model.

        Returns:
            The initialized Hugging Face Llama model instance.
        """
        from transformers import LlamaForCausalLM

        return LlamaForCausalLM.from_pretrained(str(self.input_path), torch_dtype="auto")

    def convert_state(self, source, target):
        """Convert state dict from HF format to megatron_hub format.

        Maps the weights from the HF model to the megatron_hub model according to
        the appropriate mapping scheme.

        Args:
            source: Source HF model
            target: Target megatron_hub model

        Returns:
            The result of applying the transforms
        """
        mapping = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.norm.weight": "decoder.final_layernorm.weight",
            "lm_head.weight": "output_layer.weight",
        }
        if getattr(source.config, "tie_word_embeddings", False):
            # llama 3.2 1B and 3B models have no shared input output embeddings
            del mapping["lm_head.weight"]

        transforms = [
            state_transform(
                source_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                target_key="decoder.layers.*.self_attention.linear_qkv.weight",
                fn=TransformFns.merge_qkv,
            )
        ]
        if "llama4" in getattr(source.config, "model_type"):
            source = self._modify_llama4_source_state(source)
            # Update mapping for Llama4 model
            llama4_mapping = {
                # Post Attention LayerNorm
                "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.pre_mlp_layernorm.weight",
                "model.layers.*.dense-post_attention_layernorm.weight": (
                    "decoder.layers.*.mlp.linear_fc1.layer_norm_weight"
                ),
                # MoE Router
                "model.layers.*.feed_forward.router.weight": "decoder.layers.*.mlp.router.weight",
                # MoE Shared Experts
                "model.layers.*.feed_forward.shared_expert.down_proj.weight": (
                    "decoder.layers.*.mlp.shared_experts.linear_fc2.weight"
                ),
                # MoE Experts
                "model.layers.*.feed_forward.experts.*.down_proj": ("decoder.layers.*.mlp.experts.linear_fc2.weight*"),
                "model.layers.*.feed_forward.experts.*.gate_up_proj": (
                    "decoder.layers.*.mlp.experts.linear_fc1.weight*"
                ),
                # Dense MLP (for moe_layer_freq != 1)
                "model.layers.*.feed_forward.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            }

            # Update the main mapping dictionary with Llama4-specific mappings
            mapping.update(llama4_mapping)

            transforms.extend(
                [
                    state_transform(
                        source_key=(
                            "model.layers.*.feed_forward.shared_expert.gate_proj.weight",
                            "model.layers.*.feed_forward.shared_expert.up_proj.weight",
                        ),
                        target_key="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                        fn=TransformFns.merge_fc1,
                    ),
                    state_transform(
                        source_key=(
                            "model.layers.*.feed_forward.gate_proj.weight",
                            "model.layers.*.feed_forward.up_proj.weight",
                        ),
                        target_key="decoder.layers.*.mlp.linear_fc1.weight",
                        fn=TransformFns.merge_fc1,
                    ),
                ]
            )
        else:
            # Dense Mapping
            mapping.update(
                {
                    "model.layers.*.post_attention_layernorm.weight": (
                        "decoder.layers.*.mlp.linear_fc1.layer_norm_weight"
                    ),
                    "model.layers.*.mlp.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
                }
            )
            transforms.append(
                state_transform(
                    source_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
                    target_key="decoder.layers.*.mlp.linear_fc1.weight",
                    fn=TransformFns.merge_fc1,
                )
            )

        return apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def hf_config(self) -> "HFLlamaConfig":
        """Load and return the Hugging Face LlamaConfig from the input path.

        Returns:
            The loaded Hugging Face LlamaConfig instance.
        """
        from transformers import LlamaConfig as HFLlamaConfig

        if self._hf_config is not None:
            return self._hf_config
        self._hf_config = HFLlamaConfig.from_pretrained(str(self.input_path))
        return self._hf_config

    @property
    def tron_config(self) -> LlamaConfig:
        """Create a megatron_hub LlamaConfig from the HF model config.

        Translates the HF configuration parameters to the equivalent megatron_hub
        configuration.

        Returns:
            LlamaConfig: megatron_hub configuration for Llama models
        """
        if self._tron_config is not None:
            return self._tron_config

        from transformers import AutoConfig, GenerationConfig

        source = AutoConfig.from_pretrained(str(self.input_path))
        try:
            generation_config = GenerationConfig.from_pretrained(str(self.input_path))
        except Exception:
            generation_config = None

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        if getattr(source, "rope_scaling", None) is not None and source.rope_scaling.get("rope_type") == "llama3":
            # Apply Llama3.1 customize rope scaling
            cls = partial(Llama31Config, scale_factor=source.rope_scaling.get("factor", 8.0))
        else:
            cls = LlamaConfig

        args = {}
        if "llama4" in getattr(source, "model_type", None):
            # Llama4 Uses MoE Arch
            cls = Llama4Config
            # Parse Llama4 related args
            if getattr(source, "model_type", None) == "llama4":
                # Passing in a VL Llama4 Config
                # Only need to keep the text component
                source = source.text_config
            # for_llm_compressor
            # no_rope_layers
            args = {
                "moe_router_topk": source.num_experts_per_tok,
                "num_moe_experts": source.num_local_experts,
                "qk_l2_norm": source.use_qk_norm,
                "moe_shared_expert_intermediate_size": source.intermediate_size,
                "moe_ffn_hidden_size": source.intermediate_size,
            }
            if getattr(source, "rope_scaling", None) is not None and source.rope_scaling.get("rope_type") == "llama3":
                args.update({"rope_scaling": True, "rope_scaling_factor": source.rope_scaling.get("factor", 8.0)})
            else:
                args.update({"rope_scaling": False})
            if getattr(source, "interleave_moe_layer_step", 1) != 1:
                assert source.num_hidden_layers % source.interleave_moe_layer_step == 0
                pattern = [0] * (source.interleave_moe_layer_step - 1) + [1]
                num_patterns = source.num_hidden_layers // source.interleave_moe_layer_step
                args.update({"moe_layer_freq": pattern * num_patterns})

        output = cls(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=(
                source.intermediate_size
                if not getattr(source, "intermediate_size_mlp", None)
                else source.intermediate_size_mlp
            ),
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.rms_norm_eps,
            num_query_groups=source.num_key_value_heads,
            seq_length=source.max_position_embeddings,
            rotary_base=source.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=getattr(source, "tie_word_embeddings", False),
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
            generation_config=generation_config,
            vocab_size=source.vocab_size,
            kv_channels=getattr(source, "head_dim"),
            **args,
        )

        self._tron_config = output
        return self._tron_config
