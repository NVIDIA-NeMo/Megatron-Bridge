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

from collections.abc import Mapping

import torch
from megatron.core.models.bert.bert_model import BertModel as MCoreBertModel
from transformers import MegatronBertForMaskedLM

from megatron.bridge.models.bert.bert_provider import BertModelProvider
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import AutoMapping, QKVMapping, ReplicatedMapping
from megatron.bridge.models.hf_pretrained.masked_lm import PreTrainedMaskedLM


@MegatronModelBridge.register_bridge(
    source=MegatronBertForMaskedLM, target=MCoreBertModel, provider=BertModelProvider, model_type="megatron-bert"
)
class BertBridge(MegatronModelBridge):
    """
    Megatron Bridge for HuggingFace ``MegatronBertForMaskedLM``.

    Targets Megatron-Core's stock, Pre-LayerNorm ``BertModel``. HuggingFace's
    vanilla ``transformers.BertModel`` (e.g. ``bert-base-uncased``) uses
    Post-LayerNorm and is a different architecture that this bridge does
    **not** support -- ``MegatronBertForMaskedLM`` (e.g.
    ``nvidia/megatron-bert-uncased-345m``) is the HuggingFace architecture
    that was purpose-built to mirror this Pre-LN layout.

    As a user you would not use this bridge directly, but through `AutoBridge`.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("nvidia/megatron-bert-uncased-345m")
        >>> provider = bridge.to_megatron_provider()
    """

    @staticmethod
    def _validate_hf_config(hf_config: object) -> None:
        hidden_act = getattr(hf_config, "hidden_act", "gelu")
        if hidden_act != "gelu":
            raise ValueError(
                "BertBridge only supports hidden_act='gelu' because Megatron-Core's BertLMHead "
                f"hardcodes GELU, but the Hugging Face config uses hidden_act={hidden_act!r}."
            )
        if getattr(hf_config, "is_decoder", False) or getattr(hf_config, "add_cross_attention", False):
            raise ValueError("BertBridge only supports encoder-only MegatronBertForMaskedLM configurations.")

    def provider_bridge(self, hf_pretrained: PreTrainedMaskedLM) -> BertModelProvider:
        hf_config = hf_pretrained.config
        self._validate_hf_config(hf_config)

        params_dtype = self.dtype_from_hf(hf_config, default=torch.float32)

        return BertModelProvider(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=hf_config.num_attention_heads,
            init_method_std=hf_config.initializer_range,
            layernorm_epsilon=hf_config.layer_norm_eps,
            max_position_embeddings=hf_config.max_position_embeddings,
            num_tokentypes=hf_config.type_vocab_size,
            vocab_size=hf_config.vocab_size,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),
            should_pad_vocab=False,
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", True),
            add_binary_head=False,
            activation_func=self.hf_to_megatron_activation(hf_config.hidden_act),
            hidden_dropout=hf_config.hidden_dropout_prob,
            attention_dropout=hf_config.attention_probs_dropout_prob,
            fp16=(params_dtype == torch.float16),
            bf16=(params_dtype == torch.bfloat16),
            params_dtype=params_dtype,
        )

    @classmethod
    def megatron_to_hf_config(cls, provider: BertModelProvider) -> dict[str, object]:
        """Convert a `BertModelProvider` back into a `MegatronBertConfig`-shaped dict.

        Not routed through the shared GPT-oriented `CONFIG_MAPPING` machinery
        (e.g. it maps `max_position_embeddings` to `seq_length` and
        `rms_norm_eps` to `layernorm_epsilon`, neither of which apply to BERT)
        -- BERT's config fields are mapped directly instead.
        """
        hidden_act = cls.megatron_to_hf_activation(provider.activation_func)
        if hidden_act != "gelu":
            raise ValueError(
                "BertBridge only supports GELU because Megatron-Core's BertLMHead hardcodes GELU, "
                f"but the provider uses activation_func={provider.activation_func!r}."
            )

        hf_config = {
            "num_hidden_layers": provider.num_layers,
            "hidden_size": provider.hidden_size,
            "intermediate_size": provider.ffn_hidden_size,
            "num_attention_heads": provider.num_attention_heads,
            "vocab_size": provider.vocab_size,
            "max_position_embeddings": provider.max_position_embeddings,
            "type_vocab_size": provider.num_tokentypes,
            "layer_norm_eps": provider.layernorm_epsilon,
            "initializer_range": provider.init_method_std,
            "tie_word_embeddings": provider.share_embeddings_and_output_weights,
            "hidden_act": hidden_act,
            "hidden_dropout_prob": provider.hidden_dropout,
            "attention_probs_dropout_prob": provider.attention_dropout,
        }

        if provider.bf16:
            hf_config["torch_dtype"] = "bfloat16"
        elif provider.fp16:
            hf_config["torch_dtype"] = "float16"
        else:
            hf_config["torch_dtype"] = "float32"

        if cls.SOURCE_NAME is not None:
            hf_config["architectures"] = [cls.SOURCE_NAME]
        if cls.MODEL_TYPE is not None:
            hf_config["model_type"] = cls.MODEL_TYPE

        return hf_config

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Dictionary maps Megatron parameter names -> HF parameter names.
        # Supports wildcard (*) patterns for per-layer parameters.
        param_mappings = {
            "embedding.word_embeddings.weight": "bert.embeddings.word_embeddings.weight",
            # Fused pre-attention / pre-MLP LayerNorm (folded into the following linear by the TE spec).
            "encoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "bert.encoder.layer.*.attention.ln.weight",
            "encoder.layers.*.self_attention.linear_qkv.layer_norm_bias": "bert.encoder.layer.*.attention.ln.bias",
            "encoder.layers.*.mlp.linear_fc1.layer_norm_weight": "bert.encoder.layer.*.ln.weight",
            "encoder.layers.*.mlp.linear_fc1.layer_norm_bias": "bert.encoder.layer.*.ln.bias",
            "encoder.layers.*.self_attention.linear_proj.weight": "bert.encoder.layer.*.attention.output.dense.weight",
            "encoder.layers.*.self_attention.linear_proj.bias": "bert.encoder.layer.*.attention.output.dense.bias",
            "encoder.layers.*.mlp.linear_fc1.weight": "bert.encoder.layer.*.intermediate.dense.weight",
            "encoder.layers.*.mlp.linear_fc1.bias": "bert.encoder.layer.*.intermediate.dense.bias",
            "encoder.layers.*.mlp.linear_fc2.weight": "bert.encoder.layer.*.output.dense.weight",
            "encoder.layers.*.mlp.linear_fc2.bias": "bert.encoder.layer.*.output.dense.bias",
            "encoder.final_layernorm.weight": "bert.encoder.ln.weight",
            "encoder.final_layernorm.bias": "bert.encoder.ln.bias",
            "lm_head.layer_norm.weight": "cls.predictions.transform.LayerNorm.weight",
            "lm_head.layer_norm.bias": "cls.predictions.transform.LayerNorm.bias",
            "output_layer.weight": "cls.predictions.decoder.weight",
            "output_layer.bias": "cls.predictions.bias",
        }

        mapping_list = [
            AutoMapping(megatron_param=megatron_param, hf_param=hf_param)
            for megatron_param, hf_param in param_mappings.items()
        ]

        mapping_list.extend(
            [
                # Plain nn.Embedding / nn.Linear modules (not tensor-parallel), unlike the
                # vocab-parallel word embeddings -- AutoMapping cannot auto-detect these, so
                # map explicitly.
                ReplicatedMapping(
                    megatron_param="embedding.position_embeddings.weight",
                    hf_param="bert.embeddings.position_embeddings.weight",
                ),
                ReplicatedMapping(
                    megatron_param="embedding.tokentype_embeddings.weight",
                    hf_param="bert.embeddings.token_type_embeddings.weight",
                ),
                ReplicatedMapping(
                    megatron_param="lm_head.dense.weight",
                    hf_param="cls.predictions.transform.dense.weight",
                ),
                ReplicatedMapping(
                    megatron_param="lm_head.dense.bias",
                    hf_param="cls.predictions.transform.dense.bias",
                ),
                QKVMapping(
                    megatron_param="encoder.layers.*.self_attention.linear_qkv.weight",
                    q="bert.encoder.layer.*.attention.self.query.weight",
                    k="bert.encoder.layer.*.attention.self.key.weight",
                    v="bert.encoder.layer.*.attention.self.value.weight",
                ),
                QKVMapping(
                    megatron_param="encoder.layers.*.self_attention.linear_qkv.bias",
                    q="bert.encoder.layer.*.attention.self.query.bias",
                    k="bert.encoder.layer.*.attention.self.key.bias",
                    v="bert.encoder.layer.*.attention.self.value.bias",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Duplicate the tied MLM-head bias into HF's second key on export.

        `MegatronBertLMPredictionHead` ties `cls.predictions.decoder.bias` to
        `cls.predictions.bias` (`self.decoder.bias = self.bias`, the same
        underlying `nn.Parameter`), so both keys appear in HF's state dict.
        Megatron only has a single `output_layer.bias`, so the second HF key
        must be synthesized here.
        """
        if task.global_param_name != "output_layer.bias":
            return converted_weights_dict

        bias_key = "cls.predictions.bias"
        decoder_bias_key = "cls.predictions.decoder.bias"
        if bias_key not in converted_weights_dict or decoder_bias_key in converted_weights_dict:
            return converted_weights_dict
        if hf_state_dict is not None and decoder_bias_key not in hf_state_dict:
            return converted_weights_dict

        converted_weights_dict[decoder_bias_key] = converted_weights_dict[bias_key]
        return converted_weights_dict
