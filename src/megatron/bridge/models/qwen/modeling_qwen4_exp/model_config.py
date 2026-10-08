# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Declarative configuration for the Bridge-local Qwen4-Exp HybridModel."""

from dataclasses import dataclass
from typing import Any, ClassVar

from megatron.bridge.models.common.base import ModelConfig, ModelConfigOverrideMixin
from megatron.bridge.models.hybrid.hybrid_builder import HybridModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.utils.activation_map import callable_to_str, str_to_callable
from megatron.bridge.utils.instantiate_utils import _resolve_target


@dataclass
class Qwen4ExpTransformerConfig(TransformerConfig):
    """Transformer settings for QSA, gated residuals and lexical embeddings."""

    # Derived from the outer config for conversion helpers that inspect model.config.
    share_embeddings_and_output_weights: bool = False
    qwen4_mrope: bool = False
    apply_rotary_pos_emb_in_fp32: bool = False
    linear_attention_output_gate_activation: str | None = None
    mhc_variant: str = "gated_residual"
    mhc_gated_residual_rank: int | None = None
    qsa_indexer_n_heads: int | None = None
    qsa_indexer_kv_heads: int = 1
    qsa_indexer_head_dim: int | None = None
    qsa_indexer_budget: int | None = None
    qsa_indexer_compress_ratio: int | None = None
    qsa_force_sparse: bool = False
    ple_layer_ids: list[int] | None = None
    ple_embed_dim: int | None = None
    ple_conv_kernel_size: int = 4
    ple_ngram_size: int = 3
    ple_heads_per_ngram: int = 8
    ple_ngram_vocab_size_base: int = 20_000_000
    ple_ngram_vocab_divisible_by: int = 128
    ple_seed: int = 1234
    ple_eos_token_id: int | None = None
    ple_unigram_vocab_size: int | None = None


@dataclass(kw_only=True)
class Qwen4ExpModelConfig(ModelConfigOverrideMixin, HybridModelConfig, ModelConfig):
    """Build settings with one complete attention/MLP block per hybrid position."""

    builder: ClassVar[str] = "megatron.bridge.models.qwen.modeling_qwen4_exp.Qwen4ExpModelBuilder"
    transformer_config_class: ClassVar[type[TransformerConfig]] = Qwen4ExpTransformerConfig
    hybrid_attention_layers_include_mlp: ClassVar[bool] = True
    position_embedding_type: str = "rope"
    scatter_embedding_sequence_parallel: bool = True
    mrope_section: list[int] | None = None
    hf_model_text_only: bool = False
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    hetereogenous_dist_checkpoint: bool = True

    def _validate_qwen4_exp(self) -> None:
        if self.enable_mhc_connections:
            raise ValueError("Disable Core mHC expansion/contraction for the local Qwen4-Exp model.")
        if self.pipeline_model_parallel_size != 1 or self.context_parallel_size != 1:
            raise NotImplementedError("Qwen4-Exp currently requires PP=CP=1.")
        if self.virtual_pipeline_model_parallel_size is not None:
            raise NotImplementedError("Qwen4-Exp virtual pipeline stages are not supported.")
        if self.cuda_graph_impl != "none" or self.recompute_granularity is not None:
            raise NotImplementedError("Qwen4-Exp CUDA graphs and activation recomputation are not supported yet.")
        if self.mhc_recompute_layer_num is not None:
            raise NotImplementedError("Core mHC recomputation is not supported by Qwen4-Exp gated residuals.")
        if self.mtp_num_layers or self.fp8:
            raise NotImplementedError("Qwen4-Exp MTP and FP8 are not supported yet.")
        if (
            self.mhc_variant != "gated_residual"
            or not self.mhc_gated_residual_rank
            or self.mhc_gated_residual_rank < 1
        ):
            raise ValueError("Qwen4-Exp requires a positive gated-residual mixer rank.")
        if self.mhc_num_residual_streams < 1:
            raise ValueError("Qwen4-Exp requires at least one residual stream.")
        if self.linear_attention_output_gate_activation not in (None, "silu", "sigmoid"):
            raise ValueError("Qwen4-Exp output gate must be silu or sigmoid.")
        if self.qsa_indexer_n_heads is not None:
            for name in (
                "qsa_indexer_n_heads",
                "qsa_indexer_head_dim",
                "qsa_indexer_budget",
                "qsa_indexer_compress_ratio",
            ):
                value = getattr(self, name)
                if value is None or value <= 0:
                    raise ValueError(f"Qwen4-Exp requires a positive {name}.")
            if self.qsa_indexer_kv_heads != 1:
                raise ValueError("QSA requires one indexer key head.")
            # Narrow optional checkpoint fields after validating them above.
            assert self.qsa_indexer_budget is not None
            assert self.qsa_indexer_compress_ratio is not None
            assert self.qsa_indexer_head_dim is not None
            if self.qsa_indexer_budget % self.qsa_indexer_compress_ratio:
                raise ValueError("QSA budget must be divisible by its compression ratio.")
            if int(self.kv_channels * self.rotary_percent) > self.qsa_indexer_head_dim:
                raise ValueError("The attention RoPE width must fit in the QSA indexer head.")
        if self.ple_layer_ids:
            if len(set(self.ple_layer_ids)) != len(self.ple_layer_ids):
                raise ValueError("PLE layer IDs must be unique; their order defines the hash-table layout.")
            if any(layer < 1 or layer > self.num_layers for layer in self.ple_layer_ids):
                raise ValueError("PLE layer IDs must be one-indexed decoder layer numbers.")
            if self.ple_eos_token_id is None or not self.ple_unigram_vocab_size or self.ple_unigram_vocab_size < 1:
                raise ValueError("PLE requires EOS and unigram vocabulary settings.")
            heads = (self.ple_ngram_size - 1) * self.ple_heads_per_ngram
            if heads < 1 or (self.ple_embed_dim or self.hidden_size) % heads:
                raise ValueError("PLE embedding width must be divisible by its n-gram head count.")
            if min(self.ple_conv_kernel_size, self.ple_ngram_vocab_size_base, self.ple_ngram_vocab_divisible_by) < 1:
                raise ValueError("PLE convolution and vocabulary dimensions must be positive.")

    def finalize(self) -> None:
        """Resolve transformer settings and validate the supported hybrid architecture."""
        super().finalize()
        self.transformer.share_embeddings_and_output_weights = self.share_embeddings_and_output_weights
        self._validate_qwen4_exp()
        if self.position_embedding_type != "rope":
            raise ValueError("Qwen4-Exp requires rotary position embeddings.")
        frequency = self.linear_attention_freq
        if isinstance(frequency, int):
            if frequency < 1:
                raise ValueError("linear_attention_freq must be positive.")
            frequency = [int((i + 1) % frequency != 0) for i in range(self.num_layers)]
        if (
            not isinstance(frequency, list)
            or len(frequency) != self.num_layers
            or any(value not in (0, 1) for value in frequency)
        ):
            raise ValueError("linear_attention_freq must select one attention type per decoder layer.")
        pattern = "".join("G" if value else "*" for value in frequency)
        if self.hybrid_layer_pattern is not None and self.hybrid_layer_pattern != pattern:
            raise ValueError("hybrid_layer_pattern must match the linear_attention_freq schedule.")
        self.hybrid_layer_pattern = pattern
        if self.hybrid_override_pattern is not None or self.hybrid_attention_ratio or self.hybrid_mlp_ratio:
            raise ValueError("Use the Qwen4-Exp linear_attention_freq schedule rather than hybrid ratios.")
        if self.mrope_section is not None:
            width = int(self.kv_channels * self.rotary_percent)
            if len(self.mrope_section) != 3 or sum(self.mrope_section) * 2 != width:
                raise ValueError("mrope_section must partition the rotary head dimension into three axes.")

    def get_builder_cls(self) -> type:
        """Resolve the configured builder through Bridge's target allowlist."""
        builder_cls = _resolve_target(self.builder, full_key="_builder_")
        if not isinstance(builder_cls, type):
            raise TypeError(f"Builder target '{self.builder}' did not resolve to a class.")
        return builder_cls

    def as_dict(self) -> dict[str, Any]:
        """Serialize the config with a symbolic activation function."""
        data = super().as_dict()
        transformer_data = data.get("transformer")
        if not isinstance(transformer_data, dict):
            raise TypeError("Serialized Qwen4-Exp model config must contain a transformer mapping.")

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
    def from_dict(cls, data: dict[str, Any]) -> "Qwen4ExpModelConfig":
        """Deserialize a config while restoring its activation callable."""
        restored_data = dict(data)
        transformer_data = restored_data.get("transformer")
        if isinstance(transformer_data, dict):
            restored_transformer = dict(transformer_data)
            activation_name = restored_transformer.get("activation_func")
            if isinstance(activation_name, str):
                restored_transformer["activation_func"] = str_to_callable(activation_name)
            restored_data["transformer"] = restored_transformer

        result = ModelConfig.from_dict(restored_data)
        if not isinstance(result, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(result).__name__}.")
        return result
