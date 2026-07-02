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

"""Builder-backed model configurations for Gemma text models."""

import copy
from dataclasses import dataclass, field, replace
from typing import Callable, ClassVar, cast

from megatron.core.models.gpt import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import ModuleSpec
from megatron.training.models.gpt import GPTModelBuilder

from megatron.bridge.models.gemma.modeling_gemma2 import Gemma2OutputLayer, gemma2_layer_spec
from megatron.bridge.models.gemma.modeling_gemma3 import (
    Gemma3LanguageModelEmbedding,
    Gemma3RotaryEmbedding,
    gemma3_layer_spec,
)
from megatron.bridge.models.gemma.modeling_gemma4 import (
    Gemma4DenseRotaryEmbedding,
    Gemma4OutputLayer,
    Gemma4RotaryEmbedding,
    _attach_ple_modules,
    _gemma4_block_spec,
    _install_ple_forward,
    _install_tied_kv,
    get_gemma4_layer_spec,
    wire_gemma4_kv_sharing,
)
from megatron.bridge.models.gemma.modules import EmbeddingScalingMixin, extend_instance
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


def _install_gemma4_dense_load_state_aliases(model: object) -> None:
    """Translate dense sliding/global attention aliases during state loading."""
    if getattr(model, "_gemma4_dense_load_state_aliases_installed", False):
        return

    def _pre_hook(state_dict, prefix, *unused):
        del unused
        for key in list(state_dict):
            if prefix and not key.startswith(prefix):
                continue
            canonical = key.replace(".self_attention_sliding.", ".self_attention.")
            canonical = canonical.replace(".self_attention_global.", ".self_attention.")
            if canonical != key:
                state_dict.setdefault(canonical, state_dict[key])
                state_dict.pop(key)

    model._register_load_state_dict_pre_hook(_pre_hook)
    model._gemma4_dense_load_state_aliases_installed = True


@dataclass(kw_only=True)
class GemmaModelConfig(BridgeGPTModelConfig):
    """Pure model configuration for builder-backed Gemma models."""

    builder: ClassVar[str] = "megatron.bridge.models.gemma.model_config.GemmaModelBuilder"


class GemmaModelBuilder(GPTModelBuilder):
    """Build a Gemma model and install its embedding scaling behavior."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> GPTModel:
        """Build one Gemma pipeline stage.

        Args:
            pg_collection: Process groups for distributed model construction.
            pre_process: Whether the stage owns the embedding layer.
            post_process: Whether the stage owns the output layer.
            vp_stage: Virtual pipeline stage index.

        Returns:
            Constructed Gemma model stage.
        """
        model = super().build_model(
            pg_collection=pg_collection,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )
        if hasattr(model, "embedding"):
            extend_instance(model.embedding, EmbeddingScalingMixin)
        return model


@dataclass(kw_only=True)
class Gemma2ModelConfig(GemmaModelConfig):
    """Pure model configuration for builder-backed Gemma2 models."""

    builder: ClassVar[str] = "megatron.bridge.models.gemma.model_config.Gemma2ModelBuilder"
    transformer_layer_spec: Callable[..., ModuleSpec] = field(default_factory=lambda: gemma2_layer_spec)
    query_pre_attn_scalar: int = 224
    attn_logit_softcapping: float = 50.0
    final_logit_softcapping: float = 30.0


class Gemma2ModelBuilder(GemmaModelBuilder):
    """Build a Gemma2 model with embedding scaling and logit softcapping."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> GPTModel:
        """Build one Gemma2 pipeline stage.

        Args:
            pg_collection: Process groups for distributed model construction.
            pre_process: Whether the stage owns the embedding layer.
            post_process: Whether the stage owns the output layer.
            vp_stage: Virtual pipeline stage index.

        Returns:
            Constructed Gemma2 model stage.
        """
        config = cast(Gemma2ModelConfig, self._model_config)
        model = super().build_model(
            pg_collection=pg_collection,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )
        if hasattr(model, "output_layer"):
            extend_instance(model.output_layer, Gemma2OutputLayer)
            model.output_layer.final_logit_softcapping = config.final_logit_softcapping
        return model


@dataclass(kw_only=True)
class Gemma3ModelConfig(BridgeGPTModelConfig):
    """Pure model configuration for builder-backed Gemma3 text models."""

    builder: ClassVar[str] = "megatron.bridge.models.gemma.model_config.Gemma3ModelBuilder"
    transformer_layer_spec: Callable[..., ModuleSpec] = field(default_factory=lambda: gemma3_layer_spec)
    rotary_base_local: int = 10_000
    interleaved_attn_pattern: tuple[int, int] = (5, 1)
    is_vision_language: bool = False

    def __post_init__(self) -> None:
        """Restore tuple-valued runtime fields after JSON/YAML deserialization."""
        if isinstance(self.transformer.window_size, list):
            self.transformer.window_size = tuple(self.transformer.window_size)
        if isinstance(self.interleaved_attn_pattern, list):
            self.interleaved_attn_pattern = tuple(self.interleaved_attn_pattern)


class Gemma3ModelBuilder(GPTModelBuilder):
    """Build a Gemma3 text model and install its custom embedding and RoPE."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> GPTModel:
        """Build one Gemma3 pipeline stage.

        Args:
            pg_collection: Process groups for distributed model construction.
            pre_process: Whether the stage owns the embedding layer.
            post_process: Whether the stage owns the output layer.
            vp_stage: Virtual pipeline stage index.

        Returns:
            Constructed Gemma3 model stage.
        """
        model = super().build_model(
            pg_collection=pg_collection,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )
        config = cast(Gemma3ModelConfig, self._model_config)
        if hasattr(model, "embedding"):
            assert config.vocab_size is not None
            model.embedding = Gemma3LanguageModelEmbedding(
                config=config.transformer,
                vocab_size=config.vocab_size,
                max_sequence_length=config.seq_length,
                position_embedding_type=config.position_embedding_type,
                scatter_to_sequence_parallel=config.scatter_embedding_sequence_parallel,
            )
        model.rotary_pos_emb = Gemma3RotaryEmbedding(
            kv_channels=config.transformer.kv_channels,
            rotary_percent=config.rotary_percent,
            rotary_interleaved=config.transformer.rotary_interleaved,
            seq_len_interpolation_factor=config.seq_len_interpolation_factor,
            rotary_base=config.rotary_base,
            rope_scaling=False,
            rope_scaling_factor=config.rope_scaling_factor,
            use_cpu_initialization=config.transformer.use_cpu_initialization,
            rotary_base_local=config.rotary_base_local,
        )
        if hasattr(model, "embedding") or hasattr(model, "output_layer"):
            model.setup_embeddings_and_output_layer()
        return model


@dataclass(kw_only=True)
class Gemma4ModelConfig(BridgeGPTModelConfig):
    """Pure Gemma4 MoE construction state."""

    builder: ClassVar[str] = "megatron.bridge.models.gemma.model_config.Gemma4ModelBuilder"
    transformer_layer_spec: Callable[..., ModuleSpec] = field(default_factory=lambda: _gemma4_block_spec)
    rotary_base_local: float = 10_000.0
    interleaved_attn_pattern: tuple[int, int] = (5, 1)
    global_head_dim: int = 512
    num_global_key_value_heads: int = 2
    global_rotary_percent: float = 0.25
    attention_k_eq_v: bool = False
    final_logit_softcapping: float = 30.0


class Gemma4ModelBuilder(GPTModelBuilder):
    """Build Gemma4 MoE and install dual RoPE and tied-KV ownership."""

    def build_model(self, pg_collection, pre_process=None, post_process=None, vp_stage=None):
        """Build one Gemma4 MoE pipeline stage."""
        config = cast(Gemma4ModelConfig, self._model_config)
        model = super().build_model(pg_collection, pre_process, post_process, vp_stage)
        model.rotary_pos_emb = Gemma4RotaryEmbedding(
            kv_channels=config.transformer.kv_channels,
            rotary_percent=1.0,
            rotary_interleaved=config.transformer.rotary_interleaved,
            seq_len_interpolation_factor=config.seq_len_interpolation_factor,
            rotary_base=config.rotary_base,
            rope_scaling=False,
            use_cpu_initialization=config.transformer.use_cpu_initialization,
            rotary_base_local=config.rotary_base_local,
            global_kv_channels=config.global_head_dim,
            global_rotary_percent=config.global_rotary_percent,
        )
        if hasattr(model, "output_layer") and config.final_logit_softcapping:
            extend_instance(model.output_layer, Gemma4OutputLayer)
            model.output_layer.final_logit_softcapping = config.final_logit_softcapping
        _install_tied_kv(model, config)
        return model


@dataclass(kw_only=True)
class Gemma4DenseModelConfig(BridgeGPTModelConfig):
    """Pure Gemma4 Dense construction state including PLE and KV sharing."""

    builder: ClassVar[str] = "megatron.bridge.models.gemma.model_config.Gemma4DenseModelBuilder"
    global_kv_channels: int = 512
    num_global_query_groups: int = 2
    attention_k_eq_v: bool = False
    sliding_window_rope_base: float = 10_000.0
    full_attention_rope_base: float = 1_000_000.0
    full_attention_rope_partial_factor: float = 0.25
    num_kv_shared_layers: int = 0
    per_layer_embed_vocab_size: int = 0
    per_layer_embed_dim: int = 0
    window_attn_skip_freq: int | list[bool] = 6


class Gemma4DenseModelBuilder(GPTModelBuilder):
    """Build Gemma4 Dense with explicit PLE, KV-sharing, and dual RoPE state."""

    def build_model(self, pg_collection, pre_process=None, post_process=None, vp_stage=None):
        """Build one Gemma4 Dense pipeline stage."""
        config = cast(Gemma4DenseModelConfig, self._model_config)
        runtime_transformer = copy.copy(config.transformer)
        for name in (
            "global_kv_channels",
            "num_global_query_groups",
            "attention_k_eq_v",
            "num_kv_shared_layers",
            "per_layer_embed_vocab_size",
            "per_layer_embed_dim",
            "window_attn_skip_freq",
        ):
            setattr(runtime_transformer, name, getattr(config, name))
        runtime_config = replace(
            config,
            transformer=runtime_transformer,
            transformer_layer_spec=get_gemma4_layer_spec(runtime_transformer),
        )
        model = GPTModelBuilder(runtime_config).build_model(pg_collection, pre_process, post_process, vp_stage)
        model.rotary_pos_emb = Gemma4DenseRotaryEmbedding(config)
        if getattr(model, "pre_process", False):
            _attach_ple_modules(model, runtime_transformer, config)
        wire_gemma4_kv_sharing(model)
        _install_ple_forward(model)
        _install_gemma4_dense_load_state_aliases(model)
        return model


__all__ = [
    "Gemma2ModelBuilder",
    "Gemma2ModelConfig",
    "Gemma3ModelBuilder",
    "Gemma3ModelConfig",
    "Gemma4DenseModelBuilder",
    "Gemma4DenseModelConfig",
    "Gemma4ModelBuilder",
    "Gemma4ModelConfig",
    "GemmaModelBuilder",
    "GemmaModelConfig",
]
