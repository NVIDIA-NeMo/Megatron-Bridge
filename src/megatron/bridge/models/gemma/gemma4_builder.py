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

"""Model builders for Gemma 4 text models.

Both builders construct the same modules, in the same order, as the legacy
``Gemma4DenseProvider`` and ``Gemma4ModelProvider``.
"""

from typing import Any

from megatron.core.models.gpt import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import ModuleSpec
from megatron.training.models.gpt import GPTModelConfig as MCoreGPTModelConfig

from megatron.bridge.models.gemma.gemma3_provider import Gemma3LanguageModelEmbedding
from megatron.bridge.models.gemma.gemma4_config import Gemma4DenseModelConfig, Gemma4ModelConfig
from megatron.bridge.models.gemma.modeling_gemma4 import (
    HAVE_TE,
    Gemma4DenseRotaryEmbedding,
    Gemma4OutputLayer,
    Gemma4RotaryEmbedding,
    _attach_ple_modules,
    _install_ple_forward,
    _install_tied_kv,
    gemma4_block_spec,
    get_gemma4_layer_spec,
    install_gemma4_dense_load_state_aliases,
    wire_gemma4_kv_sharing,
)
from megatron.bridge.models.gemma.modules import extend_instance
from megatron.bridge.models.gpt.gpt_builder import GPTModelBuilder
from megatron.bridge.models.logit_dtype import logit_dtype_kwarg
from megatron.bridge.models.transformer_config import TransformerConfig


# The legacy dense provider hides these while MCore builds the GPT model; the builder does the same.
_DENSE_DUAL_ROPE_FIELDS = (
    "sliding_window_rope_base",
    "full_attention_rope_base",
    "full_attention_rope_partial_factor",
)


def gemma4_moe_layer_spec(config: MCoreGPTModelConfig | TransformerConfig) -> ModuleSpec:
    """Return the Gemma 4 MoE block spec for a builder config or its transformer config."""
    transformer_config = config.transformer if isinstance(config, MCoreGPTModelConfig) else config
    return gemma4_block_spec(transformer_config, use_transformer_engine=HAVE_TE)


def _install_softcapped_output_layer(model: GPTModel, config: TransformerConfig) -> None:
    if (
        hasattr(model, "output_layer")
        and config.final_logit_softcapping is not None
        and not isinstance(model.output_layer, Gemma4OutputLayer)
    ):
        extend_instance(model.output_layer, Gemma4OutputLayer)


def _split_rotary_base(rotary_base: Any) -> tuple[Any, Any]:
    if isinstance(rotary_base, (list, tuple)) and len(rotary_base) == 2:
        return rotary_base[0], rotary_base[1]
    raise ValueError(f"Gemma 4 MoE rotary_base must be (sliding base, global base), got {rotary_base!r}.")


class Gemma4DenseModelBuilder(GPTModelBuilder):
    """Construct Gemma 4 dense text models on MCore GPTModel."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> GPTModel:
        """Build a Gemma 4 dense GPTModel with dual RoPE, per-layer embeddings and shared KV.

        Args:
            pg_collection: Process groups used for distributed construction.
            pre_process: Whether to build the embeddings. Defaults to True.
            post_process: Whether to build the output layer. Defaults to True.
            vp_stage: Virtual pipeline stage; Gemma 4 dense supports none.

        Returns:
            The constructed model.
        """
        model_config = self._model_config
        if not isinstance(model_config, Gemma4DenseModelConfig):
            raise TypeError(f"Expected Gemma4DenseModelConfig, got {type(model_config).__name__}.")
        config = model_config.transformer
        if vp_stage is not None or config.pipeline_model_parallel_size != 1:
            raise NotImplementedError("Gemma 4 dense models currently support PP=1 only.")
        if config.context_parallel_size != 1:
            # The dense layer spec uses the local DotProductAttention, which asserts CP=1 deep
            # inside layer construction; fail here with a message that names the knob instead.
            raise NotImplementedError(
                "Gemma 4 dense models currently support CP=1 only: their layer spec uses the local "
                "DotProductAttention, which does not implement context parallelism."
            )
        if model_config.vocab_size is None:
            raise ValueError("Gemma 4 vocab_size must be configured before model construction.")

        divisor = model_config.make_vocab_size_divisible_by
        padded_vocab_size = (model_config.vocab_size + divisor - 1) // divisor * divisor
        pre_process = True if pre_process is None else pre_process
        post_process = True if post_process is None else post_process

        dual_rope = {name: getattr(config, name) for name in _DENSE_DUAL_ROPE_FIELDS}
        for name in dual_rope:
            setattr(config, name, None)
        try:
            model = GPTModel(
                config=config,
                transformer_layer_spec=get_gemma4_layer_spec(config),
                vocab_size=padded_vocab_size,
                max_sequence_length=model_config.seq_length,
                **logit_dtype_kwarg(GPTModel, model_config.logit_dtype),
                position_embedding_type=model_config.position_embedding_type,
                rotary_percent=model_config.rotary_percent,
                share_embeddings_and_output_weights=model_config.share_embeddings_and_output_weights,
                pre_process=pre_process,
                post_process=post_process,
                pg_collection=pg_collection,
            )
        finally:
            for name, value in dual_rope.items():
                setattr(config, name, value)

        model.rotary_pos_emb = Gemma4DenseRotaryEmbedding(config)
        if pre_process:
            _attach_ple_modules(model, config, config)
        wire_gemma4_kv_sharing(model)
        _install_ple_forward(model)
        install_gemma4_dense_load_state_aliases(model)
        _install_softcapped_output_layer(model, config)
        return model


class Gemma4ModelBuilder(GPTModelBuilder):
    """Construct Gemma 4 MoE text models on MCore GPTModel."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> GPTModel:
        """Build a Gemma 4 MoE GPTModel with dual RoPE, scaled embeddings and K=V tying.

        Args:
            pg_collection: Process groups used for distributed construction.
            pre_process: Whether to build the embeddings; defaults to the first pipeline stage.
            post_process: Whether to build the output layer; defaults to the last pipeline stage.
            vp_stage: Virtual pipeline stage.

        Returns:
            The constructed model stage.
        """
        model_config = self._model_config
        if not isinstance(model_config, Gemma4ModelConfig):
            raise TypeError(f"Expected Gemma4ModelConfig, got {type(model_config).__name__}.")
        rotary_base_local, rotary_base_global = _split_rotary_base(model_config.rotary_base)

        # MCore's GPT model takes a single RoPE base; build with the sliding base, as the legacy
        # provider does, then replace the rotary embedding with Gemma 4's dual one.
        overrides: dict[str, Any] = {"rotary_base": rotary_base_local}
        if model_config.transformer_layer_spec is None:
            overrides["transformer_layer_spec"] = gemma4_moe_layer_spec
        saved = {name: getattr(model_config, name) for name in overrides}
        for name, value in overrides.items():
            setattr(model_config, name, value)
        try:
            model = super().build_model(pg_collection, pre_process, post_process, vp_stage)
        finally:
            for name, value in saved.items():
                setattr(model_config, name, value)

        config = model_config.transformer
        if hasattr(model, "embedding"):
            model.embedding = Gemma3LanguageModelEmbedding(
                config=config,
                vocab_size=model.vocab_size,
                max_sequence_length=model_config.seq_length,
                position_embedding_type=model_config.position_embedding_type,
                scatter_to_sequence_parallel=model_config.scatter_embedding_sequence_parallel,
            )

        model.rotary_pos_emb = Gemma4RotaryEmbedding(
            kv_channels=config.kv_channels,
            rotary_percent=1.0,
            rotary_interleaved=config.rotary_interleaved,
            seq_len_interpolation_factor=model_config.seq_len_interpolation_factor,
            rotary_base=rotary_base_global,
            rope_scaling=False,
            use_cpu_initialization=config.use_cpu_initialization,
            rotary_base_local=rotary_base_local,
            global_kv_channels=config.global_head_dim,
            global_rotary_percent=config.global_rotary_percent,
        )

        _install_softcapped_output_layer(model, config)
        if hasattr(model, "embedding") or hasattr(model, "output_layer"):
            model.setup_embeddings_and_output_layer()

        _install_tied_kv(model, config)
        return model


__all__ = [
    "Gemma4DenseModelBuilder",
    "Gemma4ModelBuilder",
    "gemma4_moe_layer_spec",
]
