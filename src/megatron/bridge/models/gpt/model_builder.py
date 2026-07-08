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

"""GPT builders for Bridge-selected layer specifications."""

import inspect

from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.dot_product_attention import DotProductAttention as MCoreDotProductAttention
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.training.models.gpt import GPTModelBuilder, default_layer_spec, mtp_block_spec as default_mtp_block_spec
from megatron.training.vocab_utils import calculate_padded_vocab_size

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


def mtp_block_spec_from_layer_spec(
    config: BridgeGPTModelConfig,
    transformer_layer_spec: ModuleSpec | TransformerBlockSubmodules,
    vp_stage: int | None = None,
) -> ModuleSpec | None:
    """Build an MTP block from the decoder spec selected by a Bridge."""
    if not config.transformer.mtp_num_layers:
        return None

    decoder_spec = transformer_layer_spec
    if hasattr(transformer_layer_spec, "layer_specs"):
        layer_specs = transformer_layer_spec.layer_specs
        try:
            decoder_spec = layer_specs[-1]
        except IndexError:
            # A pipeline stage can own no decoder layers. Defer to MCore's
            # backend-aware fallback for an ordinary empty list; specialized
            # containers such as Step's dense-MTP list override ``[-1]``.
            return default_mtp_block_spec(config, transformer_layer_spec, vp_stage=vp_stage)

    return get_gpt_mtp_block_spec(
        config.transformer,
        decoder_spec,
        use_transformer_engine=config.transformer.transformer_impl == "transformer_engine",
        vp_stage=vp_stage,
    )


class LayerSpecGPTModelBuilder(GPTModelBuilder):
    """Build GPT models whose supplied decoder spec must also drive MTP."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> GPTModel:
        """Build one GPT pipeline stage using the configured layer spec."""
        config = self._model_config
        assert isinstance(config, BridgeGPTModelConfig)

        layer_spec = config.transformer_layer_spec
        if layer_spec is None:
            layer_spec = default_layer_spec(config, vp_stage)
        elif not isinstance(layer_spec, ModuleSpec) and callable(layer_spec):
            if "vp_stage" in inspect.signature(layer_spec).parameters:
                layer_spec = layer_spec(config, vp_stage=vp_stage)
            else:
                layer_spec = layer_spec(config)

        mtp_spec = mtp_block_spec_from_layer_spec(config, layer_spec, vp_stage=vp_stage)
        if config.attention_backend == AttnBackend.local and hasattr(layer_spec, "submodules"):
            layer_spec.submodules.self_attention.submodules.core_attention = MCoreDotProductAttention

        assert config.vocab_size is not None
        vocab_size = config.vocab_size
        if config.should_pad_vocab:
            vocab_size = calculate_padded_vocab_size(
                vocab_size,
                config.make_vocab_size_divisible_by,
                config.transformer.tensor_model_parallel_size,
            )

        vp_size = config.transformer.virtual_pipeline_model_parallel_size
        if pre_process is None:
            pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(pg_collection.pp)
        if post_process is None:
            post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(pg_collection.pp)

        return GPTModel(
            config=config.transformer,
            transformer_layer_spec=layer_spec,
            mtp_block_spec=mtp_spec,
            vocab_size=vocab_size,
            max_sequence_length=config.seq_length,
            fp16_lm_cross_entropy=config.fp16_lm_cross_entropy,
            parallel_output=config.parallel_output,
            share_embeddings_and_output_weights=config.share_embeddings_and_output_weights,
            position_embedding_type=config.position_embedding_type,
            rotary_percent=config.rotary_percent,
            rotary_base=config.rotary_base,
            rope_scaling=config.rope_scaling,
            rope_scaling_factor=config.rope_scaling_factor,
            seq_len_interpolation_factor=config.seq_len_interpolation_factor,
            scatter_embedding_sequence_parallel=config.scatter_embedding_sequence_parallel,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )


__all__ = ["LayerSpecGPTModelBuilder", "mtp_block_spec_from_layer_spec"]
