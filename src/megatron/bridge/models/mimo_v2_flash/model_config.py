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

"""Pure model config and standalone builder for MiMo-V2-Flash."""

from dataclasses import dataclass, field
from typing import ClassVar

from megatron.core.models.gpt import GPTModel
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.models.gpt import GPTModelBuilder
from megatron.training.vocab_utils import calculate_padded_vocab_size

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.mimo_v2_flash.modeling_mimo_v2_flash import (
    MiMoV2FlashRotaryEmbedding,
    mimo_v2_flash_layer_spec,
    mimo_v2_flash_mtp_block_spec,
)


@dataclass(kw_only=True)
class MiMoV2FlashModelConfig(BridgeGPTModelConfig):
    """Serializable MiMo-V2-Flash construction inputs."""

    builder: ClassVar[str] = "megatron.bridge.models.mimo_v2_flash.model_config.MiMoV2FlashModelBuilder"
    hybrid_attention_pattern: list[int] = field(default_factory=list)
    sliding_window_size: int = 128
    rotary_base_local: float = 10_000.0
    full_attn_num_query_groups: int = 4
    swa_num_query_groups: int = 8
    v_head_dim: int = 128
    attention_value_scale: float | None = None


class MiMoV2FlashModelBuilder(GPTModelBuilder):
    """Build MiMo-V2-Flash without mutating its MCore transformer config."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> GPTModel:
        """Build one MiMo-V2-Flash pipeline stage."""
        config = self._model_config
        assert isinstance(config, MiMoV2FlashModelConfig)
        min_kv_groups = min(config.full_attn_num_query_groups, config.swa_num_query_groups)
        if config.transformer.tensor_model_parallel_size > min_kv_groups:
            raise ValueError("MiMo-V2-Flash requires TP size <= the minimum query-group count.")
        if config.transformer.context_parallel_size > 1:
            raise ValueError("MiMo-V2-Flash does not support context parallelism.")

        layer_spec = mimo_v2_flash_layer_spec(config)
        mtp_spec = mimo_v2_flash_mtp_block_spec(config, vp_stage=vp_stage)
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
        model = GPTModel(
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
        model.rotary_pos_emb = MiMoV2FlashRotaryEmbedding(
            kv_channels=config.transformer.kv_channels,
            rotary_percent=config.rotary_percent,
            rotary_interleaved=config.transformer.rotary_interleaved,
            seq_len_interpolation_factor=config.seq_len_interpolation_factor,
            rotary_base=config.rotary_base,
            rope_scaling=False,
            use_cpu_initialization=config.transformer.use_cpu_initialization,
            rotary_base_local=config.rotary_base_local,
        )
        return model


__all__ = ["MiMoV2FlashModelBuilder", "MiMoV2FlashModelConfig"]
