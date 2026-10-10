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

"""Construction of Qwen4-Exp models from declarative configuration."""

import contextlib

import torch
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.bridge.models.hybrid.hybrid_builder import HybridModelBuilder
from megatron.bridge.models.logit_dtype import logit_dtype_kwarg
from megatron.bridge.models.qwen.modeling_qwen4_exp.layer_specs import get_qwen4_exp_hybrid_stack_spec
from megatron.bridge.models.qwen.modeling_qwen4_exp.model import Qwen4ExpHybridModel
from megatron.bridge.utils import fusions
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


class Qwen4ExpModelBuilder(HybridModelBuilder):
    """Build the Bridge-local blocks through the Core HybridModel API."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Qwen4ExpHybridModel:
        """Build a complete PP=CP=1 language decoder with explicit process groups."""
        config = self._model_config
        config.finalize()
        if vp_stage is not None or pre_process is False or post_process is False:
            raise NotImplementedError("Qwen4-Exp requires a complete decoder on each PP=1 rank.")
        if pg_collection is None:
            raise ValueError("Qwen4-Exp requires an explicit process-group collection.")
        if config.vocab_size is None:
            raise ValueError("vocab_size must be configured before calling build_model().")
        if config.hybrid_stack_spec is not None:
            raise ValueError("Qwen4-Exp owns its hybrid stack specification.")
        if not fusions.validate_rope_fusion_compatibility(config):
            config.apply_rope_fusion = False
        vocab_size = config.vocab_size
        if config.should_pad_vocab:
            vocab_size = calculate_padded_vocab_size(
                vocab_size, config.make_vocab_size_divisible_by, config.tensor_model_parallel_size
            )
        init_context = torch.device("meta") if config.init_model_with_meta_device else contextlib.nullcontext()
        with init_context:
            return Qwen4ExpHybridModel(
                config=config.transformer,
                hybrid_stack_spec=get_qwen4_exp_hybrid_stack_spec(config.transformer),
                hybrid_layer_pattern=config.hybrid_layer_pattern,
                vocab_size=vocab_size,
                max_sequence_length=config.seq_length,
                fp16_lm_cross_entropy=config.fp16_lm_cross_entropy,
                **logit_dtype_kwarg(HybridModel, config.logit_dtype),
                parallel_output=config.parallel_output,
                share_embeddings_and_output_weights=config.share_embeddings_and_output_weights,
                position_embedding_type=config.position_embedding_type,
                rotary_percent=config.rotary_percent,
                rotary_base=config.rotary_base,
                seq_len_interpolation_factor=config.seq_len_interpolation_factor,
                pre_process=True,
                post_process=True,
                scatter_embedding_sequence_parallel=config.scatter_embedding_sequence_parallel,
                pg_collection=pg_collection,
                mrope_section=config.mrope_section,
            )
