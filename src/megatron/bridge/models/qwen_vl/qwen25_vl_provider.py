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

import inspect
from dataclasses import dataclass, field
from typing import List

from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import ModuleSpec
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionConfig

from megatron.bridge.models.gpt_provider import GPTModelProvider, mtp_block_spec
from megatron.bridge.models.qwen_vl.modeling_qwen25_vl import Qwen25VLModel
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


@dataclass
class Qwen25VLModelProvider(GPTModelProvider):
    """
    Base model provider for Qwen 2.5 VL Models.
    """

    # VL models shouldn't scatter embeddings across sequence parallel regions because
    # the vision embeddings are going to be inserted into the language embeddings.
    scatter_embedding_sequence_parallel: bool = False
    position_embedding_type: str = "mrope"
    mrope_section: List[int] = field(default_factory=lambda: [16, 24, 24])

    # Vision configuration
    vision_config: Qwen2_5_VLVisionConfig = field(default_factory=Qwen2_5_VLVisionConfig)
    return_dict: bool = True

    # Token IDs
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    image_token_id: int = 151655
    video_token_id: int = 151656

    # Freeze options
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> Qwen25VLModel:
        pg_collection = self._pg_collection or ProcessGroupCollection.use_mpu_process_groups()
        vp_size = self.virtual_pipeline_model_parallel_size
        if pre_process is None:
            pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(pg_collection.pp)
        if post_process is None:
            post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(pg_collection.pp)

        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            if "vp_stage" in inspect.signature(transformer_layer_spec).parameters:
                transformer_layer_spec = transformer_layer_spec(self, vp_stage=vp_stage)
            else:
                transformer_layer_spec = transformer_layer_spec(self)

        if self.vocab_size is None:
            raise ValueError("vocab_size must be configured before building Qwen2.5-VL")
        language_vocab_size = self.vocab_size
        if self.should_pad_vocab:
            language_vocab_size = calculate_padded_vocab_size(
                self.vocab_size, self.make_vocab_size_divisible_by, self.tensor_model_parallel_size
            )

        model = Qwen25VLModel(
            language_transformer_config=self,
            language_transformer_layer_spec=transformer_layer_spec,
            vision_transformer_config=self.vision_config,
            model_config=self,
            language_vocab_size=language_vocab_size,
            parallel_output=self.parallel_output,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            mtp_block_spec=mtp_block_spec(self, vp_stage=vp_stage),
            vp_stage=vp_stage,
        )

        # Apply freeze options if any are enabled
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        return GPTModelProvider.provide(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
