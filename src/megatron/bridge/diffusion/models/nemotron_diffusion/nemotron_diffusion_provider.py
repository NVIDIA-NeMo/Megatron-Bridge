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

"""NemotronDiffusion model provider: text-only GPTModel + NemotronDiffusionAttention for sbd_block_diff."""

import inspect
from dataclasses import dataclass

from megatron.bridge.diffusion.models.common.nemotron_diffusion_attention import NemotronDiffusionAttention
from megatron.bridge.models import Ministral3ModelProvider
from megatron.bridge.models.gpt_provider import ModuleSpec


@dataclass
class NemotronDiffusionModelProvider(Ministral3ModelProvider):
    """Text-only diffusion LM with NemotronDiffusionAttention (sbd_block_diff) for dLLM training."""

    mask_token_id: int = 100
    dlm_paradigm: str = "sbd_block_diff"
    block_size: int = 64
    different_seed_per_dp: bool = True
    apply_llama4_style_query_key_layer_scaling: bool = True
    dlm_loss_weight: float = 0.3
    ar_loss_weight: float = 1.0
    position_embedding_type: str = "none"

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            if "vp_stage" in inspect.signature(transformer_layer_spec).parameters:
                transformer_layer_spec = transformer_layer_spec(self, vp_stage=vp_stage)
            else:
                transformer_layer_spec = transformer_layer_spec(self)

        if hasattr(transformer_layer_spec, "submodules"):
            transformer_layer_spec.submodules.self_attention.submodules.core_attention = NemotronDiffusionAttention

        original_spec = self.transformer_layer_spec
        self.transformer_layer_spec = transformer_layer_spec
        result = super().provide_language_model(pre_process, post_process, vp_stage)
        self.transformer_layer_spec = original_spec
        return result
