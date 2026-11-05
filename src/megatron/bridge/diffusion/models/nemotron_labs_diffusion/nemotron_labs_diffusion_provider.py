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

"""NemotronLabsDiffusion model provider: text-only GPTModel + NemotronLabsDiffusionAttention for sbd_block_diff."""

import copy
import functools
import inspect
from dataclasses import dataclass

from megatron.bridge.diffusion.models.common.nemotron_labs_diffusion_attention import (
    Ministral3RotaryEmbedding,
    NemotronLabsDiffusionAttention,
)
from megatron.bridge.models.gpt_provider import ModuleSpec
from megatron.bridge.models.mistral.mistral_provider import MistralModelProvider


@dataclass
class NemotronLabsDiffusionModelProvider(MistralModelProvider):
    """Text-only diffusion LM with NemotronLabsDiffusionAttention (sbd_block_diff) for dLLM training."""

    # Preserve the text architecture defaults without inheriting VLM configuration.
    num_attention_heads: int = 32
    num_query_groups: int = 8
    kv_channels: int = 128
    yarn_rotary_scaling_factor: float = 16.0
    yarn_original_max_position_embeddings: int = 16384
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_correction_range_round_to_int: bool = False
    yarn_mscale: float | None = 1.0
    yarn_mscale_all_dim: float | None = 1.0

    mask_token_id: int = 100
    dlm_paradigm: str = "sbd_block_diff"
    block_size: int = 64
    different_seed_per_dp: bool = True
    # Native runtime configuration, populated by provider_bridge from the checkpoint.
    rope_type: str = "default"
    yarn_attention_factor: float | None = None
    apply_llama4_style_query_key_layer_scaling: bool = False
    llama4_scaling_beta: float | None = None
    dlm_loss_weight: float = 0.3
    ar_loss_weight: float = 1.0
    position_embedding_type: str = "none"
    # This text-only path uses GPT embeddings, without the VLM wrapper scatter.
    scatter_embedding_sequence_parallel: bool = True

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            if "vp_stage" in inspect.signature(transformer_layer_spec).parameters:
                transformer_layer_spec = transformer_layer_spec(self, vp_stage=vp_stage)
            else:
                transformer_layer_spec = transformer_layer_spec(self)

        # Keep runtime RoPE modules out of caller-owned specs and checkpoint config.
        transformer_layer_spec = copy.deepcopy(transformer_layer_spec)
        if hasattr(transformer_layer_spec, "submodules"):
            rope_module = Ministral3RotaryEmbedding.from_megatron_config(self)
            transformer_layer_spec.submodules.self_attention.submodules.core_attention = functools.partial(
                NemotronLabsDiffusionAttention, rope_module=rope_module
            )

        original_spec = self.transformer_layer_spec
        self.transformer_layer_spec = transformer_layer_spec
        try:
            return super().provide(pre_process, post_process, vp_stage)
        finally:
            self.transformer_layer_spec = original_spec
