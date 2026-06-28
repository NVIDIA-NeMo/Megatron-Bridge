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

"""Pure DeepSeek-V4 model config and builder."""

import copy
from dataclasses import dataclass, field, replace
from typing import ClassVar

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.models.gpt import GPTModelBuilder

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


def deepseek_v4_layer_spec(config):
    """Dispatch to the installed MCore DeepSeek-v4 implementation.

    Bridge keeps the family fields outside the exact MLA config and attaches
    them only to this ephemeral construction copy. MCore versions without the
    ``dsv4_hybrid`` implementation raise their native unsupported-variant error
    here; versions carrying the existing compatibility implementation build it.
    """
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_block_with_experimental_attention_variant_spec,
    )

    return get_transformer_block_with_experimental_attention_variant_spec(config)


@dataclass(kw_only=True)
class DeepSeekV4ModelConfig(BridgeGPTModelConfig):
    """Serializable DSv4 construction fields not owned by MCore MLA config."""

    builder: ClassVar[str] = "megatron.bridge.models.deepseek.deepseek_v4_model_config.DeepSeekV4ModelBuilder"
    experimental_attention_variant: str = "dsv4_hybrid"
    o_groups: int = 8
    o_lora_rank: int = 1024
    csa_compress_ratios: list[int] = field(default_factory=list)
    csa_window_size: int = 128
    csa_compress_rotary_base: float = 160_000.0
    dsa_indexer_n_heads: int = 64
    dsa_indexer_head_dim: int = 128
    dsa_indexer_topk: int = 512
    apply_dsa_kernel_fusion: bool = False
    enable_hyper_connections: bool = True
    use_fused_mhc: bool = False
    num_residual_streams: int = 4
    mhc_sinkhorn_iterations: int = 20
    moe_n_hash_layers: int = 0
    actual_vocab_size: int = 0
    activation_func_clamp_value: float = 10.0


class DeepSeekV4ModelBuilder(GPTModelBuilder):
    """Build DSv4 from an exact MCore MLA config plus explicit family state."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> GPTModel:
        """Build one DSv4 pipeline stage."""
        config = self._model_config
        assert isinstance(config, DeepSeekV4ModelConfig)
        layer_config = copy.deepcopy(config.transformer)
        for name in (
            "experimental_attention_variant",
            "o_groups",
            "o_lora_rank",
            "csa_compress_ratios",
            "csa_window_size",
            "csa_compress_rotary_base",
            "dsa_indexer_n_heads",
            "dsa_indexer_head_dim",
            "dsa_indexer_topk",
            "apply_dsa_kernel_fusion",
            "enable_hyper_connections",
            "use_fused_mhc",
            "num_residual_streams",
            "mhc_sinkhorn_iterations",
            "moe_n_hash_layers",
            "actual_vocab_size",
            "activation_func_clamp_value",
        ):
            setattr(layer_config, name, getattr(config, name))
        runtime_config = replace(config, transformer_layer_spec=deepseek_v4_layer_spec(layer_config))
        return GPTModelBuilder(runtime_config).build_model(pg_collection, pre_process, post_process, vp_stage)


__all__ = ["DeepSeekV4ModelBuilder", "DeepSeekV4ModelConfig"]
