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

from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_inference_stack_spec as default_hybrid_inference_stack_spec,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec as default_hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel as MCoreHybridModel
from megatron.core.post_training.modelopt.hybrid.model_specs import get_hybrid_stack_modelopt_spec
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.models.hybrid import HybridModelBuilder, HybridModelConfig


def _hybrid_model_uses_cuda_graphs(transformer_config: TransformerConfig) -> bool:
    """Return whether current or deprecated config fields enable CUDA graphs."""
    return (
        transformer_config.cuda_graph_impl != "none"
        or transformer_config.enable_cuda_graph
        or transformer_config.external_cuda_graph
    )


def _hybrid_model_supports_ep_overlap(transformer_config: TransformerConfig) -> bool:
    """Return whether the installed MCore supports HybridModel EP overlap for this config."""
    return callable(getattr(MCoreHybridModel, "build_schedule_plan", None)) and not _hybrid_model_uses_cuda_graphs(
        transformer_config
    )


def _validate_hybrid_model_ep_overlap(transformer_config: TransformerConfig) -> None:
    """Reject HybridModel EP overlap when the installed MCore cannot support it."""
    if not transformer_config.overlap_moe_expert_parallel_comm:
        return
    if not callable(getattr(MCoreHybridModel, "build_schedule_plan", None)):
        raise ValueError(
            "HybridModel does not support overlap_moe_expert_parallel_comm with the current Megatron Core. "
            "Disable comm_overlap.overlap_moe_expert_parallel_comm and comm_overlap.delay_wgrad_compute."
        )
    if _hybrid_model_uses_cuda_graphs(transformer_config):
        raise ValueError(
            "HybridModel EP overlap does not support CUDA graphs with the current Megatron Core. "
            "Set model.cuda_graph_impl=none or disable MoE expert-parallel communication overlap."
        )


def transformer_engine_hybrid_stack_spec() -> ModuleSpec:
    """Return the default Hybrid stack spec with Transformer Engine layers.

    This is a named function (not a lambda) to allow proper serialization
    and reconstruction from checkpoints. Named functions can be imported
    via their module path, unlike lambdas.

    Returns:
        Default Hybrid stack specification from megatron.core.
    """
    return default_hybrid_stack_spec


def modelopt_hybrid_stack_spec(config: "HybridModelConfig | None" = None) -> ModuleSpec:
    """Hybrid stack specification for quantization with ModelOpt.

    Uses Norm instead of TENorm and ColumnParallelLinear/RowParallelLinear
    instead of TE layers to enable proper quantizer insertion by ModelOpt.

    Args:
        config: Optional Hybrid configuration object.

    Returns:
        Module specification for quantization-ready Hybrid stack.
    """
    return get_hybrid_stack_modelopt_spec(
        local_core_attention=False,
        remap_te_layernorm=True,
    )


def get_default_hybrid_stack_spec(config: "HybridModelConfig") -> ModuleSpec:
    """Determine the most appropriate Hybrid stack specification based on configuration.

    Args:
        config: Hybrid configuration object.

    Returns:
        Appropriate module specification based on config.
    """
    transformer = getattr(config, "transformer", config)
    if transformer.transformer_impl == "inference_optimized":
        return default_hybrid_inference_stack_spec
    if config.restore_modelopt_state:
        return modelopt_hybrid_stack_spec(config)
    return transformer_engine_hybrid_stack_spec()


__all__ = [
    "HybridModelBuilder",
    "HybridModelConfig",
    "get_default_hybrid_stack_spec",
    "modelopt_hybrid_stack_spec",
    "transformer_engine_hybrid_stack_spec",
]
