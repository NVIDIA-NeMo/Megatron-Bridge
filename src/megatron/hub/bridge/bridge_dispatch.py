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

"""
Bridge dispatch for managing dispatch registration of model bridges.

This module handles the registration of MegatronModelBridge implementations
with the dispatch system, separating the registration logic from the core
bridge functionality.
"""

from typing import TYPE_CHECKING, Callable, Iterable, List, Literal, NamedTuple, Type, TypeVar, Union

from megatron.hub.common.decorators import dispatch
from megatron.hub.bridge.model_bridge import WeightDistributionMode, HFWeightTuple

if TYPE_CHECKING:
    from megatron.core.transformer.module import MegatronModule
    from transformers.modeling_utils import PreTrainedModel
    import torch


# Type variables for bridge registration
HFPreTrained = TypeVar("HFPreTrained")
MegatronModel = TypeVar("MegatronModel", bound="MegatronModule")
BridgeClass = TypeVar("BridgeClass", bound="MegatronModelBridge")


# Core dispatch functions
@dispatch
def get_model_bridge(hf_architecture) -> "MegatronModelBridge":
    """Get the appropriate model bridge for a given HuggingFace architecture."""
    ...


@dispatch
def stream_weights_megatron_to_hf(
    megatron_models: List[MegatronModel],
    hf_pretrained: HFPreTrained,
    cpu: bool = True,
    order: Literal["megatron", "hf", "safetensors"] = "safetensors",
    show_progress: bool = True,
    mode: Union[str, WeightDistributionMode] = WeightDistributionMode.CONSOLIDATE,
) -> Iterable[HFWeightTuple]:
    """Bridge Megatron model state to HuggingFace format."""
    ...


def register_bridge_implementation(
    *,
    source: Type["PreTrainedModel"],
    target: Type["MegatronModule"],
    bridge_class: Type["MegatronModelBridge"],
) -> None:
    """Register a bridge implementation with the dispatch system.
    
    Args:
        source: HuggingFace PreTrainedModel class (e.g., LlamaForCausalLM)
        target: Megatron model class (e.g., GPTModel)
        bridge_class: MegatronModelBridge implementation class
    """
    bridge_class_name = bridge_class.__name__

    @get_model_bridge.impl(source)
    def _get_model_bridge_impl(_) -> "MegatronModelBridge":
        bridge = bridge_class()
        return bridge

    @stream_weights_megatron_to_hf.impl((source, target))
    def _from_megatron_registered_impl(
        megatron_models: List["MegatronModule"],
        hf_pretrained: HFPreTrained,
        cpu: bool = True,
        order: Literal["megatron", "hf", "safetensors"] = "safetensors",
        show_progress: bool = True,
        mode: Union[str, WeightDistributionMode] = WeightDistributionMode.CONSOLIDATE,
    ) -> Iterable[HFWeightTuple]:
        bridge = bridge_class()
        return bridge.stream_weights_megatron_to_hf(
            megatron_models, hf_pretrained, cpu=cpu, order=order, show_progress=show_progress, mode=mode
        )

    # Set meaningful names for debugging
    _get_model_bridge_impl.__name__ = f"_bridge_with_{bridge_class_name}"
    _from_megatron_registered_impl.__name__ = f"_from_megatron_with_{bridge_class_name}"


def create_bridge_decorator(
    *, source: Type["PreTrainedModel"], target: Type["MegatronModule"]
) -> Callable[[Type["MegatronModelBridge"]], Type["MegatronModelBridge"]]:
    """Create a decorator for registering bridge implementations.
    
    Args:
        source: HuggingFace PreTrainedModel class
        target: Megatron model class
        
    Returns:
        Decorator function that registers the bridge implementation
    """
    def decorator(bridge_class: Type["MegatronModelBridge"]) -> Type["MegatronModelBridge"]:
        register_bridge_implementation(source=source, target=target, bridge_class=bridge_class)
        return bridge_class
    
    return decorator 