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

from megatron.bridge.models.distillation_provider import DistillationProvider
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.forward_step_func_types import ForwardStepCallable
from megatron.bridge.training.gpt_step import forward_step_modelopt
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.utils.decorators import experimental_fn


def _check_step_handles_context_parallelism(config: ConfigContainer, forward_step_func: ForwardStepCallable) -> None:
    """The GPT step CP-shards the batch, so models that shard it themselves would shard twice."""
    if getattr(config.model, "context_parallel_size", 1) == 1 or forward_step_func is not forward_step_modelopt:
        return
    if getattr(config.model, "position_embedding_type", None) == "mrope":
        raise ValueError(
            "mrope models shard the sequence inside model.forward, so the default GPT step double-shards "
            "it under context parallelism. Pass the model's own step, e.g. "
            "megatron.bridge.models.qwen_vl.qwen3_vl_step.forward_step_modelopt."
        )


@experimental_fn
def distill(
    config: ConfigContainer,
    forward_step_func: ForwardStepCallable = forward_step_modelopt,
) -> None:
    """Main function to run knowledge distillation (KD).

    Args:
        config: The main configuration container holding all necessary parameters.
        forward_step_func: Step function to distill with, defaulting to the GPT step. Models
            that shard the sequence themselves need their own -- pass
            ``qwen3_vl_step.forward_step_modelopt`` for Qwen3-VL under context parallelism.
            It must attach the KD loss, or training silently runs without distillation.

    Warnings:
        This is an experimental API and is subject to change in backwards
        incompatible ways without notice.
    """
    assert isinstance(config.model, DistillationProvider), "Distillation requires a DistillationProvider"
    _check_step_handles_context_parallelism(config, forward_step_func)

    return pretrain(config, forward_step_func)
