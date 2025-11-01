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

from megatron.bridge.models.gpt_provider import GPTDistillationProvider
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import forward_step_modelopt
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.utils.decorators import experimental_fn


@experimental_fn
def distill(
    config: ConfigContainer,
) -> None:
    """Main function to run knowledge distillation (KD).

    Args:
        config: The main configuration container holding all necessary parameters.

    Warnings:
        This is an experimental API and is subject to change in backwards
        incompatible ways without notice.
    """
    assert isinstance(config.model, GPTDistillationProvider), "Distillation requires a GPTDistillationProvider"

    return pretrain(config, forward_step_modelopt)
