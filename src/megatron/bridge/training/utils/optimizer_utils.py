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

"""Resolve optimizer distribution independently of model and training setup."""

from megatron.core.optimizer import OptimizerConfig


def uses_layer_wise_optimizer(config: OptimizerConfig) -> bool:
    """Return whether optimizer state is distributed by whole parameter tensors."""
    return config.use_layer_wise_distributed_optimizer or config.optimizer.startswith("dist_")


def uses_distributed_optimizer(config: OptimizerConfig) -> bool:
    """Return whether either distributed optimizer implementation is requested."""
    return config.use_distributed_optimizer or uses_layer_wise_optimizer(config)


def normalize_optimizer_config(config: OptimizerConfig, *, ddp_use_distributed_optimizer: bool = False) -> None:
    """Resolve legacy aliases and distributed requests before model construction.

    Emerging optimizers use layer-wise distribution, while Adam and SGD use
    the ordinary distributed optimizer. Match Core's command-line normalization
    for programmatically constructed Bridge configurations as well.
    """
    if config.optimizer.startswith("dist_"):
        config.optimizer = config.optimizer.removeprefix("dist_")
        config.use_layer_wise_distributed_optimizer = True
    if config.optimizer in ("adam", "sgd"):
        if config.use_layer_wise_distributed_optimizer:
            raise ValueError("Layer-wise distribution requires an emerging optimizer such as muon, not Adam or SGD.")
    elif config.use_distributed_optimizer or ddp_use_distributed_optimizer:
        config.use_layer_wise_distributed_optimizer = True
    if config.use_layer_wise_distributed_optimizer:
        config.use_distributed_optimizer = False
        if config.overlap_param_gather_with_optimizer_step:
            raise ValueError("Layer-wise optimizers do not support overlap_param_gather_with_optimizer_step.")
