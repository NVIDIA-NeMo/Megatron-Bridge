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

import logging

from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.utils.common_utils import get_rank_safe


logger: logging.Logger = logging.getLogger(__name__)


def apply_moe_token_drop(
    model_provider: TransformerConfig,
    moe_expert_capacity_factor: float = 1.0,
    moe_pad_expert_input_to_capacity: bool = True,
) -> None:
    """Token drop improves performance by better balancing work across experts, but may affect convergence.

    MoE token drop is applicable to MoE models only.

    Args:
        model_provider (TransformerConfig): The transformer config to apply the token drop settings to
        moe_expert_capacity_factor (float): The capacity factor for all experts
        moe_pad_expert_input_to_capacity (bool): Pad the input for each expert to the expert capacity length

    Raises:
        AssertionError: If moe_router_load_balancing_type is not aux_loss, seq_aux_loss, or none
        AssertionError: If moe_token_dispatcher_type is not alltoall or alltoall_seq
        ValueError: If moe_expert_capacity_factor is not set and moe_pad_expert_input_to_capacity is True
    """
    if model_provider.num_moe_experts is None or model_provider.num_moe_experts == 0:
        if get_rank_safe() == 0:
            logger.warning(
                "MoE token drop is only applicable to MoE models. "
                "Model config does not use MoE (num_moe_experts is not set or is 0). "
                "Skipping MoE token drop configuration."
            )
        return

    if moe_expert_capacity_factor < 0:
        moe_expert_capacity_factor = None

    assert model_provider.moe_token_dispatcher_type in (
        "alltoall",
        "alltoall_seq",
    ), "moe_expert_capacity_factor only works with alltoall token dispatcher"

    assert model_provider.moe_router_load_balancing_type in (
        "seq_aux_loss",
        "aux_loss",
        "none",
    ), "moe_expert_capacity_factor only works with aux_loss or none load balancing"

    if moe_pad_expert_input_to_capacity:
        if moe_expert_capacity_factor is None:
            raise ValueError("moe_expert_capacity_factor must be set to use moe_pad_expert_input_to_capacity")

    model_provider.moe_expert_capacity_factor = moe_expert_capacity_factor
    model_provider.moe_pad_expert_input_to_capacity = moe_pad_expert_input_to_capacity

    return model_provider
