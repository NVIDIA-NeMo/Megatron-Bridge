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

import torch
from megatron.core.transformer import TransformerConfig

from megatron.bridge.utils.common_utils import get_rank_safe


logger: logging.Logger = logging.getLogger(__name__)


def apply_deepep(model_config: TransformerConfig) -> None:
    """Apply DeepEP optimizations to the model config.

    DeepEP is applicable only to MoE models on Ampere and Hopper GPUs.
    """
    if model_config.num_moe_experts is None or model_config.num_moe_experts == 0:
        if get_rank_safe() == 0:
            logger.warning(
                "DeepEP is only applicable to MoE models. "
                "Model config does not use MoE (num_moe_experts is not set or is 0). "
                "Skipping DeepEP configuration."
            )
        return

    if torch.cuda.get_device_properties(0).major not in [8, 9]:
        if get_rank_safe() == 0:
            logger.warning(
                "DeepEP is only applicable to Ampere (SM80) and Hopper (SM90) GPUs. Skipping DeepEP configuration."
            )
        return

    model_config.moe_token_dispatcher_type = "flex"
    model_config.moe_enable_deepep = True
    model_config.moe_shared_expert_overlap = False


def validate_deepep(model_config: TransformerConfig) -> None:
    """Validate DeepEP is supported for the current GPU architecture."""
    if model_config.moe_enable_deepep and torch.cuda.get_device_properties(0).major not in (8, 9):
        raise ValueError("DeepEP is supported for Ampere (SM80) and Hopper (SM90) GPUs")
