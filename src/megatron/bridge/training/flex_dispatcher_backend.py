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


def apply_flex_dispatcher_backend(
    model_config: TransformerConfig,
    moe_flex_dispatcher_backend: str | None = None,
) -> None:
    """Apply DeepEP or HybridEP optimizations to the model config.

    DeepEP is applicable only to MoE models on Ampere and Hopper GPUs.
    HybridEP is applicable only to MoE models on GB200 GPUs with NVL72.
    """
    num_moe_experts = getattr(model_config, "num_moe_experts", None)
    if num_moe_experts is None or num_moe_experts == 0:
        if get_rank_safe() == 0:
            logger.warning(
                "DeepEP and HybridEP are only applicable to MoE models. "
                "Model config does not use MoE (num_moe_experts is not set or is 0). "
                "Skipping DeepEP configuration."
            )
        return

    if moe_flex_dispatcher_backend == "deepep":
        if torch.cuda.get_device_properties(0).major not in [8, 9]:
            if get_rank_safe() == 0:
                logger.warning(
                    "DeepEP is only applicable to Ampere (SM80) and Hopper (SM90) GPUs. Skipping DeepEP configuration."
                )
            return
    elif moe_flex_dispatcher_backend == "hybridep":
        if not (
            torch.cuda.get_device_properties(0).major == 10
            and torch.cuda.get_device_properties(0).name in ["NVIDIA GB200", "NVIDIA GB300"]
        ):
            if get_rank_safe() == 0:
                logger.warning(
                    "HybridEP is only applicable to GB200 and GB300 GPUs with NVL72. Skipping HybridEP configuration."
                )
            return
    else:
        if get_rank_safe() == 0:
            logger.warning("Not a valid flex dispatcher backend. Skipping flex dispatcher backend configuration.")
        return

    model_config.moe_token_dispatcher_type = "flex"
    model_config.moe_flex_dispatcher_backend = moe_flex_dispatcher_backend
    model_config.moe_shared_expert_overlap = False


def validate_flex_dispatcher_backend(model_config: TransformerConfig) -> None:
    """Validate DeepEP or HybridEP is supported for the current GPU architecture."""
    if model_config.moe_token_dispatcher_type == "flex":
        if model_config.moe_flex_dispatcher_backend == "deepep" and torch.cuda.get_device_properties(0).major not in (
            8,
            9,
        ):
            raise ValueError("DeepEP is supported for Ampere (SM80) and Hopper (SM90) GPUs")

        if model_config.moe_flex_dispatcher_backend == "hybridep":
            device_properties = torch.cuda.get_device_properties(0)
            if device_properties.major != 10 or device_properties.name not in ["NVIDIA GB200", "NVIDIA GB300"]:
                raise ValueError("HybridEP is supported for GB200 or GB300 GPUs with NVL72")
