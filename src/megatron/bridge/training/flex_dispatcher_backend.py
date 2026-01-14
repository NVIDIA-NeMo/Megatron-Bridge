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
import os

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

    device_properties = torch.cuda.get_device_properties(0)
    if moe_flex_dispatcher_backend == "deepep":
        if not (device_properties.major in [8, 9] or device_properties.name in ["NVIDIA B200", "NVIDIA B300"]):
            if get_rank_safe() == 0:
                logger.warning(
                    "DeepEP is only applicable to Ampere, Hopper, and Blackwell (only B200 and B300) GPUs. Skipping DeepEP configuration."
                )
            return
    elif moe_flex_dispatcher_backend == "hybridep":
        # Always set NVLINK_DOMAIN_SIZE to 72 and USE_MNNVL to 1 for HybridEP as requested,
        # but print original values for debug first.
        original_nvl_size = os.environ.get("NVLINK_DOMAIN_SIZE")
        original_use_mnnvl = os.environ.get("USE_MNNVL")
        
        if get_rank_safe() == 0:
            print(
                f"DEBUG: HybridEP configuration. "
                f"GPU: {device_properties.name} (major={device_properties.major}). "
                f"Original environment: NVLINK_DOMAIN_SIZE={original_nvl_size}, USE_MNNVL={original_use_mnnvl}"
            )
        
        os.environ["NVLINK_DOMAIN_SIZE"] = "72"
        os.environ["USE_MNNVL"] = "1"
        
        # We allow HybridEP on any Blackwell GPU (major=10) if requested, 
        # as some GB200 systems may report as B200.
        if device_properties.major != 10:
            if get_rank_safe() == 0:
                logger.warning(
                    f"HybridEP is intended for Blackwell GPUs (major=10). "
                    f"Detected GPU: {device_properties.name} (major={device_properties.major}). "
                    "Proceeding with HybridEP as requested."
                )
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
        device_properties = torch.cuda.get_device_properties(0)
        if model_config.moe_flex_dispatcher_backend == "deepep":
            if not (device_properties.major in (8, 9) or device_properties.name in ["NVIDIA B200", "NVIDIA B300"]):
                raise ValueError("DeepEP is supported for Ampere, Hopper, and Blackwell (only B200 and B300) GPUs")

        if model_config.moe_flex_dispatcher_backend == "hybridep":
            # Always ensure NVLINK_DOMAIN_SIZE and USE_MNNVL are set for HybridEP
            os.environ["NVLINK_DOMAIN_SIZE"] = "72"
            os.environ["USE_MNNVL"] = "1"
            
            if device_properties.major != 10:
                if get_rank_safe() == 0:
                    logger.warning(
                        f"HybridEP validation: Typically requires Blackwell GPUs (major=10). "
                        f"Detected GPU: {device_properties.name} (major={device_properties.major}). "
                        "Proceeding anyway."
                    )
