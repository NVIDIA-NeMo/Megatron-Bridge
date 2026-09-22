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
    """Configure the requested dispatcher without inspecting the local GPU.

    Recipe construction may run on a different host from training, and child
    recipes may replace a parent's backend. Validate hardware support with
    ``validate_flex_dispatcher_backend`` after all recipe and user overrides.
    An unsupported backend is never replaced with another dispatcher.

    Args:
        model_config: Model configuration to update in place.
        moe_flex_dispatcher_backend: ``deepep``, ``hybridep``, or ``ncclep`` to
            select flex dispatch; ``None`` explicitly selects alltoall.

    Raises:
        ValueError: If the backend name is not recognized.
    """
    if moe_flex_dispatcher_backend is None:
        model_config.moe_token_dispatcher_type = "alltoall"
        model_config.moe_flex_dispatcher_backend = None
        return
    if moe_flex_dispatcher_backend not in ("deepep", "hybridep", "ncclep"):
        raise ValueError(
            f"Unknown flex dispatcher backend: {moe_flex_dispatcher_backend!r}. "
            "Expected deepep, hybridep, ncclep, or None to explicitly select alltoall."
        )

    num_moe_experts = getattr(model_config, "num_moe_experts", None)
    if num_moe_experts is None or num_moe_experts == 0:
        if get_rank_safe() == 0:
            logger.warning(
                "Flex dispatcher backends are only applicable to MoE models. "
                "Model config does not use MoE (num_moe_experts is not set or is 0). "
                "Skipping flex dispatcher configuration."
            )
        return

    model_config.moe_token_dispatcher_type = "flex"
    model_config.moe_flex_dispatcher_backend = moe_flex_dispatcher_backend
    model_config.moe_shared_expert_overlap = False


def validate_flex_dispatcher_backend(model_config: TransformerConfig) -> None:
    """Validate the final backend on the training GPU without changing the config.

    Raises:
        ValueError: If flex has no recognized backend or the GPU is unsupported.
    """
    if model_config.moe_token_dispatcher_type == "flex":
        if model_config.moe_flex_dispatcher_backend not in ("deepep", "hybridep", "ncclep"):
            raise ValueError(
                "moe_token_dispatcher_type='flex' requires moe_flex_dispatcher_backend "
                f"to be deepep, hybridep, or ncclep; got {model_config.moe_flex_dispatcher_backend!r}. "
                "To use alltoall, explicitly set moe_token_dispatcher_type='alltoall' "
                "and moe_flex_dispatcher_backend=None."
            )

        migration_message = (
            " Choose a supported flex backend, or explicitly set "
            "moe_token_dispatcher_type='alltoall' and moe_flex_dispatcher_backend=None. "
            "The requested backend will not be changed automatically."
        )

        device_properties = torch.cuda.get_device_properties(0)
        if model_config.moe_flex_dispatcher_backend == "deepep":
            if not (
                device_properties.major in (8, 9) or device_properties.name.startswith(("NVIDIA B200", "NVIDIA B300"))
            ):
                raise ValueError(
                    f"DeepEP is supported for Ampere, Hopper, and Blackwell (B200/B300) GPUs. "
                    f"Current GPU: {device_properties.name}." + migration_message
                )

        if model_config.moe_flex_dispatcher_backend == "hybridep":
            if not device_properties.major in [8, 9, 10]:
                raise ValueError(
                    "HybridEP is supported for GB200, GB300 with NVL72 and for Ampere, Hopper, B200 and B300 GPUs. "
                    f"Current GPU: {device_properties.name}." + migration_message
                )

        if model_config.moe_flex_dispatcher_backend == "ncclep":
            if device_properties.major not in [9, 10]:
                raise ValueError(
                    f"NCCL EP is supported for Hopper and Blackwell GPUs. Current GPU: {device_properties.name}."
                    + migration_message
                )
