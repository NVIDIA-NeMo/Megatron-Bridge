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
# ruff: noqa: F401
"""Compatibility aliases for legacy recipe names."""

from __future__ import annotations

from typing import TYPE_CHECKING

from megatron.bridge.recipes.llama.h100.llama3 import (
    llama3_8b_peft_1gpu_h100_bf16_config as llama3_8b_peft_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama3_8b_pretrain_2gpu_h100_bf16_config as llama3_8b_pretrain_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama3_8b_pretrain_2gpu_h100_fp8cs_config as _llama3_8b_pretrain_2gpu_h100_fp8cs_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama3_8b_pretrain_2gpu_h100_fp8mx_config as _llama3_8b_pretrain_2gpu_h100_fp8mx_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama3_8b_pretrain_2gpu_h100_nvfp4_config as _llama3_8b_pretrain_2gpu_h100_nvfp4_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama3_8b_pretrain_16gpu_h100_bf16_16k_config as llama3_8b_16k_pretrain_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama3_8b_pretrain_32gpu_h100_bf16_64k_config as llama3_8b_64k_pretrain_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama3_8b_pretrain_64gpu_h100_bf16_128k_config as llama3_8b_128k_pretrain_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama3_8b_sft_2gpu_h100_bf16_config as llama3_8b_sft_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama3_70b_peft_8gpu_h100_bf16_config as llama3_70b_peft_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama3_70b_pretrain_32gpu_h100_bf16_16k_config as llama3_70b_16k_pretrain_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama3_70b_pretrain_32gpu_h100_bf16_config as llama3_70b_pretrain_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama3_70b_pretrain_32gpu_h100_bf16_deterministic_config as llama3_70b_pretrain_deterministic_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama3_70b_pretrain_256gpu_h100_bf16_64k_config as llama3_70b_64k_pretrain_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama3_70b_sft_32gpu_h100_bf16_config as llama3_70b_sft_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama31_8b_peft_1gpu_h100_bf16_config as llama31_8b_peft_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama31_8b_pretrain_2gpu_h100_bf16_config as llama31_8b_pretrain_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama31_8b_sft_2gpu_h100_bf16_config as llama31_8b_sft_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama31_70b_peft_8gpu_h100_bf16_config as llama31_70b_peft_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama31_70b_pretrain_32gpu_h100_bf16_config as llama31_70b_pretrain_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama31_70b_sft_32gpu_h100_bf16_config as llama31_70b_sft_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama31_405b_peft_32gpu_h100_bf16_config as llama31_405b_peft_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama31_405b_pretrain_256gpu_h100_bf16_config as llama31_405b_pretrain_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama31_405b_pretrain_256gpu_h100_bf16_deterministic_config as llama31_405b_pretrain_deterministic_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama31_405b_sft_128gpu_h100_bf16_config as llama31_405b_sft_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama32_1b_peft_1gpu_h100_bf16_config as llama32_1b_peft_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama32_1b_pretrain_1gpu_h100_bf16_config as llama32_1b_pretrain_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama32_1b_sft_1gpu_h100_bf16_config as llama32_1b_sft_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama32_3b_peft_1gpu_h100_bf16_config as llama32_3b_peft_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama32_3b_pretrain_1gpu_h100_bf16_config as llama32_3b_pretrain_config,
)
from megatron.bridge.recipes.llama.h100.llama3 import (
    llama32_3b_sft_1gpu_h100_bf16_config as llama32_3b_sft_config,
)


if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer


def llama3_8b_low_precision_pretrain_config(
    mixed_precision_recipe: str = "bf16_with_fp8_current_scaling_mixed",
) -> "ConfigContainer":
    """Return a low-precision Llama 3 8B pre-training config.

    Args:
        mixed_precision_recipe: Low-precision recipe to use.

    Returns:
        Llama 3 8B pre-training configuration.

    Raises:
        ValueError: If ``mixed_precision_recipe`` is not supported.
    """
    recipes = {
        "bf16_with_fp8_current_scaling_mixed": _llama3_8b_pretrain_2gpu_h100_fp8cs_config,
        "bf16_with_mxfp8_mixed": _llama3_8b_pretrain_2gpu_h100_fp8mx_config,
        "bf16_with_nvfp4_mixed": _llama3_8b_pretrain_2gpu_h100_nvfp4_config,
    }
    try:
        recipe = recipes[mixed_precision_recipe]
    except KeyError as error:
        raise ValueError(f"Unsupported low-precision recipe: {mixed_precision_recipe}") from error
    return recipe()


__all__ = [
    "llama31_405b_peft_config",
    "llama31_405b_pretrain_config",
    "llama31_405b_pretrain_deterministic_config",
    "llama31_405b_sft_config",
    "llama31_70b_peft_config",
    "llama31_70b_pretrain_config",
    "llama31_70b_sft_config",
    "llama31_8b_peft_config",
    "llama31_8b_pretrain_config",
    "llama31_8b_sft_config",
    "llama32_1b_peft_config",
    "llama32_1b_pretrain_config",
    "llama32_1b_sft_config",
    "llama32_3b_peft_config",
    "llama32_3b_pretrain_config",
    "llama32_3b_sft_config",
    "llama3_70b_16k_pretrain_config",
    "llama3_70b_64k_pretrain_config",
    "llama3_70b_peft_config",
    "llama3_70b_pretrain_config",
    "llama3_70b_pretrain_deterministic_config",
    "llama3_70b_sft_config",
    "llama3_8b_128k_pretrain_config",
    "llama3_8b_16k_pretrain_config",
    "llama3_8b_64k_pretrain_config",
    "llama3_8b_peft_config",
    "llama3_8b_pretrain_config",
    "llama3_8b_sft_config",
    "llama3_8b_low_precision_pretrain_config",
]
