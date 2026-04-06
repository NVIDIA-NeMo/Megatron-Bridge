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

"""Wan diffusion model recipes exposed under megatron.bridge.recipes.wan.

The CI argument builder maps MODEL_RECIPE_NAME (e.g. "wan_1_3b") to a function
name "{model_recipe_name}_pretrain_config" (e.g. "wan_1_3b_pretrain_config").
These wrappers normalise the capitalisation used in the underlying diffusion
recipe module so that the CI lookup succeeds.
"""

from megatron.bridge.diffusion.recipes.wan.wan import (
    wan_1_3B_pretrain_config as _wan_1_3B_pretrain_config,
    wan_14B_pretrain_config as _wan_14B_pretrain_config,
)
from megatron.bridge.training.config import ConfigContainer


def wan_1_3b_pretrain_config() -> ConfigContainer:
    """Return the pre-training config for the Wan 1.3B model.

    CI MODEL_RECIPE_NAME: wan_1_3b
    Delegates to megatron.bridge.diffusion.recipes.wan.wan_1_3B_pretrain_config.
    """
    return _wan_1_3B_pretrain_config()


def wan_14b_pretrain_config() -> ConfigContainer:
    """Return the pre-training config for the Wan 14B model.

    CI MODEL_RECIPE_NAME: wan_14b
    Delegates to megatron.bridge.diffusion.recipes.wan.wan_14B_pretrain_config.
    """
    return _wan_14B_pretrain_config()
