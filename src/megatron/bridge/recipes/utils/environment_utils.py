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

"""Composable environment defaults for library and performance recipes."""

from megatron.bridge.training.config import ConfigContainer


def set_common_recipe_environment_defaults(config: ConfigContainer) -> None:
    """Set common Transformer Engine and compilation environment defaults."""
    defaults = {
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "TORCHINDUCTOR_WORKER_START": "fork",
        "QUANTIZATION_TYPE_DEBUG": 1,
    }
    for name, value in defaults.items():
        config.env_vars.setdefault(name, value)


def set_hybridep_environment_defaults(
    config: ConfigContainer,
    *,
    ranks_per_nvlink_domain: int,
    use_mnnvl: bool,
) -> None:
    """Set HybridEP topology defaults on a recipe config.

    Args:
        config: Recipe config to update.
        ranks_per_nvlink_domain: Number of HybridEP ranks in each NVLink domain.
        use_mnnvl: Whether the workload uses a multi-node NVLink domain.
    """
    defaults = {
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": ranks_per_nvlink_domain,
        "NVLINK_DOMAIN_SIZE": 72 if use_mnnvl else 8,
        "USE_MNNVL": int(use_mnnvl),
    }
    for name, value in defaults.items():
        config.env_vars.setdefault(name, value)
