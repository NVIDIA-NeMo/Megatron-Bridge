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

"""Config-level overrides for deterministic training."""

from megatron.bridge.training.config import ConfigContainer


def apply_determinism_overrides(cfg: ConfigContainer) -> None:
    """Apply determinism config overrides to an existing ConfigContainer in-place.

    Sets the model-level flags required for bit-exact reproducibility and
    disables TP comm overlap (which uses non-deterministic NCCL collectives).
    Attention backend selection is a separate concern and is not touched here.

    The matching validator that enforces these flags at training time is
    :meth:`megatron.bridge.training.config.ConfigContainer._validate_and_apply_deterministic_mode`.

    This function is idempotent and is safe to call on configs with
    ``comm_overlap = None``.

    Args:
        cfg: Recipe config to modify.
    """
    cfg.model.deterministic_mode = True
    cfg.model.cross_entropy_loss_fusion = False
    cfg.env_vars.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NCCL_ALGO": "Ring",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
        }
    )

    if cfg.comm_overlap is not None:
        cfg.comm_overlap.tp_comm_overlap = False
