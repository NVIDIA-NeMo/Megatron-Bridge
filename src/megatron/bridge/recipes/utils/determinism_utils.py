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

from megatron.training.determinism import DETERMINISM_ENV_VAR_DEFAULTS

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
    # Only the MoE aux-loss fusion is non-deterministic; fused TopK routing stays on.
    cfg.model.moe_router_aux_loss_fusion = False
    # Exported at launch so the values are in place before TE, cuBLAS and NCCL first read them.
    cfg.env_vars.update(DETERMINISM_ENV_VAR_DEFAULTS)
    # Pin the Mamba/causal-conv1d kernels explicitly instead of letting them follow torch's flag.
    cfg.env_vars.update({"MAMBA_DETERMINISTIC": "1", "CAUSAL_CONV1D_DETERMINISTIC": "1"})

    if cfg.comm_overlap is not None:
        cfg.comm_overlap.tp_comm_overlap = False
