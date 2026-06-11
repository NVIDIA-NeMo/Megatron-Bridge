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
    """Flip a recipe into deterministic mode in-place.

    Sets ``cfg.model.deterministic_mode = True``. The matching validator
    :meth:`megatron.bridge.training.config.ConfigContainer._validate_and_apply_deterministic_mode`
    delegates to ``megatron.training.determinism.apply_determinism_to_args``
    at training time, which is the single source of truth for what
    deterministic mode enforces (cross-entropy fusion off, NCCL_ALGO
    membership, tp_comm_overlap disabled, env-var setdefault,
    ``torch.use_deterministic_algorithms(True)``).

    Idempotent. Safe to call on configs with ``comm_overlap = None``.

    Note:
        Bit-exact reproducibility also requires the launcher to export the
        determinism env vars (``NCCL_ALGO=Ring``, ``NVTE_ALLOW_NONDETERMINISTIC_ALGO=0``,
        ``CUBLAS_WORKSPACE_CONFIG=:4096:8``). The performance launcher does
        this via ``PerfEnvPlugin(deterministic=True)``; callers outside that
        launcher must set them themselves (or rely on the in-process
        setdefault that the validator triggers).

    Args:
        cfg: Recipe config to modify.
    """
    cfg.model.deterministic_mode = True
