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

"""Optional Megatron Core runtime patches for performance workloads."""

import logging
import os
from importlib import import_module


logger = logging.getLogger(__name__)

SKIP_FP8_DEQUANT_ON_LOAD_ENV = "MBRIDGE_SKIP_FP8_DEQUANT_ON_LOAD"
_PATCH_APPLIED = False


def _is_env_enabled() -> bool:
    return os.environ.get(SKIP_FP8_DEQUANT_ON_LOAD_ENV, "").lower() in ("1", "true", "yes")


def patch_skip_force_all_tensors_to_non_fp8() -> bool:
    """No-op ``force_all_tensors_to_non_fp8`` when ``MBRIDGE_SKIP_FP8_DEQUANT_ON_LOAD`` is set.

    Megatron Core calls ``force_all_tensors_to_non_fp8()`` at the start of
    ``dist_checkpointing.load()``, materializing high-precision copies of all FP8
    weight shards before reading the checkpoint. On memory-constrained GPUs this
    can OOM during load.

    Enable with::

        export MBRIDGE_SKIP_FP8_DEQUANT_ON_LOAD=1

    Returns:
        True if the patch was applied (or was already applied), False if disabled.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return True
    if not _is_env_enabled():
        return False

    def _noop_force_all_tensors_to_non_fp8(sharded_state_dict) -> None:
        del sharded_state_dict

    import megatron.core.dist_checkpointing.utils as dist_checkpointing_utils

    dist_checkpointing_utils.force_all_tensors_to_non_fp8 = _noop_force_all_tensors_to_non_fp8
    serialization_module = import_module("megatron.core.dist_checkpointing.serialization")
    serialization_module.force_all_tensors_to_non_fp8 = _noop_force_all_tensors_to_non_fp8
    _PATCH_APPLIED = True
    logger.info(
        "Patched force_all_tensors_to_non_fp8 to no-op (%s is set).",
        SKIP_FP8_DEQUANT_ON_LOAD_ENV,
    )
    return True
