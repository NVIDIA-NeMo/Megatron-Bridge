# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve MCore FSDP wrapper names across the v1/v2 API split."""

from megatron.core.distributed.fsdp import mcore_fsdp_adapter

from megatron.bridge.utils.import_utils import UnavailableMeta


if hasattr(mcore_fsdp_adapter, "FullyShardedDataParallelV1"):
    FullyShardedDataParallelV1 = mcore_fsdp_adapter.FullyShardedDataParallelV1
else:
    # Before the v2 adapter was introduced, the v1 wrapper had this name.
    FullyShardedDataParallelV1 = mcore_fsdp_adapter.FullyShardedDataParallel

if hasattr(mcore_fsdp_adapter, "FullyShardedDataParallelV2"):
    FullyShardedDataParallelV2 = mcore_fsdp_adapter.FullyShardedDataParallelV2
else:
    # Safe in isinstance checks; constructing v2 on older MCore fails explicitly.
    FullyShardedDataParallelV2 = UnavailableMeta(
        "FullyShardedDataParallelV2",
        (),
        {"_msg": "Megatron-FSDP v2 requires an MCore revision that provides FullyShardedDataParallelV2."},
    )
