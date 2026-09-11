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

"""Megatron-FSDP wrapper compatibility helpers."""

try:
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
        FullyShardedDataParallelV1,
        FullyShardedDataParallelV2,
    )

    MEGATRON_FSDP_TYPES = (FullyShardedDataParallelV1, FullyShardedDataParallelV2)
    MCORE_HAS_MEGATRON_FSDP_V2 = True
except ImportError:
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel

    MEGATRON_FSDP_TYPES = (FullyShardedDataParallel,)
    MCORE_HAS_MEGATRON_FSDP_V2 = False
