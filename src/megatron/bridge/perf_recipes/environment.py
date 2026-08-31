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

"""Process environment defaults shared by every flat performance recipe.

Keep this mapping deliberately small. Values that depend on a model, GPU,
parallel layout, precision, or CUDA graph mode belong next to the corresponding
recipe builder so users can see the exact benchmark environment.
"""

COMMON_PERF_ENV_VARS: dict[str, str | int | float | bool] = {
    # This is the only process setting with the same value in every flat recipe.
    # Run NCCL work on its high-priority stream for every measured workload.
    "TORCH_NCCL_HIGH_PRIORITY": 1,
}

# HybridEP tuning, read by DeepEP's HybridEPBuffer. Megatron Core only builds that buffer when
# moe_flex_dispatcher_backend is "hybridep", so these names are inert on any other backend. NCCL EP
# recipes still drop them so the launched environment, which is dumped with the run, does not
# advertise HybridEP tuning that nothing reads.
HYBRID_EP_ENV_NAMES: frozenset[str] = frozenset(
    {
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN",
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API",
        "NVLINK_DOMAIN_SIZE",
        "USE_MNNVL",
    }
)
