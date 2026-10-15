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

"""Process environment defaults shared by every hardware recipe.

Keep this mapping limited to values that are identical across all hardware
recipes. Model- and topology-specific values belong directly in the owning
recipe builder so users can see the complete launch environment.
"""

COMMON_RECIPE_ENV_VARS: dict[str, str | int | float | bool] = {
    # Disable graph registration because these recipes use the expandable
    # allocator rather than NCCL user-buffer graph registration.
    "NCCL_GRAPH_REGISTER": 0,
    # Recipe baselines do not enable NCCL user buffers by default.
    "NCCL_NVLS_ENABLE": 0,
    # Let long-running training jobs grow allocator segments when needed.
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    # Keep NCCL stream handling consistent across all hardware recipes.
    "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
    "TORCH_NCCL_HIGH_PRIORITY": 1,
}

# Values required for bit-exact reproducibility. Identical to what
# apply_determinism_overrides exports; spread into a deterministic recipe's
# static env_vars dict, which is reassigned after that call.
DETERMINISM_ENV_VARS: dict[str, str | int | float | bool] = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NCCL_ALGO": "Ring",
    "NCCL_EP_HT_EM_AG_SCAN_MODE": "1",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "MAMBA_DETERMINISTIC": "1",
    "CAUSAL_CONV1D_DETERMINISTIC": "1",
}
