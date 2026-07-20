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
    # Use cuDNN LayerNorm for the common Transformer Engine baseline.
    "NVTE_NORM_BWD_USE_CUDNN": 1,
    "NVTE_NORM_FWD_USE_CUDNN": 1,
    # Let long-running training jobs grow allocator segments when needed.
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    # Keep NCCL stream handling consistent across all hardware recipes.
    "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
    "TORCH_NCCL_HIGH_PRIORITY": 1,
}
