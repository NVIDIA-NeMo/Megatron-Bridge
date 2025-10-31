# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Callable
import torch

def get_no_weight_decay_cond(
    no_weight_decay_cond_type: str, default_skip_embedding_weight_decay: bool
    ) -> Callable[[str, torch.Tensor], bool]:
    """Get the no weight decay condition function."""

    # Default case: no_wd_decay_cond_type is None
    no_weight_decay_cond_fn = None

    if no_weight_decay_cond_type == 'qwen3_next':
        # Qwen3-Next applies weight decay to qk layernorm as a special case
        def qwen3_next_no_weight_decay_cond(name, param):
            if "q_layernorm" in name or "k_layernorm" in name:
                no_wd = False
            else:
                no_wd = (
                    name.endswith(".bias")
                    or len(param.shape) == 1
                    or (default_skip_embedding_weight_decay and "embedding" in name)
                )
            return no_wd
        no_weight_decay_cond_fn = qwen3_next_no_weight_decay_cond
    elif no_weight_decay_cond_type is not None:
        raise ValueError(f"Invalid no_weight_decay_cond_type: {no_weight_decay_cond_type}")

    return no_weight_decay_cond_fn
