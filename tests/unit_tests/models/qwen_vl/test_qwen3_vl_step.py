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

import pytest
import torch

from megatron.bridge.models.qwen_vl.qwen3_vl_step import get_batch_from_iterator


pytestmark = pytest.mark.unit


def test_get_batch_from_iterator_rejects_collate_time_packing_metadata():
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "visual_inputs": None,
        "cu_seqlens": torch.tensor([[0, 3]], dtype=torch.int32),
    }

    with pytest.raises(ValueError, match="does not support collate-time in-batch packing"):
        get_batch_from_iterator(
            iter([batch]),
            is_first_pp_stage=True,
            is_last_pp_stage=True,
        )
