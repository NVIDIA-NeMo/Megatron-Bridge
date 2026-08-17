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

from types import SimpleNamespace

import pytest
import torch
from megatron.core.ssm.gated_delta_net import GatedDeltaNet

from megatron.bridge.training.eval_context_parallel_rebinding import eval_cp_context


class _FakeProcessGroup:
    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


def _make_gdn(cp_size: int) -> GatedDeltaNet:
    gdn = GatedDeltaNet.__new__(GatedDeltaNet)
    torch.nn.Module.__init__(gdn)
    gdn.config = SimpleNamespace(context_parallel_size=cp_size, deterministic_mode=False)
    gdn.pg_collection = SimpleNamespace(cp=_FakeProcessGroup(cp_size))
    gdn.cp_size = cp_size
    gdn.tp_size = 1
    gdn.qk_dim_local_tp = 256
    gdn.v_dim_local_tp = 256
    gdn.num_value_heads = 8
    gdn.num_v_heads_local_tp = 8
    gdn._setup_variant_attrs()
    return gdn


@pytest.mark.unit
def test_eval_cp_context_refreshes_gdn_runtime_shape_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torch.distributed,
        "get_process_group_ranks",
        lambda group: list(range(group.size())),
    )
    gdn = _make_gdn(cp_size=1)
    train_pgs = gdn.pg_collection
    eval_pgs = SimpleNamespace(cp=_FakeProcessGroup(2))

    with eval_cp_context(gdn, eval_pgs, train_pgs):
        assert gdn.cp_size == 2
        live_feature_width = sum(gdn.in_proj_split_sections) // gdn.cp_size
        torch.split(torch.empty(live_feature_width), gdn.feat_dim_split)

    assert gdn.cp_size == 1
    assert gdn.config.context_parallel_size == 1
