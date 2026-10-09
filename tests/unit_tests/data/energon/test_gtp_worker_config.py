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
from unittest.mock import Mock, patch

import pytest

from megatron.bridge.data.energon.base_energon_datamodule import EnergonMultiModalDataModule


pytestmark = pytest.mark.unit


def _group(rank: int, size: int) -> Mock:
    group = Mock()
    group.rank.return_value = rank
    group.size.return_value = size
    return group


@pytest.mark.parametrize("split", ["train", "val"])
def test_gtp_workers_distinguish_remat_peers_and_share_context_parallel_samples(split):
    configs = {}
    for cp_rank in range(2):
        for data_rank in range(4):
            full_group = _group(data_rank, 4)
            pg = SimpleNamespace(
                dp=_group(data_rank // 2, 2),
                cp=_group(cp_rank, 2),
                gtp_remat=_group(data_rank % 2, 2),
            )
            module = EnergonMultiModalDataModule(path="unused", tokenizer=None, pg_collection=pg)
            with patch("megatron.bridge.training.gtp.parallel_state.get_data_parallel_group", return_value=full_group):
                config = module._build_worker_config(0, split=split)
            assert config.data_parallel_group is full_group
            assert config.world_size == 4
            configs[cp_rank, data_rank] = config

    assert {configs[0, rank].rank for rank in range(4)} == {0, 1, 2, 3}
    for data_rank in range(4):
        assert configs[0, data_rank].rank == configs[1, data_rank].rank == data_rank


def test_expert_only_gtp_already_uses_dense_data_parallel_workers():
    dp_group = _group(3, 4)
    pg = SimpleNamespace(dp=dp_group, cp=_group(1, 2), gtp_remat=None, expt_gtp_remat=_group(1, 2))
    module = EnergonMultiModalDataModule(path="unused", tokenizer=None, pg_collection=pg)
    with patch("megatron.bridge.training.gtp.parallel_state.get_data_parallel_group") as global_group:
        config = module._build_worker_config(0)
    assert config.rank == 3
    assert config.world_size == 4
    assert config.data_parallel_group is dp_group
    global_group.assert_not_called()
