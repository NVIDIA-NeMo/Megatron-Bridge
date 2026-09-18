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

"""Tests for the Qwen3.5-35B-A3B online sequence packing + dynamic CP recipe."""

import pytest

from megatron.bridge.data.builders.synthetic_varlen import SyntheticVarlenDatasetConfig
from megatron.bridge.recipes.qwen.gb200 import qwen35
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


def test_dynamic_cp_recipe_sets_scheduler_topology_and_data():
    cfg = qwen35.qwen35_text_35b_a3b_pretrain_8gpu_gb200_bf16_dynamic_cp_config()

    assert cfg.model.sequence_packing_scheduler == "default_dynamic_cp"
    assert cfg.model.dynamic_context_parallel is True
    assert cfg.model.min_dynamic_context_parallel_size == 1
    assert cfg.model.context_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.tensor_model_parallel_size == 1 and cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.seq_length == 32768
    assert cfg.model.max_seqlen_per_dp_cp_rank == 32768 // 4
    assert cfg.model.calculate_per_token_loss is True
    assert cfg.ddp.average_in_collective is False
    assert cfg.train.micro_batch_size == 1
    assert cfg.model.cuda_graph_impl == "none" and cfg.model.cuda_graph_modules is None
    assert cfg.model.moe_token_dispatcher_type == "alltoall"
    assert cfg.model.recompute_granularity == "full"
    assert cfg.optimizer.use_precision_aware_optimizer is True

    assert isinstance(cfg.dataset, SyntheticVarlenDatasetConfig)
    assert cfg.dataset.seq_length == 32768
    assert cfg.dataset.dataloader_type == "single"
    assert cfg.dataset.sequence_padding_multiple is None  # derived from the parallel layout at validation
    assert cfg.dataset.fold_padding_into_sequence is True  # 256-wide heads train on FlashAttention only
