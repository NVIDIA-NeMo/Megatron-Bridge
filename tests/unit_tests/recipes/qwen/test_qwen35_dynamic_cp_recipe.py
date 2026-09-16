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

"""Tests for the Qwen3.5-35B-A3B global-batch packing + dynamic CP SFT recipe."""

import pytest

from megatron.bridge.data.builders import GPTSFTDatasetConfig
from megatron.bridge.recipes.qwen.gb200 import qwen35
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


def test_dynamic_cp_sft_recipe_uses_real_data_and_dataset_driven_packing():
    cfg = qwen35.qwen35_text_35b_a3b_sft_8gpu_gb200_bf16_dynamic_cp_config()

    assert isinstance(cfg.dataset, GPTSFTDatasetConfig)
    assert cfg.dataset.hf_dataset is not None and cfg.dataset.hf_dataset.dataset_name == "coderforge"
    assert cfg.dataset.enable_global_batch_packing is True
    assert cfg.dataset.enable_offline_packing is False and cfg.dataset.enable_in_batch_packing is False
    assert cfg.dataset.dataloader_type == "single"
    assert cfg.dataset.fold_alignment_padding is True  # 256-wide heads train on FlashAttention only
    assert cfg.dataset.seq_length == 32768 == cfg.model.seq_length

    # The scheduler itself is selected at validation from the dataset switch.
    assert getattr(cfg.model, "sequence_packing_scheduler", None) is None
    assert cfg.model.dynamic_context_parallel is True
    assert cfg.model.min_dynamic_context_parallel_size == 1
    assert cfg.model.max_seqlen_per_dp_cp_rank == 32768 // 4
    assert cfg.model.context_parallel_size == 4 and cfg.model.expert_model_parallel_size == 8
    assert cfg.model.tensor_model_parallel_size == 1 and cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.calculate_per_token_loss is True
    assert cfg.ddp.average_in_collective is False
    assert cfg.train.micro_batch_size == 1
    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.model.recompute_granularity == "full"
    assert cfg.optimizer.use_precision_aware_optimizer is True
    assert cfg.validation.eval_iters == 0
