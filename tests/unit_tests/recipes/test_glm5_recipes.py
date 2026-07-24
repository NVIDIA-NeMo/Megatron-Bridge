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
"""Unit tests for functional GLM-5.2 recipes."""

from types import SimpleNamespace

import pytest

from megatron.bridge.recipes.glm.h100 import glm5


pytestmark = pytest.mark.unit


class _FakeAutoBridge:
    @classmethod
    def from_hf_pretrained(cls, model_id: str, revision: str) -> "_FakeAutoBridge":
        assert model_id == "zai-org/GLM-5.2"
        assert len(revision) == 40
        return cls()

    def to_megatron_provider(self, load_weights: bool = False) -> SimpleNamespace:
        assert load_weights is False
        return SimpleNamespace(
            dsa_indexer_loss_coeff=0.001,
            dsa_indexer_use_sparse_loss=True,
            dsa_indexer_topk_freq=4,
            dsa_indexer_skip_topk_offset=3,
            mtp_num_layers=None,
        )


@pytest.fixture(autouse=True)
def _patch_autobridge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(glm5, "AutoBridge", _FakeAutoBridge)


@pytest.mark.parametrize(
    ("recipe", "world_size", "pp", "cp", "ep", "gbs", "steps"),
    [
        (glm5.glm52_pretrain_416gpu_h100_bf16_config, 416, 13, 1, 32, 1024, 100),
        (glm5.glm52_sft_functional_416gpu_h100_bf16_config, 416, 13, 16, 32, 32, 100),
        (glm5.glm52_sft_long_context_608gpu_h100_bf16_config, 608, 19, 32, 32, 13, 20),
        (glm5.glm52_peft_208gpu_h100_bf16_config, 208, 13, 1, 16, 32, 100),
    ],
)
def test_glm52_functional_recipe_topologies(recipe, world_size, pp, cp, ep, gbs, steps) -> None:
    cfg = recipe()

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == pp
    assert cfg.model.context_parallel_size == cp
    assert cfg.model.expert_model_parallel_size == ep
    assert cfg.train.global_batch_size == gbs
    assert cfg.train.micro_batch_size == 1
    assert cfg.train.train_iters == steps
    assert cfg.model.dsa_kernel_backend == "cudnn"
    assert cfg.model.mtp_num_layers == 1
    assert cfg.model.dsa_indexer_loss_coeff == 0.001
    assert cfg.model.dsa_indexer_use_sparse_loss is True
    assert cfg.model.moe_router_force_load_balancing is False
    assert world_size % (pp * cp) == 0
    assert world_size % (pp * ep) == 0


def test_glm52_long_context_recipe_uses_200k_packed_cp() -> None:
    cfg = glm5.glm52_sft_long_context_608gpu_h100_bf16_config()

    assert cfg.model.seq_length == 200000
    assert cfg.dataset.seq_length == 200000
    assert cfg.dataset.offline_packing_specs.packed_sequence_size == 200000
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 64
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.microbatch_group_size_per_vp_stage is None
    assert cfg.model.pipeline_model_parallel_layout == glm5._GLM52_PP19_LONG_CONTEXT_LAYOUT
    stages = cfg.model.pipeline_model_parallel_layout.split("|")
    assert [stage.count("t") for stage in stages] == [6] + [4] * 18
    decoder_starts = []
    decoder_count = 0
    for stage in stages:
        decoder_starts.append(decoder_count)
        decoder_count += stage.count("t")
    assert decoder_starts == [0, *range(6, 78, 4)]
    assert cfg.dataset.dataset_root == "work/data/glm5-2/synthetic-200k"
    assert cfg.dataset.hf_dataset is None


def test_glm52_pretrain_uses_reference_gradient_path() -> None:
    cfg = glm5.glm52_pretrain_416gpu_h100_bf16_config()

    assert cfg.mixed_precision.grad_reduce_in_fp32 is False
    assert cfg.ddp.grad_reduce_in_fp32 is False
    assert cfg.ddp.average_in_collective is False
    assert cfg.optimizer.use_precision_aware_optimizer is True


def test_glm52_peft_targets_mla_attention_projections() -> None:
    cfg = glm5.glm52_peft_208gpu_h100_bf16_config()

    assert cfg.peft.dim == 8
    assert cfg.peft.alpha == 16
    assert cfg.peft.dropout == 0.0
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 4
    assert cfg.dataset.hf_dataset.split == "train[:10000]"
    assert cfg.peft.target_modules == [
        "linear_q_down_proj",
        "linear_q_up_proj",
        "linear_kv_down_proj",
        "linear_kv_up_proj",
        "linear_proj",
    ]


def test_glm52_recipes_are_exported() -> None:
    from megatron.bridge.recipes import glm as glm_recipes
    from megatron.bridge.recipes.glm import h100

    assert glm_recipes.glm52_pretrain_config is glm5.glm52_pretrain_416gpu_h100_bf16_config
    for recipe_name in glm5.__all__:
        assert getattr(h100, recipe_name) is getattr(glm5, recipe_name)
        assert recipe_name in h100.__all__
