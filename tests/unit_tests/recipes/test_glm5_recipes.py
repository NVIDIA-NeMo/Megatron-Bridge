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
"""Unit tests for GLM-5.2 recipes."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from megatron.bridge.data.builders import ChatSFTPreprocessingConfig
from megatron.bridge.recipes.glm import gb200
from megatron.bridge.recipes.glm.gb200 import glm5 as gb200_glm5
from megatron.bridge.recipes.glm.h100 import glm5


pytestmark = pytest.mark.unit


class _FakeMegatronProvider(SimpleNamespace):
    _KNOWN_FIELDS = {
        "account_for_embedding_in_pipeline_split",
        "account_for_loss_in_pipeline_split",
        "apply_rope_fusion",
        "bias_activation_fusion",
        "bias_dropout_fusion",
        "calculate_per_token_loss",
        "context_parallel_size",
        "cross_entropy_fusion_impl",
        "cross_entropy_loss_fusion",
        "deallocate_pipeline_outputs",
        "dsa_indexer_loss_coeff",
        "dsa_indexer_skip_topk_offset",
        "dsa_indexer_topk_freq",
        "dsa_indexer_use_sparse_loss",
        "dsa_kernel_backend",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "gradient_accumulation_fusion",
        "microbatch_group_size_per_vp_stage",
        "moe_flex_dispatcher_backend",
        "moe_grouped_gemm",
        "moe_permute_fusion",
        "moe_router_force_load_balancing",
        "moe_shared_expert_overlap",
        "moe_token_dispatcher_type",
        "mtp_num_layers",
        "num_layers_in_first_pipeline_stage",
        "num_layers_in_last_pipeline_stage",
        "persist_layer_norm",
        "pipeline_model_parallel_layout",
        "pipeline_model_parallel_size",
        "recompute_granularity",
        "recompute_method",
        "recompute_num_layers",
        "seq_length",
        "sequence_parallel",
        "tensor_model_parallel_size",
        "virtual_pipeline_model_parallel_size",
    }

    def __init__(self, **kwargs: object) -> None:
        unknown_fields = set(kwargs) - self._KNOWN_FIELDS
        if unknown_fields:
            raise AssertionError(f"Unexpected GLM-5.2 model config fields: {sorted(unknown_fields)}")
        super().__init__(**kwargs)

    def __setattr__(self, name: str, value: object) -> None:
        if name not in self._KNOWN_FIELDS:
            raise AssertionError(f"Unexpected GLM-5.2 model config field: {name}")
        super().__setattr__(name, value)


class _FakeAutoBridge:
    @classmethod
    def from_hf_pretrained(cls, model_id: str, revision: str) -> "_FakeAutoBridge":
        assert model_id == "zai-org/GLM-5.2"
        assert len(revision) == 40
        return cls()

    def to_megatron_provider(self, load_weights: bool = False) -> _FakeMegatronProvider:
        assert load_weights is False
        return _FakeMegatronProvider(
            dsa_indexer_loss_coeff=0.001,
            dsa_indexer_use_sparse_loss=True,
            dsa_indexer_topk_freq=4,
            dsa_indexer_skip_topk_offset=3,
            mtp_num_layers=None,
        )


@pytest.fixture(autouse=True)
def _patch_autobridge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gb200_glm5, "AutoBridge", _FakeAutoBridge)
    monkeypatch.setattr(glm5, "AutoBridge", _FakeAutoBridge)


@pytest.mark.parametrize(
    ("recipe", "world_size", "pp", "cp", "ep", "gbs", "steps"),
    [
        (glm5.glm52_pretrain_416gpu_h100_bf16_config, 416, 13, 1, 32, 1024, 100),
        (glm5.glm52_sft_416gpu_h100_bf16_config, 416, 13, 16, 32, 32, 100),
        (glm5.glm52_sft_608gpu_h100_bf16_200k_config, 608, 19, 32, 32, 13, 20),
        (glm5.glm52_peft_208gpu_h100_bf16_config, 208, 13, 1, 16, 32, 100),
    ],
)
def test_glm52_h100_recipe_topologies(recipe, world_size, pp, cp, ep, gbs, steps) -> None:
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


def test_glm52_h100_200k_recipe_uses_packed_cp() -> None:
    cfg = glm5.glm52_sft_608gpu_h100_bf16_200k_config()

    assert cfg.model.seq_length == 200000
    assert cfg.dataset.seq_length == 200000
    assert cfg.dataset.offline_packing_specs.packed_sequence_size == 200000
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 64
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.microbatch_group_size_per_vp_stage is None
    assert cfg.model.pipeline_model_parallel_layout == glm5._GLM52_PP19_200K_LAYOUT
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


@pytest.mark.parametrize(
    ("recipe", "cp", "gbs", "steps", "dispatcher", "backend"),
    [
        (gb200.glm52_pretrain_192gpu_gb200_bf16_config, 1, 1024, 100, "alltoall", None),
        (gb200.glm52_sft_192gpu_gb200_bf16_config, 4, 8, 100, "flex", "hybridep"),
        (gb200.glm52_peft_192gpu_gb200_bf16_config, 1, 32, 100, "alltoall", None),
    ],
)
def test_glm52_gb200_recipe_topologies(recipe, cp, gbs, steps, dispatcher, backend) -> None:
    cfg = recipe()

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 6
    assert cfg.model.context_parallel_size == cp
    assert cfg.model.expert_model_parallel_size == 32
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.model.virtual_pipeline_model_parallel_size is None
    assert cfg.model.pipeline_model_parallel_layout is None
    assert cfg.model.num_layers_in_first_pipeline_stage == 14
    assert cfg.model.num_layers_in_last_pipeline_stage == 16
    assert cfg.model.microbatch_group_size_per_vp_stage == 6
    assert cfg.model.moe_token_dispatcher_type == dispatcher
    assert cfg.model.moe_flex_dispatcher_backend == backend
    assert cfg.model.apply_rope_fusion is False
    assert cfg.train.global_batch_size == gbs
    assert cfg.train.micro_batch_size == 1
    assert cfg.train.train_iters == steps
    data_parallel_size = 192 // (cfg.model.pipeline_model_parallel_size * cfg.model.context_parallel_size)
    assert cfg.train.global_batch_size % data_parallel_size == 0


def test_glm52_gb200_128k_recipe_uses_packed_cp() -> None:
    cfg = gb200.glm52_sft_192gpu_gb200_bf16_128k_config()

    assert cfg.model.seq_length == 131072
    assert cfg.model.pipeline_model_parallel_size == 6
    assert cfg.model.context_parallel_size == 32
    assert cfg.model.expert_model_parallel_size == 32
    assert cfg.model.pipeline_model_parallel_layout == gb200_glm5._GLM52_PP6_128K_LAYOUT
    assert cfg.model.num_layers_in_first_pipeline_stage is None
    assert cfg.model.num_layers_in_last_pipeline_stage is None
    stages = cfg.model.pipeline_model_parallel_layout.split("|")
    assert [stage.count("t") for stage in stages] == [14, 16, 12, 12, 12, 12]
    decoder_starts = []
    decoder_count = 0
    for stage in stages:
        decoder_starts.append(decoder_count)
        decoder_count += stage.count("t")
    assert decoder_starts == [0, 14, 30, 42, 54, 66]
    assert all((start + 1 - 3) % 4 == 0 for start in decoder_starts[1:])
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_shared_expert_overlap is False
    assert cfg.model.apply_rope_fusion is False
    assert cfg.train.global_batch_size == 56
    assert cfg.train.micro_batch_size == 1
    assert cfg.train.train_iters == 20
    assert cfg.dataset.seq_length == 131072
    assert cfg.dataset.hf_dataset is None
    assert cfg.dataset.dataset_root == "work/data/glm5-2/synthetic-long-sft-128k"
    assert cfg.dataset.hf_output_root is None
    assert cfg.dataset.max_train_samples == 1120
    assert isinstance(cfg.dataset.preprocessing, ChatSFTPreprocessingConfig)
    assert cfg.dataset.offline_packing_specs.packed_sequence_size == 131072
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 64
    assert cfg.dataset.offline_packing_specs.pad_cu_seqlens is False
    assert cfg.dataset.dataset_kwargs == {"pad_to_max_length": True}
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 32
    assert cfg.env_vars["NUM_OF_TOKENS_PER_CHUNK_COMBINE_API"] == 128
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 72
    assert cfg.env_vars["USE_MNNVL"] == 1


def test_glm52_gb200_sft_uses_8k_packed_tulu3() -> None:
    cfg = gb200.glm52_sft_192gpu_gb200_bf16_config()

    assert cfg.model.seq_length == 8192
    assert cfg.model.context_parallel_size == 4
    assert cfg.dataset.seq_length == 8192
    assert cfg.train.global_batch_size == 8
    assert cfg.train.micro_batch_size == 1
    assert cfg.dataset.hf_dataset.split == "train[:10000]"
    assert cfg.dataset.hf_dataset.load_kwargs == {"revision": gb200_glm5._TULU3_REVISION}
    assert cfg.dataset.hf_output_root == "work/data/glm5-2/tulu3-full-sft-gb200-8k-v5"
    assert cfg.dataset.offline_packing_specs.packed_sequence_size == 8192
    assert (
        cfg.dataset.offline_packing_specs.max_single_sequence_length
        == cfg.model.seq_length - cfg.dataset.offline_packing_specs.pad_seq_to_mult
    )
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 2 * cfg.model.context_parallel_size
    assert cfg.dataset.offline_packing_specs.pad_cu_seqlens is False
    assert cfg.dataset.dataset_kwargs == {"pad_to_max_length": True}
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"


def test_glm52_gb200_recipes_do_not_depend_on_h100_recipes() -> None:
    source = Path(gb200_glm5.__file__).read_text(encoding="utf-8")

    assert "megatron.bridge.recipes.glm.h100" not in source


@pytest.mark.parametrize(
    "recipe",
    [
        glm5.glm52_pretrain_416gpu_h100_bf16_config,
        glm5.glm52_sft_416gpu_h100_bf16_config,
    ],
)
def test_glm52_vpp2_default_layout_shape(recipe) -> None:
    cfg = recipe()

    assert cfg.model.virtual_pipeline_model_parallel_size == 2
    assert cfg.model.pipeline_model_parallel_layout == glm5._GLM52_VPP2_LAYOUT
    stages = cfg.model.pipeline_model_parallel_layout.split("|")
    assert len(stages) == cfg.model.pipeline_model_parallel_size * cfg.model.virtual_pipeline_model_parallel_size
    assert [stage.count("t") for stage in stages] == [
        1,
        1,
        0,
        4,
        4,
        4,
        4,
        0,
        4,
        4,
        4,
        4,
        0,
        4,
        4,
        4,
        4,
        0,
        4,
        4,
        4,
        4,
        0,
        4,
        4,
        4,
    ]
    assert sum(stage.count("t") for stage in stages) == 78
    assert cfg.model.pipeline_model_parallel_layout.count("E") == 1
    assert cfg.model.pipeline_model_parallel_layout.count("m") == 1
    assert cfg.model.pipeline_model_parallel_layout.count("L") == 1


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


def test_glm52_platform_recipes_are_exported() -> None:
    import megatron.bridge.recipes as recipes
    from megatron.bridge.recipes import glm as glm_recipes
    from megatron.bridge.recipes.glm import h100

    for recipe_name in gb200.__all__:
        assert getattr(glm_recipes, recipe_name) is getattr(gb200, recipe_name)
        assert getattr(recipes, recipe_name) is getattr(gb200, recipe_name)
        assert recipe_name in glm_recipes.__all__
    for recipe_name in glm5.__all__:
        assert getattr(h100, recipe_name) is getattr(glm5, recipe_name)
        assert getattr(recipes, recipe_name) is getattr(glm5, recipe_name)
        assert recipe_name in h100.__all__
