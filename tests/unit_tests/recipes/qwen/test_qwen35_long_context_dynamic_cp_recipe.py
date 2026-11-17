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

"""Tests for the eight-GPU GB300 Qwen3.5-35B-A3B 128K dynamic-CP recipes on CoderForge."""

import pytest
import torch
from transformers import Qwen3_5MoeConfig

from megatron.bridge.data.builders import GPTSFTDatasetConfig
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.recipes.qwen.gb200 import qwen35 as gb200_qwen35
from megatron.bridge.recipes.qwen.gb300 import qwen35
from tests.unit_tests.recipes.qwen.test_qwen35_dynamic_cp_mock_recipe import _TEXT_CONFIG
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.fixture
def requested(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Build recipes through the real Qwen3.5 bridge on the published 35B-A3B config, offline."""
    patch_recipe_construction_dependencies(monkeypatch)
    monkeypatch.setattr(gb200_qwen35, "AutoBridge", AutoBridge)
    requested = {}

    def from_pretrained(path, **kwargs):
        requested.update(path=path, **kwargs)
        return Qwen3_5MoeConfig(text_config=_TEXT_CONFIG, tie_word_embeddings=False)

    monkeypatch.setattr(gb200_qwen35.AutoConfig, "from_pretrained", from_pretrained)
    return requested


@pytest.fixture
def online(requested):
    return qwen35.qwen35_text_35b_a3b_sft_long_context_8gpu_gb300_fp8mx_dynamic_cp_config()


@pytest.fixture
def offline(requested):
    return qwen35.qwen35_text_35b_a3b_sft_long_context_8gpu_gb300_fp8mx_offline_dynamic_cp_config()


def test_model_and_tokenizer_share_the_pinned_revision(online, requested):
    revision = gb200_qwen35._QWEN35_35B_A3B_INSTRUCT_REVISION

    assert requested == {"path": "Qwen/Qwen3.5-35B-A3B", "revision": revision}
    assert online.tokenizer.tokenizer_model == "Qwen/Qwen3.5-35B-A3B"
    assert online.tokenizer.hf_tokenizer_kwargs == {"revision": revision}
    assert (online.model.num_layers, online.model.num_moe_experts, online.model.mtp_num_layers) == (40, 256, 1)


def test_128k_context_runs_on_a_cp8_pool_with_16k_tokens_per_rank(online):
    model = online.model

    assert model.seq_length == 131072 == online.dataset.seq_length
    assert (model.tensor_model_parallel_size, model.pipeline_model_parallel_size) == (1, 1)
    assert (model.context_parallel_size, model.expert_model_parallel_size, model.expert_tensor_parallel_size) == (
        8,
        8,
        1,
    )
    assert model.max_seqlen_per_dp_cp_rank == 16384
    assert model.dynamic_context_parallel is True
    assert model.min_dynamic_context_parallel_size == 1
    # The scheduler is selected at validation from the dataset switch.
    assert getattr(model, "sequence_packing_scheduler", None) is None
    assert model.calculate_per_token_loss is True
    assert online.ddp.average_in_collective is False
    assert (online.train.micro_batch_size, online.train.global_batch_size) == (1, 16)
    assert online.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 8


def test_data_is_coderforge_with_global_batch_packing(online):
    dataset = online.dataset

    assert isinstance(dataset, GPTSFTDatasetConfig)
    assert dataset.hf_dataset is not None and dataset.hf_dataset.dataset_name == "coderforge"
    assert dataset.hf_dataset.load_kwargs == {"revision": gb200_qwen35._CODERFORGE_REVISION}
    assert dataset.enable_global_batch_packing is True and dataset.yields_unpacked_samples is True
    assert dataset.enable_offline_packing is False and dataset.enable_in_batch_packing is False
    assert dataset.dataloader_type == "single"
    assert dataset.fold_alignment_padding is True  # 256-wide heads train on FlashAttention only
    # Lazily tokenized long trajectories need enough loader workers to keep the GPUs busy.
    assert dataset.num_workers == 16
    assert dataset.do_validation is False and online.validation.eval_iters == 0
    dataset.validate()


def test_performance_settings_match_the_mock_data_recipe(online):
    model = online.model

    assert online.mixed_precision.fp8_recipe == "mxfp8"
    assert online.optimizer.optimizer_cpu_offload is True and online.optimizer.optimizer_offload_fraction == 1.0
    for name in ("main_params_dtype", "main_grads_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
        assert getattr(online.optimizer, name) == torch.float32
    assert model.moe_token_dispatcher_type == "flex" and model.moe_flex_dispatcher_backend == "hybridep"
    assert (model.recompute_granularity, model.recompute_method, model.recompute_num_layers) == ("full", "uniform", 1)
    assert online.train.cuda_memory_fraction == 0.9
    assert online.checkpoint.pretrained_checkpoint is None


def test_offline_variant_packs_128k_rows_and_counts_the_global_batch_in_rows(online, offline):
    dataset = offline.dataset

    assert dataset.enable_offline_packing is True and dataset.enable_global_batch_packing is True
    specs = dataset.offline_packing_specs
    # Aligned to 2 x CP8, so a longer trajectory is truncated to the row instead of dropped.
    assert (specs.packed_sequence_size, specs.pad_seq_to_mult, specs.pad_cu_seqlens) == (131072, 16, False)
    assert offline.train.global_batch_size == 8
    dataset.validate()
    # Everything else matches the online recipe: the same trajectories, model and dynamic CP.
    assert dataset.hf_dataset == online.dataset.hf_dataset
    assert dataset.fold_alignment_padding is True and dataset.dataloader_type == "single"
    assert offline.model.dynamic_context_parallel is True
    assert offline.model.max_seqlen_per_dp_cp_rank == online.model.max_seqlen_per_dp_cp_rank


def test_common_builder_rejects_a_context_the_cp_group_cannot_split(requested):
    with pytest.raises(ValueError, match="multiple of context_parallel_size"):
        gb200_qwen35.qwen35_text_35b_a3b_fp8mx_dynamic_cp_common(
            seq_length=65537, context_parallel_size=4, expert_model_parallel_size=4
        )
