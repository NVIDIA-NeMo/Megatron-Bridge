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

"""Tests for the independent perf-recipe feature toggles in ``perf_recipes._common``."""

from types import SimpleNamespace

import pytest

from megatron.bridge.perf_recipes._common import (
    _enable_cutedsl_fused_grouped_mlp,
    _enable_full_iteration_cuda_graph,
    _enable_moe_a2a_overlap,
)


pytestmark = pytest.mark.unit


def _make_cfg() -> SimpleNamespace:
    """Return a config with every toggled field at its disabled default."""
    return SimpleNamespace(
        model=SimpleNamespace(
            cuda_graph_impl="transformer_engine",
            cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
            use_te_rng_tracker=False,
            moe_pad_experts_for_cuda_graph_inference=False,
            moe_expert_rank_capacity_factor=None,
            moe_paged_stash=False,
            moe_paged_stash_buffer_size_factor_cuda=None,
            moe_paged_stash_buffer_size_factor_cpu=None,
            moe_shared_expert_overlap=True,
            use_transformer_engine_op_fuser=False,
            moe_mlp_glu_interleave_size=None,
            high_priority_a2a_comm_stream=False,
            moe_hybridep_num_sms_preprocessing=108,
        ),
        rng=SimpleNamespace(te_rng_tracker=False),
        rerun_state_machine=SimpleNamespace(check_for_nan_in_loss=True),
        comm_overlap=SimpleNamespace(overlap_moe_expert_parallel_comm=False, delay_wgrad_compute=False),
    )


def _joint_settings(cfg: SimpleNamespace) -> tuple[bool, int]:
    return cfg.model.high_priority_a2a_comm_stream, cfg.model.moe_hybridep_num_sms_preprocessing


def test_full_iteration_cuda_graph_only_touches_graph_and_moe_buffer_settings():
    cfg = _make_cfg()
    _enable_full_iteration_cuda_graph(cfg)

    assert cfg.model.cuda_graph_impl == "full_iteration"
    assert cfg.model.cuda_graph_scope == []
    assert cfg.model.use_te_rng_tracker and cfg.rng.te_rng_tracker
    assert cfg.rerun_state_machine.check_for_nan_in_loss is False
    assert cfg.model.moe_pad_experts_for_cuda_graph_inference is True
    assert cfg.model.moe_expert_rank_capacity_factor == 1.5
    assert cfg.model.moe_paged_stash is True
    assert cfg.model.moe_paged_stash_buffer_size_factor_cuda == 1.2
    assert cfg.model.moe_paged_stash_buffer_size_factor_cpu == 1.0

    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is False
    assert cfg.model.use_transformer_engine_op_fuser is False
    assert _joint_settings(cfg) == (False, 108)


def test_moe_a2a_overlap_alone_leaves_graph_and_fused_mlp_settings_unchanged():
    cfg = _make_cfg()
    _enable_moe_a2a_overlap(cfg)

    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is True
    assert cfg.comm_overlap.delay_wgrad_compute is True
    assert cfg.model.moe_shared_expert_overlap is False

    assert cfg.model.cuda_graph_impl == "transformer_engine"
    assert cfg.model.use_transformer_engine_op_fuser is False
    assert _joint_settings(cfg) == (False, 108)


def test_cutedsl_fused_grouped_mlp_alone_leaves_a2a_settings_unchanged():
    cfg = _make_cfg()
    _enable_cutedsl_fused_grouped_mlp(cfg)

    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.model.moe_mlp_glu_interleave_size == 32

    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is False
    assert cfg.model.cuda_graph_impl == "transformer_engine"
    assert _joint_settings(cfg) == (False, 108)


def test_cutedsl_fused_grouped_mlp_without_comm_overlap_config():
    cfg = _make_cfg()
    cfg.comm_overlap = None
    _enable_cutedsl_fused_grouped_mlp(cfg)

    assert cfg.model.use_transformer_engine_op_fuser is True
    assert _joint_settings(cfg) == (False, 108)


@pytest.mark.parametrize(
    "toggles",
    [
        (_enable_moe_a2a_overlap, _enable_cutedsl_fused_grouped_mlp),
        (_enable_cutedsl_fused_grouped_mlp, _enable_moe_a2a_overlap),
    ],
    ids=["a2a_then_cutedsl", "cutedsl_then_a2a"],
)
def test_cutedsl_with_moe_a2a_overlap_applies_joint_settings_in_either_order(toggles):
    cfg = _make_cfg()
    for toggle in toggles:
        toggle(cfg)

    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is True
    assert _joint_settings(cfg) == (True, 32)
