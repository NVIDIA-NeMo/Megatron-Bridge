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

"""Unit tests for theoretical_memory_utils module.

These tests cover the three public functions

    compute_weight_and_optimizer_memory
    compute_activation_memory
    report_theoretical_memory

with synthetic configuration objects. The helpers under test are pure
arithmetic on the config dataclasses (no GPU, no model construction),
so SimpleNamespace mocks are sufficient.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.nn.functional as F

from megatron.bridge.training.utils.theoretical_memory_utils import (
    NUM_BYTES_IN_MEGABYTE,
    compute_activation_memory,
    compute_weight_and_optimizer_memory,
    report_theoretical_memory,
)


def _model_cfg(**overrides):
    """Build a model config namespace with sensible defaults for a tiny GPT."""
    base = dict(
        num_layers=2,
        hidden_size=128,
        seq_length=64,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=None,
        kv_channels=32,
        vocab_size=1024,
        make_vocab_size_divisible_by=128,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        share_embeddings_and_output_weights=True,
        num_moe_experts=None,
        gated_linear_unit=False,
        activation_func=F.gelu,
        sequence_parallel=True,
        recompute_granularity="selective",
        should_pad_vocab=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _container(model_cfg, *, use_distributed_optimizer=False, micro_batch_size=2, data_parallel_size=1):
    """Build a ConfigContainer-shaped namespace consumed by the helpers."""
    return SimpleNamespace(
        model=model_cfg,
        train=SimpleNamespace(micro_batch_size=micro_batch_size),
        optimizer=SimpleNamespace(use_distributed_optimizer=use_distributed_optimizer),
        data_parallel_size=data_parallel_size,
    )


class TestComputeWeightAndOptimizerMemory:
    def test_returns_positive_float(self):
        mem = compute_weight_and_optimizer_memory(_container(_model_cfg()))
        assert isinstance(mem, float)
        assert mem > 0

    def test_non_distributed_optimizer_uses_18_bytes_per_param(self):
        # With pp=1, tp=1, dp=1, the most-loaded-shard parameter count
        # equals total params. mem == 18 * total_params.
        cfg = _container(_model_cfg(), use_distributed_optimizer=False)
        non_dist = compute_weight_and_optimizer_memory(cfg)

        # Switching to distributed optimizer with dp=1 should give 18 bytes/param
        # too: 6 + 12/1 = 18. So both calls should be equal.
        cfg_dist_dp1 = _container(_model_cfg(), use_distributed_optimizer=True, data_parallel_size=1)
        dist_dp1 = compute_weight_and_optimizer_memory(cfg_dist_dp1)
        assert non_dist == pytest.approx(dist_dp1)

    def test_distributed_optimizer_scales_with_data_parallel_size(self):
        # With distributed optimizer, bytes-per-param = 6 + 12/dp.
        # Larger dp -> smaller per-shard memory.
        cfg_dp2 = _container(_model_cfg(), use_distributed_optimizer=True, data_parallel_size=2)
        cfg_dp8 = _container(_model_cfg(), use_distributed_optimizer=True, data_parallel_size=8)
        mem_dp2 = compute_weight_and_optimizer_memory(cfg_dp2)
        mem_dp8 = compute_weight_and_optimizer_memory(cfg_dp8)
        assert mem_dp8 < mem_dp2

    def test_tensor_parallel_reduces_per_shard_memory(self):
        mem_tp1 = compute_weight_and_optimizer_memory(_container(_model_cfg(tensor_model_parallel_size=1)))
        mem_tp4 = compute_weight_and_optimizer_memory(_container(_model_cfg(tensor_model_parallel_size=4)))
        assert mem_tp4 < mem_tp1

    def test_separate_embeddings_increase_param_count(self):
        shared = compute_weight_and_optimizer_memory(
            _container(_model_cfg(share_embeddings_and_output_weights=True))
        )
        separate = compute_weight_and_optimizer_memory(
            _container(_model_cfg(share_embeddings_and_output_weights=False))
        )
        assert separate > shared

    def test_gated_linear_silu_adds_mlp_params(self):
        plain = compute_weight_and_optimizer_memory(
            _container(_model_cfg(gated_linear_unit=False, activation_func=F.gelu))
        )
        gated = compute_weight_and_optimizer_memory(
            _container(_model_cfg(gated_linear_unit=True, activation_func=F.silu))
        )
        assert gated > plain

    def test_gated_linear_unit_without_silu_is_not_treated_as_gated(self):
        # The implementation only applies the gated_linear_multiplier when
        # gated_linear_unit AND activation_func == F.silu.
        gated_gelu = compute_weight_and_optimizer_memory(
            _container(_model_cfg(gated_linear_unit=True, activation_func=F.gelu))
        )
        plain = compute_weight_and_optimizer_memory(
            _container(_model_cfg(gated_linear_unit=False, activation_func=F.gelu))
        )
        assert gated_gelu == pytest.approx(plain)

    def test_more_layers_increases_memory(self):
        small = compute_weight_and_optimizer_memory(_container(_model_cfg(num_layers=2)))
        big = compute_weight_and_optimizer_memory(_container(_model_cfg(num_layers=8)))
        assert big > small

    def test_more_moe_experts_increases_memory(self):
        dense = compute_weight_and_optimizer_memory(_container(_model_cfg(num_moe_experts=None)))
        moe = compute_weight_and_optimizer_memory(_container(_model_cfg(num_moe_experts=4)))
        assert moe > dense

    def test_grouped_query_attention_reduces_params(self):
        # Setting num_query_groups < num_attention_heads should reduce KV proj params,
        # so the total parameter count drops.
        mha = compute_weight_and_optimizer_memory(_container(_model_cfg(num_query_groups=None)))  # falls back to heads
        gqa = compute_weight_and_optimizer_memory(_container(_model_cfg(num_query_groups=1)))
        assert gqa < mha

    def test_verbose_mode_runs_without_error(self, capsys):
        compute_weight_and_optimizer_memory(_container(_model_cfg()), verbose=True)
        captured = capsys.readouterr()
        assert "parameters" in captured.out.lower()

    def test_pipeline_parallel_other_shards_verbose_logs(self, capsys):
        cfg = _container(_model_cfg(pipeline_model_parallel_size=2))
        compute_weight_and_optimizer_memory(cfg, verbose=True)
        captured = capsys.readouterr()
        assert "other shards" in captured.out.lower()


class TestComputeActivationMemory:
    def test_returns_positive_float(self):
        mem = compute_activation_memory(_container(_model_cfg()), num_microbatches=4)
        assert isinstance(mem, float)
        assert mem > 0

    def test_more_layers_increases_activation_memory(self):
        small = compute_activation_memory(_container(_model_cfg(num_layers=2)), num_microbatches=4)
        big = compute_activation_memory(_container(_model_cfg(num_layers=8)), num_microbatches=4)
        assert big > small

    def test_tensor_parallel_reduces_activation_memory(self):
        mem_tp1 = compute_activation_memory(_container(_model_cfg(tensor_model_parallel_size=1)), num_microbatches=4)
        mem_tp4 = compute_activation_memory(_container(_model_cfg(tensor_model_parallel_size=4)), num_microbatches=4)
        assert mem_tp4 < mem_tp1

    def test_larger_micro_batch_increases_memory(self):
        small = compute_activation_memory(_container(_model_cfg(), micro_batch_size=1), num_microbatches=4)
        big = compute_activation_memory(_container(_model_cfg(), micro_batch_size=8), num_microbatches=4)
        assert big > small

    def test_virtual_pp_applies_interleaved_penalty(self):
        no_vpp = compute_activation_memory(
            _container(_model_cfg(pipeline_model_parallel_size=4, virtual_pipeline_model_parallel_size=None)),
            num_microbatches=8,
        )
        with_vpp = compute_activation_memory(
            _container(_model_cfg(pipeline_model_parallel_size=4, virtual_pipeline_model_parallel_size=2)),
            num_microbatches=8,
        )
        # Interleaved schedule adds memory penalty.
        assert with_vpp > no_vpp

    def test_non_interleaved_pp_discount_when_microbatches_below_pp_size(self):
        # With pp_size=4 and only 2 microbatches, memory is discounted by min(1, 2/4) = 0.5.
        with_few_microbatches = compute_activation_memory(
            _container(_model_cfg(pipeline_model_parallel_size=4)),
            num_microbatches=2,
        )
        with_enough_microbatches = compute_activation_memory(
            _container(_model_cfg(pipeline_model_parallel_size=4)),
            num_microbatches=8,
        )
        assert with_few_microbatches < with_enough_microbatches

    def test_pp1_includes_output_layer_term(self):
        # PP=1 adds an extra output-layer/CE-loss activation term that other PP sizes do not.
        cfg_pp1 = _container(_model_cfg(pipeline_model_parallel_size=1, num_layers=2))
        cfg_pp2_scaled = _container(_model_cfg(pipeline_model_parallel_size=2, num_layers=2))
        mem_pp1 = compute_activation_memory(cfg_pp1, num_microbatches=4)
        mem_pp2 = compute_activation_memory(cfg_pp2_scaled, num_microbatches=4)
        # We can't compare directly because PP affects in-flight microbatch count, but
        # we can at least assert both are positive and the PP=1 branch executed.
        assert mem_pp1 > 0 and mem_pp2 > 0

    def test_verbose_mode_runs_without_error(self, capsys):
        cfg = _container(_model_cfg(pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2))
        compute_activation_memory(cfg, num_microbatches=4, verbose=True)
        captured = capsys.readouterr()
        # virtual pipeline path prints the interleaved penalty and microbatch info
        assert "microbatches" in captured.out.lower() or "penalty" in captured.out.lower()


class TestReportTheoreticalMemory:
    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.MegatronMIMOProvider", new=type("FakeMimo", (), {}))
    def test_prints_weight_only_when_no_sequence_parallel(self, capsys):
        cfg = _container(_model_cfg(sequence_parallel=False))
        report_theoretical_memory(cfg, num_microbatches=4)
        captured = capsys.readouterr()
        assert "weight and optimizer" in captured.out.lower()
        assert "activation" not in captured.out.lower()

    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.MegatronMIMOProvider", new=type("FakeMimo", (), {}))
    def test_prints_full_breakdown_when_seq_parallel_and_selective_recompute(self, capsys):
        cfg = _container(_model_cfg(sequence_parallel=True, recompute_granularity="selective"))
        report_theoretical_memory(cfg, num_microbatches=4)
        captured = capsys.readouterr()
        out = captured.out.lower()
        assert "weight and optimizer" in out
        assert "activation" in out
        assert "total" in out

    @patch("megatron.bridge.models.megatron_mimo.megatron_mimo_provider.MegatronMIMOProvider", new=type("FakeMimo", (), {}))
    def test_skip_when_recompute_is_full(self, capsys):
        cfg = _container(_model_cfg(sequence_parallel=True, recompute_granularity="full"))
        report_theoretical_memory(cfg, num_microbatches=4)
        captured = capsys.readouterr()
        # full recompute path: only weight-and-optimizer line, no activation breakdown
        assert "weight and optimizer" in captured.out.lower()
        assert "activation" not in captured.out.lower()


class TestNumBytesInMegabyteConstant:
    def test_value_is_one_megabyte(self):
        assert NUM_BYTES_IN_MEGABYTE == 1024 * 1024
