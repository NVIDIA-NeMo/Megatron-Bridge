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

from megatron.bridge.training.config import MegatronMIMOFeatureConfig


@pytest.mark.unit
class TestMegatronMIMOFeatureConfig:
    def test_defaults_all_off(self):
        cfg = MegatronMIMOFeatureConfig()
        # scalable_dp (the read-sharding master switch) and sequence packing are off by default.
        assert cfg.pack_sequences_in_batch is False and cfg.scalable_dp is False
        # Reordering and its overlap default on, but are gated behind scalable_dp, so they no-op
        # until scalable_dp is enabled.
        assert cfg.intra_microbatch_reorder is True and cfg.overlap_intra_microbatch_reorder is True
        assert cfg.cost_linear_vit == 1.0 and cfg.cost_linear_lm == 0.0 and cfg.pad_token_id == 0
        cfg.finalize()  # default config is valid

    def test_finalize_rejects_negative_coefficient(self):
        with pytest.raises(ValueError, match="cost_linear_vit"):
            MegatronMIMOFeatureConfig(cost_linear_vit=-1.0).finalize()

    def test_finalize_rejects_negative_lm_coefficient(self):
        with pytest.raises(ValueError, match="cost_linear_lm"):
            MegatronMIMOFeatureConfig(cost_linear_lm=-1.0).finalize()

    def test_finalize_rejects_reorder_with_non_positive_cost(self):
        # scalable_dp + reordering (default on) need a non-degenerate cost; both coefficients zero is rejected.
        with pytest.raises(ValueError, match="cost_linear_vit"):
            MegatronMIMOFeatureConfig(scalable_dp=True, cost_linear_vit=0.0, cost_linear_lm=0.0).finalize()

    def test_finalize_accepts_reorder_with_positive_cost(self):
        MegatronMIMOFeatureConfig(scalable_dp=True, cost_linear_vit=2.0).finalize()

    def test_finalize_accepts_reorder_with_lm_only_cost(self):
        # A positive LM coefficient alone is a valid (non-degenerate) cost even with zero ViT weight.
        MegatronMIMOFeatureConfig(scalable_dp=True, cost_linear_vit=0.0, cost_linear_lm=1.0).finalize()

    def test_finalize_skips_cost_check_when_reorder_disabled(self):
        # Scalable DP without reordering never consults the cost, so an all-zero cost is allowed.
        MegatronMIMOFeatureConfig(
            scalable_dp=True, intra_microbatch_reorder=False, cost_linear_vit=0.0, cost_linear_lm=0.0
        ).finalize()

    def test_pad_token_id_field_default_zero(self):
        cfg = MegatronMIMOFeatureConfig()
        assert cfg.pad_token_id == 0
        cfg.finalize()

    def test_pad_token_id_field_accepts_nonzero(self):
        MegatronMIMOFeatureConfig(pad_token_id=248044).finalize()

    def test_pad_token_id_field_rejects_negative(self):
        with pytest.raises(ValueError, match="pad_token_id"):
            MegatronMIMOFeatureConfig(pad_token_id=-1).finalize()

    def test_reorder_window_size_defaults_to_one(self):
        cfg = MegatronMIMOFeatureConfig()
        assert cfg.reorder_window_size == 1  # default W=1 is byte-for-byte per-micro-batch behavior
        cfg.finalize()

    def test_reorder_window_size_accepts_window(self):
        # W > 1 (e.g. equal to the gradient-accumulation count) is valid.
        MegatronMIMOFeatureConfig(scalable_dp=True, reorder_window_size=8).finalize()

    def test_reorder_window_size_rejects_below_one(self):
        with pytest.raises(ValueError, match="reorder_window_size"):
            MegatronMIMOFeatureConfig(reorder_window_size=0).finalize()

    def test_reorder_window_size_wires_into_buffer(self):
        # End-to-end wiring: the config value must reach ReorderingBuffer.window_size exactly as the
        # example threads it (`window_size=cfg.reorder_window_size`). dp_size=1 keeps it PG-free.
        import torch

        from megatron.bridge.data.megatron_mimo.reorder_buffer import ReorderingBuffer

        cfg = MegatronMIMOFeatureConfig(scalable_dp=True, reorder_window_size=3)
        cfg.finalize()
        items = [{"input_ids": torch.tensor([[i]])} for i in range(7)]
        buf = ReorderingBuffer(
            iter(items),
            dp_rank=0,
            dp_size=1,
            n_groups=1,
            cost_of=lambda f: 0.0,
            dp_group_gloo=None,
            dp_group_nccl=None,
            overlap=cfg.overlap_intra_microbatch_reorder,
            window_size=cfg.reorder_window_size,
        )
        assert buf._window_size == 3
        assert [int(b["input_ids"].item()) for b in buf] == list(range(7))
