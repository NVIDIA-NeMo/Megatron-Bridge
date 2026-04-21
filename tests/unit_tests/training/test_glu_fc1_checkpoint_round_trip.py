# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SwiGLU fc1 checkpoint layout: contiguous -> load (interleave) -> save (de-interleave) -> contiguous."""

from unittest.mock import patch

import pytest
import torch

from megatron.bridge.training.checkpointing import _process_state_dict_for_glu_interleaving


MOE_FC1_KEY = "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight"
MOE_FC1_BIAS_KEY = "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.bias"
DENSE_FC1_KEY = "decoder.layers.0.mlp.linear_fc1.weight"
DENSE_FC1_BIAS_KEY = "decoder.layers.0.mlp.linear_fc1.bias"


@pytest.fixture
def patch_print_rank_0():
    with patch("megatron.bridge.training.checkpointing.print_rank_0"):
        yield


class TestCheckpointLoadSaveRoundTrip:
    """Contiguous checkpoint layout -> load (interleave) -> save (de-interleave) matches original tensors."""

    @pytest.mark.parametrize("interleave_size", [4, 8])
    def test_moe_state_dict_round_trip_recover_contiguous(self, interleave_size, patch_print_rank_0):
        """MoE fc1 weight + bias + unrelated tensor: full round-trip recovers originals."""
        w = torch.randn(2 * interleave_size * 4, 16)
        b = torch.randn(2 * interleave_size * 2)
        passthrough = torch.randn(3, 7)
        original = {
            MOE_FC1_KEY: w.clone(),
            MOE_FC1_BIAS_KEY: b.clone(),
            "decoder.layers.0.mlp.linear_fc2.weight": passthrough.clone(),
        }
        after_load = _process_state_dict_for_glu_interleaving(
            {k: v.clone() for k, v in original.items()}, interleave_size, interleave=True
        )
        after_save = _process_state_dict_for_glu_interleaving(after_load, interleave_size, interleave=False)
        assert torch.equal(after_save[MOE_FC1_KEY], original[MOE_FC1_KEY])
        assert torch.equal(after_save[MOE_FC1_BIAS_KEY], original[MOE_FC1_BIAS_KEY])
        assert torch.equal(
            after_save["decoder.layers.0.mlp.linear_fc2.weight"],
            original["decoder.layers.0.mlp.linear_fc2.weight"],
        )

    @pytest.mark.parametrize("interleave_size", [4, 8])
    def test_dense_state_dict_round_trip_with_fusion_env(self, interleave_size, monkeypatch, patch_print_rank_0):
        """Dense fc1 participates only with USE_ACT_FUSION_FOR_DENSE=1; round-trip recovers contiguous tensors."""
        monkeypatch.setenv("USE_ACT_FUSION_FOR_DENSE", "1")
        w = torch.randn(2 * interleave_size * 3, 8)
        b = torch.randn(2 * interleave_size * 5)
        original = {
            DENSE_FC1_KEY: w.clone(),
            DENSE_FC1_BIAS_KEY: b.clone(),
        }
        after_load = _process_state_dict_for_glu_interleaving(
            {k: v.clone() for k, v in original.items()}, interleave_size, interleave=True
        )
        after_save = _process_state_dict_for_glu_interleaving(after_load, interleave_size, interleave=False)
        assert torch.equal(after_save[DENSE_FC1_KEY], original[DENSE_FC1_KEY])
        assert torch.equal(after_save[DENSE_FC1_BIAS_KEY], original[DENSE_FC1_BIAS_KEY])
