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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.bridge.training.checkpointing import _process_state_dict_for_glu_interleaving


_CKPT_MOD = "megatron.bridge.training.checkpointing"

MOE_FC1_KEY = "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight"
MOE_FC1_BIAS_KEY = "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.bias"
DENSE_FC1_KEY = "decoder.layers.0.mlp.linear_fc1.weight"
DENSE_FC1_BIAS_KEY = "decoder.layers.0.mlp.linear_fc1.bias"


@pytest.fixture
def patch_print_rank_0():
    with patch(f"{_CKPT_MOD}.print_rank_0"):
        yield


# ---------------------------------------------------------------------------
# Lightweight DTensor fakes for the Megatron-FSDP code path.
# Simulates single-rank sharding where the local shard IS the full tensor.
# ---------------------------------------------------------------------------


class _ChunkMeta:
    """Mimics chunk metadata returned by DTensor internals."""

    __slots__ = ("offsets", "sizes")

    def __init__(self, offsets: tuple[int, ...], sizes: tuple[int, ...]):
        self.offsets = offsets
        self.sizes = sizes


class _FakeInnerTensor:
    """Stand-in for ``DTensor._local_tensor`` with ``__create_chunk_list__``."""

    def __init__(self, data: torch.Tensor):
        self._data = data

    def __create_chunk_list__(self):
        return [
            _ChunkMeta(
                # Trivial global DTensor chunk metadata, i.e. zero offsets.
                offsets=tuple(0 for _ in self._data.shape),
                sizes=tuple(self._data.shape),
            )
        ]


class _FakeDTensor:
    """Single-rank DTensor stub (local shard == full tensor)."""

    def __init__(self, data: torch.Tensor):
        self._local_tensor = _FakeInnerTensor(data)
        self.device_mesh = None
        self.placements = None
        self._shape = data.shape
        self._stride = data.stride()

    @property
    def shape(self):
        return self._shape

    def stride(self):
        return self._stride

    @staticmethod
    def from_local(local_tensor, *, device_mesh, placements, shape, stride):
        return _FakeDTensor(local_tensor)


def _mock_gather(dtensor):
    """Single-rank gather: local shard is the full tensor."""
    return SimpleNamespace(_local_tensor=dtensor._local_tensor._data)


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


class TestMegatronFSDPCheckpointRoundTrip:
    """
    Megatron-FSDP DTensor path: contiguous → interleave → de-interleave recovers originals.

    NOTE(@cspades): These do NOT test DTensor or Megatron-FSDP un-even sharded gather.
    This only tests the non-distributed logic in _process_state_dict_for_glu_interleaving.
    """

    @pytest.fixture(autouse=True)
    def _fsdp_mocks(self, patch_print_rank_0):
        with (
            patch(f"{_CKPT_MOD}.preprocess_state_dict_for_uneven_dtensor", side_effect=lambda d: d),
            patch(f"{_CKPT_MOD}.gather_uneven_dtensor_to_full_tensor", side_effect=_mock_gather),
            patch(f"{_CKPT_MOD}.DTensor", _FakeDTensor),
        ):
            yield

    @pytest.mark.parametrize("interleave_size", [4, 8])
    def test_moe_fsdp_round_trip_recovers_contiguous(self, interleave_size):
        """MoE fc1 weight + bias + passthrough: FSDP DTensor round-trip recovers contiguous originals."""
        w = torch.randn(2 * interleave_size * 4, 16)
        b = torch.randn(2 * interleave_size * 2)
        passthrough = torch.randn(3, 7)
        original_w, original_b, original_pt = w.clone(), b.clone(), passthrough.clone()

        state = {
            MOE_FC1_KEY: _FakeDTensor(w),
            MOE_FC1_BIAS_KEY: _FakeDTensor(b),
            "decoder.layers.0.mlp.linear_fc2.weight": passthrough,
        }

        after_load = _process_state_dict_for_glu_interleaving(
            state,
            interleave_size,
            interleave=True,
            use_megatron_fsdp=True,
        )
        assert not torch.equal(after_load[MOE_FC1_KEY]._local_tensor._data, original_w)

        after_save = _process_state_dict_for_glu_interleaving(
            after_load,
            interleave_size,
            interleave=False,
            use_megatron_fsdp=True,
        )
        assert torch.equal(after_save[MOE_FC1_KEY]._local_tensor._data, original_w)
        assert torch.equal(after_save[MOE_FC1_BIAS_KEY]._local_tensor._data, original_b)
        assert torch.equal(
            after_save["decoder.layers.0.mlp.linear_fc2.weight"],
            original_pt,
        )

    @pytest.mark.parametrize("interleave_size", [4, 8])
    def test_dense_fsdp_round_trip_with_fusion_env(self, interleave_size, monkeypatch):
        """Dense fc1 with USE_ACT_FUSION_FOR_DENSE=1: FSDP DTensor round-trip recovers contiguous tensors."""
        monkeypatch.setenv("USE_ACT_FUSION_FOR_DENSE", "1")
        w = torch.randn(2 * interleave_size * 3, 8)
        b = torch.randn(2 * interleave_size * 5)
        original_w, original_b = w.clone(), b.clone()

        state = {
            DENSE_FC1_KEY: _FakeDTensor(w),
            DENSE_FC1_BIAS_KEY: _FakeDTensor(b),
        }

        after_load = _process_state_dict_for_glu_interleaving(
            state,
            interleave_size,
            interleave=True,
            use_megatron_fsdp=True,
        )
        after_save = _process_state_dict_for_glu_interleaving(
            after_load,
            interleave_size,
            interleave=False,
            use_megatron_fsdp=True,
        )
        assert torch.equal(after_save[DENSE_FC1_KEY]._local_tensor._data, original_w)
        assert torch.equal(after_save[DENSE_FC1_BIAS_KEY]._local_tensor._data, original_b)
